"""
External API Tools for AI Assistant
Provides function calling capabilities for weather, music, and other services.
"""
import os
import re
import json
import logging
import asyncio
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from functools import lru_cache

logger = logging.getLogger('Tools')

# Try to import optional dependencies
try:
    import httpx
    HTTPX_OK = True
except ImportError:
    HTTPX_OK = False
    logger.warning("httpx not available - weather and web APIs disabled")

try:
    import pyttsx3
    PYTTSX3_OK = True
except ImportError:
    PYTTSX3_OK = False
    logger.warning("pyttsx3 not available - local TTS disabled")


class ToolExecutor:
    """Executes external API calls and tools for the assistant."""
    
    def __init__(self):
        self.spotify_enabled = os.getenv("SPOTIFY_ENABLED", "false").lower() == "true"
        self.http_client = None
        
        # Tool definitions for LLM to understand what's available
        self.tools = {
            "get_weather": {
                "description": "Get current weather for a location",
                "parameters": {
                    "location": "City name or 'current' for user's location"
                },
                "keywords": ["weather", "temperature", "forecast", "rain", "sunny", "cold", "hot"]
            },
            "get_time": {
                "description": "Get current time and date",
                "parameters": {},
                "keywords": ["time", "date", "today", "day", "clock", "what time"]
            },
            "calculate": {
                "description": "Perform mathematical calculations",
                "parameters": {
                    "expression": "Math expression to evaluate"
                },
                "keywords": ["calculate", "math", "plus", "minus", "times", "divide", "equals"]
            },
            "play_music": {
                "description": "Play music or control playback",
                "parameters": {
                    "action": "play/pause/stop/next/previous",
                    "query": "Song or artist name (optional)"
                },
                "keywords": ["play", "music", "song", "pause", "stop", "next", "skip"]
            },
            "set_reminder": {
                "description": "Set a reminder for later",
                "parameters": {
                    "message": "Reminder message",
                    "minutes": "Minutes from now"
                },
                "keywords": ["remind", "reminder", "alert", "notify"]
            },
            "web_search": {
                "description": "Search the web for information",
                "parameters": {
                    "query": "Search query"
                },
                "keywords": ["search", "look up", "find", "google", "who is", "what is"]
            }
        }
    
    async def _ensure_client(self):
        """Ensure HTTP client is initialized."""
        if not HTTPX_OK:
            return False
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=10.0)
        return True
    
    async def close(self):
        """Close HTTP client."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
    
    def detect_tool_use(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Fast keyword-based detection of tool usage.
        Returns: {"tool": "tool_name", "params": {...}} or None
        """
        query_lower = query.lower()
        
        # Time/Date - check first to avoid conflicts with "what is"
        time_patterns = [
            r'\bwhat time\b',
            r'\btime is it\b',
            r'\bwhat\'?s the time\b',
            r'\bcurrent time\b',
            r'\bwhat date\b',
            r'\bwhat day\b',
            r'\btoday\'?s date\b'
        ]
        if any(re.search(p, query_lower) for p in time_patterns):
            return {"tool": "get_time", "params": {}}
        
        # Calculator - look for math expressions
        if re.search(r'\d+\s*[\+\-\*\/×÷]\s*\d+', query_lower):
            math_match = re.search(r'([\d\s\+\-\*\/\(\)\.×÷]+)', query_lower)
            if math_match:
                return {"tool": "calculate", "params": {"expression": math_match.group(1).strip()}}
        
        if 'calculate' in query_lower:
            # Extract everything after "calculate"
            calc_match = re.search(r'calculate\s+(.+)', query_lower)
            if calc_match:
                return {"tool": "calculate", "params": {"expression": calc_match.group(1).strip()}}
        
        # Weather - check for weather keywords
        weather_keywords = ['weather', 'temperature', 'forecast', 'degrees', 'cold', 'hot', 'rain', 'sunny', 'cloudy']
        if any(kw in query_lower for kw in weather_keywords):
            location = self._extract_location(query) or "Seoul"
            return {"tool": "get_weather", "params": {"location": location}}
        
        # Music control
        if 'music' in query_lower or 'song' in query_lower:
            action = "play"
            if "pause" in query_lower:
                action = "pause"
            elif "stop" in query_lower:
                action = "stop"
            elif "next" in query_lower or "skip" in query_lower:
                action = "next"
            elif "previous" in query_lower or "back" in query_lower:
                action = "previous"
            return {"tool": "play_music", "params": {"action": action, "query": ""}}
        
        # Reminders
        if 'remind' in query_lower:
            reminder_match = re.search(r'remind\s+(?:me\s+)?(?:to\s+)?(.+?)(?:\s+in\s+(\d+)\s+(?:minute|min)s?)?', query_lower)
            if reminder_match:
                message = reminder_match.group(1).strip()
                minutes = int(reminder_match.group(2)) if reminder_match.group(2) else 5
                return {"tool": "set_reminder", "params": {"message": message, "minutes": minutes}}
        
        # Web search - specific patterns only
        search_triggers = ['who is', 'what is', 'search for', 'look up', 'tell me about']
        for trigger in search_triggers:
            if trigger in query_lower:
                # Exclude if it's asking about time/date/weather
                if not any(kw in query_lower for kw in ['time', 'weather', 'temperature']):
                    # Extract the search query
                    parts = query_lower.split(trigger, 1)
                    if len(parts) > 1:
                        search_query = parts[1].strip()
                        return {"tool": "web_search", "params": {"query": search_query}}
        
        return None
    
    def _extract_location(self, query: str) -> Optional[str]:
        """Extract location from weather query."""
        query_lower = query.lower()
        
        # Common patterns: "weather in X", "weather at X", "X weather"
        patterns = [
            r'weather\s+in\s+([a-z\s]+?)(?:\s+today|\s+now|$|\?)',
            r'weather\s+at\s+([a-z\s]+?)(?:\s+today|\s+now|$|\?)',
            r'in\s+([a-z\s]+?)\s+weather',
            r'temperature\s+in\s+([a-z\s]+?)(?:\s+today|\s+now|$|\?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                location = match.group(1).strip()
                # Clean up common words
                location = re.sub(r'\b(the|today|now|there)\b', '', location).strip()
                if len(location) > 2:
                    return location.title()
        
        # If no specific location found, return None (will use Seoul default)
        return None
    
    async def execute(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool and return results.
        Returns: {"success": bool, "result": str, "data": Any}
        """
        try:
            logger.info(f"Executing tool: {tool_name} with params: {params}")
            
            if tool_name == "get_weather":
                return await self.get_weather(params.get("location", "current"))
            elif tool_name == "get_time":
                return await self.get_time()
            elif tool_name == "calculate":
                return await self.calculate(params.get("expression", ""))
            elif tool_name == "play_music":
                return await self.play_music(params.get("action", "play"), params.get("query", ""))
            elif tool_name == "set_reminder":
                return await self.set_reminder(params.get("message", ""), params.get("minutes", 5))
            elif tool_name == "web_search":
                return await self.web_search(params.get("query", ""))
            else:
                return {"success": False, "result": f"Unknown tool: {tool_name}", "data": None}
        
        except Exception as e:
            logger.error(f"Tool execution failed: {e}", exc_info=True)
            return {"success": False, "result": f"Error: {str(e)}", "data": None}
    
    async def get_weather(self, location: str = "Seoul") -> Dict[str, Any]:
        """Get weather information for a location using Open-Meteo (FREE, no API key needed!)."""
        if not await self._ensure_client():
            return {"success": False, "result": "Weather service unavailable", "data": None}
        
        try:
            # Use Seoul as default location
            if location == "current" or not location:
                location = "Seoul"
            
            # Step 1: Geocoding - Convert city name to coordinates
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
            geo_response = await self.http_client.get(geo_url)
            
            if geo_response.status_code != 200:
                return {
                    "success": False,
                    "result": f"Could not find location: {location}",
                    "data": None
                }
            
            geo_data = geo_response.json()
            
            if "results" not in geo_data or len(geo_data["results"]) == 0:
                return {
                    "success": False,
                    "result": f"City not found: {location}",
                    "data": None
                }
            
            # Get coordinates
            city_info = geo_data["results"][0]
            lat = city_info["latitude"]
            lon = city_info["longitude"]
            city_name = city_info["name"]
            country = city_info.get("country", "")
            
            # Step 2: Get weather data using coordinates
            weather_url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m"
                f"&timezone=auto"
            )
            
            weather_response = await self.http_client.get(weather_url)
            
            if weather_response.status_code != 200:
                return {
                    "success": False,
                    "result": f"Could not fetch weather for {city_name}",
                    "data": None
                }
            
            weather_data = weather_response.json()
            current = weather_data["current"]
            
            # Extract weather info
            temp = current["temperature_2m"]
            feels_like = current["apparent_temperature"]
            humidity = current["relative_humidity_2m"]
            wind_speed = current["wind_speed_10m"]
            weather_code = current["weather_code"]
            
            # Convert weather code to description
            weather_desc = self._weather_code_to_description(weather_code)
            
            # Format location string
            location_str = f"{city_name}, {country}" if country else city_name
            
            # Create TTS-friendly response (more natural, less technical)
            # Round temperature to whole number for easier listening
            temp_int = round(temp)
            feels_int = round(feels_like)
            
            # Build natural response
            if feels_int == temp_int:
                # Don't mention "feels like" if it's the same
                result = f"It's {weather_desc} in {city_name}, {temp_int} degrees"
            else:
                result = f"It's {weather_desc} in {city_name}, {temp_int} degrees, feels like {feels_int}"
            
            # Add extra info only if significantly different conditions
            if humidity > 80:
                result += ", quite humid"
            elif humidity < 30:
                result += ", quite dry"
            
            if wind_speed > 30:
                result += ", and windy"
            
            return {
                "success": True,
                "result": result,
                "data": {
                    "location": location_str,
                    "temperature": temp,
                    "feels_like": feels_like,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                    "description": weather_desc,
                    "weather_code": weather_code
                }
            }
        
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return {"success": False, "result": f"Weather service error: {str(e)}", "data": None}
    
    def _weather_code_to_description(self, code: int) -> str:
        """Convert WMO weather code to human-readable description."""
        # WMO Weather interpretation codes
        weather_codes = {
            0: "clear sky",
            1: "mainly clear",
            2: "partly cloudy",
            3: "overcast",
            45: "foggy",
            48: "depositing rime fog",
            51: "light drizzle",
            53: "moderate drizzle",
            55: "dense drizzle",
            61: "slight rain",
            63: "moderate rain",
            65: "heavy rain",
            71: "slight snow",
            73: "moderate snow",
            75: "heavy snow",
            77: "snow grains",
            80: "slight rain showers",
            81: "moderate rain showers",
            82: "violent rain showers",
            85: "slight snow showers",
            86: "heavy snow showers",
            95: "thunderstorm",
            96: "thunderstorm with slight hail",
            99: "thunderstorm with heavy hail"
        }
        return weather_codes.get(code, "unknown")
    
    async def get_time(self) -> Dict[str, Any]:
        """Get current time and date."""
        now = datetime.now()
        result = now.strftime("It's %I:%M %p on %A, %B %d, %Y")
        
        return {
            "success": True,
            "result": result,
            "data": {
                "datetime": now.isoformat(),
                "timestamp": now.timestamp()
            }
        }
    
    async def calculate(self, expression: str) -> Dict[str, Any]:
        """Perform safe mathematical calculation."""
        try:
            # Clean the expression
            expression = expression.replace("x", "*").replace("×", "*").replace("÷", "/")
            expression = re.sub(r'[^0-9+\-*/().\s]', '', expression)
            
            # Safe eval with limited scope
            allowed_chars = set("0123456789+-*/().")
            if not all(c in allowed_chars or c.isspace() for c in expression):
                return {"success": False, "result": "Invalid mathematical expression", "data": None}
            
            # Evaluate
            result = eval(expression, {"__builtins__": {}}, {})
            
            return {
                "success": True,
                "result": f"{expression} = {result}",
                "data": {"expression": expression, "result": result}
            }
        
        except Exception as e:
            return {"success": False, "result": f"Calculation error: {str(e)}", "data": None}
    
    async def play_music(self, action: str = "play", query: str = "") -> Dict[str, Any]:
        """Control music playback."""
        try:
            if not self.spotify_enabled:
                # Try basic system commands for music control
                if action == "pause":
                    # Try to pause media on different platforms
                    result = await self._control_system_media("pause")
                elif action == "play":
                    result = await self._control_system_media("play")
                elif action == "next":
                    result = await self._control_system_media("next")
                elif action == "previous":
                    result = await self._control_system_media("previous")
                else:
                    return {"success": False, "result": "Music control not configured", "data": None}
                
                return result
            else:
                # Spotify integration (requires spotify credentials)
                return {
                    "success": False,
                    "result": "Spotify integration not yet implemented",
                    "data": None
                }
        
        except Exception as e:
            logger.error(f"Music control error: {e}")
            return {"success": False, "result": f"Music control error: {str(e)}", "data": None}
    
    async def _control_system_media(self, action: str) -> Dict[str, Any]:
        """Control system media playback using platform-specific commands or robot."""
        try:
            import platform
            system = platform.system()
            
            if system == "Linux":
                # Use playerctl for Linux
                cmd = ["playerctl", action]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
                
                if result.returncode == 0:
                    return {
                        "success": True,
                        "result": f"Music {action}",
                        "data": {"action": action}
                    }
            
            elif system == "Darwin":  # macOS
                # Use osascript for macOS Music/Spotify control
                if action == "play":
                    script = 'tell application "Music" to play'
                elif action == "pause":
                    script = 'tell application "Music" to pause'
                elif action == "next":
                    script = 'tell application "Music" to next track'
                elif action == "previous":
                    script = 'tell application "Music" to previous track'
                
                result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=2)
                
                if result.returncode == 0:
                    return {
                        "success": True,
                        "result": f"Music {action}",
                        "data": {"action": action}
                    }
            
            elif system == "Windows":
                # On Windows (cloud server), send music control to robot speakers
                return await self._control_robot_music(action)
            
            return {
                "success": False,
                "result": f"Could not control media on {system}",
                "data": None
            }
        
        except Exception as e:
            logger.error(f"System media control error: {e}")
            return {
                "success": False,
                "result": "Media player not found or not responding",
                "data": None
            }
    
    async def _control_robot_music(self, action: str) -> Dict[str, Any]:
        """Send music control command to robot/rover."""
        if not await self._ensure_client():
            return {"success": False, "result": "Cannot connect to robot", "data": None}
        
        try:
            # Get robot IP from environment
            robot_ip = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")
            
            # Send music control command to robot
            url = f"http://{robot_ip}:8000/music/{action}"
            
            response = await self.http_client.post(url, json={"action": action}, timeout=5.0)
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "result": f"Music {action} on robot",
                    "data": {"action": action, "target": "robot"}
                }
            else:
                return {
                    "success": False,
                    "result": f"Robot returned status {response.status_code}",
                    "data": None
                }
        
        except Exception as e:
            logger.error(f"Robot music control error: {e}")
            return {
                "success": False,
                "result": f"Could not control robot music: {str(e)}",
                "data": None
            }
    
    async def set_reminder(self, message: str, minutes: int = 5) -> Dict[str, Any]:
        """Set a reminder (stores in memory for now)."""
        try:
            # In a production system, this would use a proper task scheduler
            # For now, we'll just acknowledge it
            
            reminder_time = datetime.now().timestamp() + (minutes * 60)
            
            # You could integrate with system notifications here
            # For Linux: notify-send
            # For macOS: osascript -e 'display notification "message" with title "Reminder"'
            # For Windows: PowerShell toast notifications
            
            result = f"I'll remind you to {message} in {minutes} minutes"
            
            return {
                "success": True,
                "result": result,
                "data": {
                    "message": message,
                    "minutes": minutes,
                    "reminder_time": reminder_time
                }
            }
        
        except Exception as e:
            logger.error(f"Reminder error: {e}")
            return {"success": False, "result": f"Could not set reminder: {str(e)}", "data": None}
    
    async def web_search(self, query: str) -> Dict[str, Any]:
        """Search the web for information (simplified version)."""
        # This is a simplified version. In production, you might use:
        # - DuckDuckGo API
        # - Google Custom Search API
        # - Bing Search API
        
        if not await self._ensure_client():
            return {"success": False, "result": "Web search unavailable", "data": None}
        
        try:
            # Use DuckDuckGo Instant Answer API (free, no API key needed)
            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1"
            response = await self.http_client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                # Get the abstract/answer
                answer = data.get("AbstractText", "")
                if not answer:
                    answer = data.get("Answer", "")
                
                if answer:
                    # Truncate to reasonable length
                    if len(answer) > 300:
                        answer = answer[:297] + "..."
                    
                    return {
                        "success": True,
                        "result": answer,
                        "data": data
                    }
                else:
                    return {
                        "success": False,
                        "result": f"No quick answer found for '{query}'",
                        "data": data
                    }
            else:
                return {
                    "success": False,
                    "result": "Web search failed",
                    "data": None
                }
        
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {"success": False, "result": f"Search error: {str(e)}", "data": None}
    
    def get_tools_description(self) -> str:
        """Get a description of available tools for the LLM prompt."""
        tools_list = []
        for name, info in self.tools.items():
            tools_list.append(f"- {name}: {info['description']}")
        
        return "\n".join(tools_list)


# Global instance
_tool_executor = None

def get_tool_executor() -> ToolExecutor:
    """Get or create global ToolExecutor instance."""
    global _tool_executor
    if _tool_executor is None:
        _tool_executor = ToolExecutor()
    return _tool_executor

