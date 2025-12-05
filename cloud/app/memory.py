"""
Memory Management System for Chat
Stores and retrieves user facts and preferences from conversations.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages persistent memory storage for user facts and preferences."""
    
    def __init__(self, memory_file: Optional[str] = None):
        """
        Initialize memory manager.
        
        Args:
            memory_file: Path to JSON file for storing memory. Defaults to memory.json in cloud directory.
        """
        if memory_file is None:
            # Default to cloud/memory.json
            project_root = Path(__file__).parent.parent
            memory_file = str(project_root / "memory.json")
        
        self.memory_file = Path(memory_file)
        self.memory: Dict[str, Any] = {
            "facts": [],  # List of user facts
            "preferences": {},  # User preferences
            "conversation_history": [],  # Recent conversation context
            "last_updated": None
        }
        self._load_memory()
    
    def _load_memory(self):
        """Load memory from JSON file."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    self.memory = json.load(f)
                logger.info(f"Loaded memory from {self.memory_file}")
            except Exception as e:
                logger.error(f"Failed to load memory: {e}")
                self.memory = {
                    "facts": [],
                    "preferences": {},
                    "conversation_history": [],
                    "last_updated": None
                }
        else:
            logger.info(f"Memory file not found, starting fresh: {self.memory_file}")
            self._save_memory()
    
    def _save_memory(self):
        """Save memory to JSON file."""
        try:
            self.memory["last_updated"] = datetime.now().isoformat()
            # Ensure directory exists
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved memory to {self.memory_file}")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    def add_fact(self, fact: str, source: Optional[str] = None):
        """
        Add a fact about the user.
        
        Args:
            fact: The fact to store (e.g., "User's name is John")
            source: Optional source of the fact (e.g., conversation message)
        """
        fact_entry = {
            "fact": fact,
            "added_at": datetime.now().isoformat(),
            "source": source
        }
        
        # Check for duplicates (simple string matching)
        existing_facts = [f["fact"].lower() for f in self.memory["facts"]]
        if fact.lower() not in existing_facts:
            self.memory["facts"].append(fact_entry)
            self._save_memory()
            logger.info(f"Added fact: {fact}")
        else:
            logger.debug(f"Fact already exists: {fact}")
    
    def get_facts(self, limit: Optional[int] = None) -> List[str]:
        """
        Get all stored facts.
        
        Args:
            limit: Maximum number of facts to return (most recent first)
        
        Returns:
            List of fact strings
        """
        facts = [f["fact"] for f in self.memory["facts"]]
        if limit:
            facts = facts[-limit:]  # Get most recent
        return facts
    
    def get_memory_context(self, max_facts: int = 5) -> str:
        """
        Get memory context as a formatted string for AI prompts.
        
        Args:
            max_facts: Maximum number of facts to include
        
        Returns:
            Formatted memory context string
        """
        facts = self.get_facts(limit=max_facts)
        if not facts:
            return ""
        
        context = "User facts I remember:\n"
        for i, fact in enumerate(facts, 1):
            context += f"{i}. {fact}\n"
        
        return context
    
    def add_preference(self, key: str, value: Any):
        """
        Add or update a user preference.
        
        Args:
            key: Preference key (e.g., "favorite_color")
            value: Preference value (e.g., "blue")
        """
        self.memory["preferences"][key] = {
            "value": value,
            "updated_at": datetime.now().isoformat()
        }
        self._save_memory()
        logger.info(f"Updated preference: {key} = {value}")
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference value."""
        pref = self.memory["preferences"].get(key, {})
        return pref.get("value", default) if isinstance(pref, dict) else pref
    
    def add_conversation(self, user_message: str, assistant_response: str):
        """
        Add a conversation turn to history (for context).
        
        Args:
            user_message: User's message
            assistant_response: Assistant's response
        """
        entry = {
            "user": user_message,
            "assistant": assistant_response,
            "timestamp": datetime.now().isoformat()
        }
        self.memory["conversation_history"].append(entry)
        
        # Keep only last 20 conversations
        if len(self.memory["conversation_history"]) > 20:
            self.memory["conversation_history"] = self.memory["conversation_history"][-20:]
        
        self._save_memory()
    
    def get_recent_conversations(self, limit: int = 5) -> List[Dict[str, str]]:
        """Get recent conversation history."""
        return self.memory["conversation_history"][-limit:]
    
    def extract_facts_from_conversation(self, user_message: str, assistant_response: str, assistant) -> List[str]:
        """
        Use AI to extract facts about the user from conversation.
        
        Args:
            user_message: User's message
            assistant_response: Assistant's response
            assistant: CloudAssistant instance for fact extraction
        
        Returns:
            List of extracted facts
        """
        # Combine message and response for context
        conversation = f"User: {user_message}\nAssistant: {assistant_response}"
        
        # Prompt AI to extract facts - be more specific about what to extract
        extraction_prompt = f"""Analyze this conversation and extract factual information about the user.

Extract ONLY clear facts like:
- Name: "User's name is John"
- Preferences: "User likes coffee"
- Personal details: "User lives in Seattle"
- Facts: "User works as a teacher"

Do NOT extract:
- Questions the user asked
- General statements
- Things the assistant said

If no clear facts about the user are present, respond with just "NONE".

Conversation:
{conversation}

Extract facts (one per line, format: "User's [fact]" or "NONE"):
"""
        
        try:
            # Use assistant to extract facts
            response = assistant.ask(extraction_prompt, max_tokens=150, temperature=0.1, disable_tools=True)
            
            facts = []
            for line in response.split('\n'):
                line = line.strip()
                # Skip empty lines, "NONE", comments, and very short lines
                if not line or line.upper() == "NONE" or line.startswith('#') or len(line) < 10:
                    continue
                
                # Clean up the fact
                fact = line.lstrip('- ').lstrip('* ').lstrip('• ').strip()
                
                # Only add if it looks like a fact about the user
                if len(fact) > 10 and ('user' in fact.lower() or any(word in fact.lower() for word in ['name', 'likes', 'lives', 'works', 'prefers', 'favorite', 'enjoys'])):
                    facts.append(fact)
            
            return facts
        except Exception as e:
            logger.error(f"Failed to extract facts: {e}")
            return []
    
    def clear_memory(self):
        """Clear all memory (use with caution)."""
        self.memory = {
            "facts": [],
            "preferences": {},
            "conversation_history": [],
            "last_updated": None
        }
        self._save_memory()
        logger.warning("Memory cleared")
    
    def get_all_memory(self) -> Dict[str, Any]:
        """Get all memory data (for API endpoint)."""
        return {
            "facts": self.memory["facts"],
            "preferences": self.memory["preferences"],
            "conversation_count": len(self.memory["conversation_history"]),
            "last_updated": self.memory.get("last_updated")
        }

