"""
Cloud AI Assistant - Ultra-Fast Inference with Optimized Small Model
Optimized for millisecond response times on RTX 4080 SUPER.
"""
import os
import re
import gc
import time
import logging
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
from functools import lru_cache

logger = logging.getLogger('Assistant')

# Try to import tool executor
TOOLS_OK = False
try:
    from tools import get_tool_executor
    TOOLS_OK = True
except ImportError as e:
    logger.warning(f"Tools not available: {e}")
    get_tool_executor = None

# Try to import Qwen2-VL dependencies
QWEN_VL_OK = False
try:
    import torch
    from transformers import (
        Qwen2VLForConditionalGeneration, 
        AutoProcessor,
        BitsAndBytesConfig
    )
    from qwen_vl_utils import process_vision_info
    QWEN_VL_OK = True
except ImportError as e:
    logger.warning(f"Qwen2-VL not available: {e}")
    logger.warning("Install: pip install transformers accelerate qwen-vl-utils torch bitsandbytes flash-attn")

try:
    from PIL import Image
    import numpy as np
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False


class CloudAssistant:
    """AI Assistant using ultra-fast optimized small model."""
    
    # Use smallest Qwen2-VL model for speed (500M params)
    MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"  # Smallest available VLM
    
    # Response cache for common queries
    _response_cache = {}
    
    def __init__(
        self,
        model_id: str = None,
        max_pixels: int = 640 * 480,  # Smaller for speed (VGA)
        lazy_load: bool = False,
        use_4bit: bool = True,  # 4-bit quantization for speed
        use_compile: bool = True,  # torch.compile for optimization
        enable_tools: bool = True  # Enable external API/tool calling
    ):
        self.model_id = model_id or self.MODEL_ID
        self.max_pixels = max_pixels
        self.use_4bit = use_4bit
        self.use_compile = use_compile
        self.enable_tools = enable_tools
        
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._compiled = False
        
        # Initialize tool executor if enabled
        self.tool_executor = None
        if enable_tools and TOOLS_OK:
            self.tool_executor = get_tool_executor()
            logger.info("Tool calling enabled")
        
        if not QWEN_VL_OK:
            logger.error("Qwen2-VL dependencies not installed!")
            return
        
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {self.model_id}")
        logger.info(f"Optimizations: 4-bit={use_4bit}, compile={use_compile}")
        
        if not lazy_load:
            self._load_model()
    
    def _free_gpu(self):
        """Free GPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _load_model(self):
        """Load the Qwen2-VL model with aggressive optimizations."""
        if self.model is not None:
            return
        
        logger.info("Loading optimized model...")
        start = time.time()
        
        self._free_gpu()
        
        try:
            # Configure 4-bit quantization for speed
            quantization_config = None
            if self.use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info("Using 4-bit quantization (NF4)")
            
            # Load model with optimizations
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "cuda:0",
                "low_cpu_mem_usage": True,
            }
            
            # Use SDPA (Scaled Dot Product Attention) - built into PyTorch 2.0+
            # SDPA is very fast and doesn't require flash-attn compilation
            model_kwargs["attn_implementation"] = "sdpa"
            logger.info("Using SDPA (fast attention, built into PyTorch)")
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            
            # Optimize model with torch.compile (if enabled)
            if self.use_compile and not self.use_4bit:  # compile doesn't work well with 4-bit
                logger.info("Compiling model with torch.compile...")
                try:
                    self.model = torch.compile(
                        self.model, 
                        mode="reduce-overhead",
                        fullgraph=False
                    )
                    self._compiled = True
                    logger.info("✅ Model compiled")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")
            
            # Load processor with smaller image constraints
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                min_pixels=128 * 128,  # Smaller minimum
                max_pixels=self.max_pixels,
            )
            
            elapsed = time.time() - start
            logger.info(f"✅ Model loaded in {elapsed:.1f}s")
            
            # Log VRAM usage
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"VRAM: {vram_used:.1f}GB")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.processor = None
    
    def classify_intent(self, question: str) -> dict:
        """Use AI to classify query intent - simple word matching approach."""
        try:
            # Very simple prompt - just ask for type
            classification_prompt = f"""Query: {question}

Is this asking about:
1. VISION - what you see with camera
2. WEATHER - weather/temperature/forecast  
3. TIME - current time/date
4. MATH - calculation/arithmetic
5. MUSIC - play/pause/control music
6. SEARCH - who is/what is/look up
7. CHAT - general conversation

Answer with just the word: VISION, WEATHER, TIME, MATH, MUSIC, SEARCH, or CHAT

Answer:"""

            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": classification_prompt}]
            }]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Fast generation
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    use_cache=True
                )
            
            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().upper()
            
            logger.info(f"AI classification: '{response}'")
            
            # Parse the response
            if "VISION" in response:
                return {"type": "vision", "params": {}}
            elif "WEATHER" in response:
                location = self._extract_location_ai(question)
                return {"type": "weather", "params": {"location": location}}
            elif "TIME" in response:
                return {"type": "time", "params": {}}
            elif "MATH" in response:
                expr = self._extract_math(question)
                return {"type": "calculator", "params": {"expression": expr}}
            elif "MUSIC" in response:
                action = self._extract_music_action(question)
                return {"type": "music", "params": {"action": action}}
            elif "SEARCH" in response:
                return {"type": "search", "params": {"query": question}}
            else:
                return {"type": "conversational", "params": {}}
                
        except Exception as e:
            logger.warning(f"AI classification failed: {e}")
            return {"type": "conversational", "params": {}}
    
    def _extract_location_ai(self, question: str) -> str:
        """Extract location from weather question."""
        words = question.split()
        skip = {'what', 'is', 'the', 'weather', 'in', 'at', 'for', 'like', 'today', 'whats', 'how', 'hows', 'there', 'here', 'now'}
        locations = [w.strip('?.,!') for w in words if w.lower() not in skip and len(w) > 2]
        # Default to Seoul if no specific location mentioned
        return locations[0] if locations else "Seoul"
    
    def _extract_math(self, question: str) -> str:
        """Extract math expression from question."""
        match = re.search(r'([\d\s\+\-\*\/\(\)\.]+)', question)
        if match:
            return match.group(1).strip()
        return question
    
    def _extract_music_action(self, question: str) -> str:
        """Extract music action from question."""
        q = question.lower()
        if 'pause' in q:
            return 'pause'
        elif 'stop' in q:
            return 'stop'
        elif 'next' in q or 'skip' in q:
            return 'next'
        elif 'previous' in q or 'back' in q:
            return 'previous'
        return 'play'
    
    def ask(self, question: str, max_tokens: int = None, temperature: float = 0.3) -> str:
        """Ask a text-only question with LLM-based routing and optional tool calling."""
        if not self.model:
            self._load_model()
            if not self.model:
                return "Model not available."
        
        # Check cache for common queries
        cache_key = question.lower().strip()
        if cache_key in self._response_cache:
            logger.info(f"Cache hit: {question}")
            return self._response_cache[cache_key]
        
        logger.info(f"Query: {question}")
        start = time.time()
        question_len = len(question.split())
        
        # Use fast keyword matching for tool detection
        tool_result = None
        if self.enable_tools and self.tool_executor:
            tool_request = self.tool_executor.detect_tool_use(question)
            if tool_request:
                logger.info(f"Tool detected: {tool_request['tool']}")
                # Execute tool
                loop = self._get_event_loop()
                tool_result = loop.run_until_complete(
                    self.tool_executor.execute(tool_request['tool'], tool_request['params'])
                )
                
                # If tool executed successfully, return result DIRECTLY
                if tool_result and tool_result.get("success"):
                    result_text = tool_result['result']
                    logger.info(f"✅ Tool SUCCESS - Returning directly: {result_text}")
                    return result_text
                elif tool_result:
                    logger.warning(f"⚠️ Tool FAILED: {tool_result.get('result', 'No result')}")
        
        # Fall back to LLM for conversational or if no tool was triggered
        try:
            # Dynamic max_tokens based on query complexity
            if max_tokens is None:
                if question_len <= 5:
                    max_tokens = 30  # Short answers for short questions
                elif question_len <= 15:
                    max_tokens = 60
                else:
                    max_tokens = 100
            
            # Add time context if needed
            time_ctx = ""
            if any(p in question.lower() for p in ['time', 'date', 'today', 'day']):
                time_ctx = f"Current: {datetime.now().strftime('%I:%M %p, %A %B %d, %Y')}. "
            
            # Add tool result context if available but failed
            tool_ctx = ""
            if tool_result and not tool_result.get("success"):
                tool_ctx = f"(Note: Tried to use external tool but it failed: {tool_result.get('result', 'Unknown error')}). "
            
            # Ultra-concise system prompt - multilingual support
            system_instruction = "You are Jarvis. Respond in the SAME language as the user's question. Keep replies under 20 words."
            prompt = f"{system_instruction}\n{time_ctx}{tool_ctx}{question}"
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
            
            # Prepare input
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Optimized generation parameters for speed
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "min_new_tokens": 2,
                "temperature": temperature,
                "do_sample": temperature > 0,
                "pad_token_id": self.processor.tokenizer.pad_token_id,
                "use_cache": True,  # Enable KV cache
                "num_beams": 1,  # Greedy decoding (faster)
            }
            
            # Add sampling parameters only if sampling
            if temperature > 0:
                gen_kwargs["top_p"] = 0.9
                gen_kwargs["top_k"] = 50
            
            # Generate with optimizations
            with torch.no_grad():
                torch.cuda.empty_cache()  # Free memory before generation
                output_ids = self.model.generate(**inputs, **gen_kwargs)
            
            # Decode response
            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
            answer = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            
            answer = self._clean(answer)
            elapsed = time.time() - start
            
            # Cache common greetings and simple queries
            if question_len <= 3 and len(self._response_cache) < 50:
                self._response_cache[cache_key] = answer
            
            logger.info(f"Response in {elapsed*1000:.0f}ms")
            return answer
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return f"Error: {e}"
    
    def ask_with_vision(self, question: str, image, max_tokens: int = None) -> str:
        """Ask a question about an image with optimized generation."""
        if not self.model:
            self._load_model()
            if not self.model:
                return "Model not available."
        
        logger.info(f"Vision query: {question}")
        start = time.time()
        
        try:
            # Dynamic max_tokens for vision queries
            if max_tokens is None:
                question_len = len(question.split())
                max_tokens = min(80 + question_len * 5, 150)
            
            # Convert to PIL Image
            pil_img = self._to_pil(image)
            if not pil_img:
                return "Could not process image."
            
            # Aggressively resize for speed
            pil_img = self._resize_image(pil_img, max_size=640)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": f"{question} (Reply in under 30 words)"}
                    ]
                }
            ]
            
            # Prepare input
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Optimized generation
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "min_new_tokens": 3,
                "temperature": 0.3,
                "do_sample": True,
                "top_p": 0.9,
                "use_cache": True,
                "num_beams": 1,
                "pad_token_id": self.processor.tokenizer.pad_token_id,
            }
            
            with torch.no_grad():
                torch.cuda.empty_cache()
                output_ids = self.model.generate(**inputs, **gen_kwargs)
            
            # Decode response
            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
            answer = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            
            answer = self._clean(answer)
            elapsed = time.time() - start
            logger.info(f"Vision response in {elapsed*1000:.0f}ms")
            return answer if answer else "I couldn't understand what I'm seeing."
            
        except Exception as e:
            logger.error(f"Vision query failed: {e}")
            return f"Error: {e}"
    
    def _resize_image(self, img: Image.Image, max_size: int = 640) -> Image.Image:
        """Aggressively resize image for maximum speed."""
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            # Use BILINEAR for speed (faster than LANCZOS)
            img = img.resize((new_w, new_h), Image.BILINEAR)
        return img
    
    def _to_pil(self, image) -> Optional[Image.Image]:
        """Convert various formats to PIL Image."""
        if not PIL_OK:
            return None
        
        try:
            if isinstance(image, Image.Image):
                return image.convert("RGB")
            if isinstance(image, bytes):
                import io
                return Image.open(io.BytesIO(image)).convert("RGB")
            if isinstance(image, np.ndarray):
                if CV2_OK and len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return Image.fromarray(image).convert("RGB")
            if isinstance(image, str) and os.path.exists(image):
                return Image.open(image).convert("RGB")
        except Exception as e:
            logger.error(f"Image conversion failed: {e}")
        return None
    
    def _clean(self, text: str) -> str:
        """Clean model output for TTS."""
        text = text.replace('</s>', '').replace('#', '')
        text = re.sub(r'[\U0001F300-\U0001F9FF]', '', text)  # Remove emojis
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Remove italic
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _get_event_loop(self):
        """Get or create event loop."""
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
    
    def extract_movement(self, response: str, query: str) -> Optional[Dict[str, Any]]:
        """Extract movement commands from text in multiple languages."""
        text = f"{query} {response}".lower()
        
        # Multilingual movement patterns
        patterns = {
            'forward': [
                # English
                r'go\s+forward', r'move\s+forward', r'ahead', r'drive\s+forward',
                # Spanish
                r'avanza', r'adelante', r've\s+adelante', r'mueve\s+adelante',
                # French
                r'avance', r'en\s+avant', r'vers\s+l\'avant',
                # German
                r'vorwärts', r'nach\s+vorne', r'fahre\s+vorwärts',
                # Italian
                r'avanti', r'vai\s+avanti',
                # Portuguese
                r'para\s+frente', r'em\s+frente',
                # Russian
                r'вперёд', r'вперед',
                # Japanese
                r'前', r'進め',
                # Chinese
                r'前进', r'向前'
            ],
            'backward': [
                # English
                r'go\s+back', r'move\s+back', r'reverse', r'backward',
                # Spanish
                r'retrocede', r'atrás', r'para\s+atrás', r've\s+atrás',
                # French
                r'recule', r'en\s+arrière', r'vers\s+l\'arrière',
                # German
                r'rückwärts', r'zurück', r'nach\s+hinten',
                # Italian
                r'indietro', r'vai\s+indietro',
                # Portuguese
                r'para\s+trás', r'ré',
                # Russian
                r'назад',
                # Japanese
                r'後ろ', r'バック',
                # Chinese
                r'后退', r'向后'
            ],
            'left': [
                # English
                r'turn\s+left', r'go\s+left', r'\bleft\b',
                # Spanish
                r'gira\s+izquierda', r'a\s+la\s+izquierda', r'izquierda',
                # French
                r'tourne\s+à\s+gauche', r'à\s+gauche', r'gauche',
                # German
                r'nach\s+links', r'links',
                # Italian
                r'gira\s+a\s+sinistra', r'sinistra',
                # Portuguese
                r'vire\s+à\s+esquerda', r'esquerda',
                # Russian
                r'налево', r'влево',
                # Japanese
                r'左', r'左に',
                # Chinese
                r'左转', r'向左'
            ],
            'right': [
                # English
                r'turn\s+right', r'go\s+right', r'\bright\b',
                # Spanish
                r'gira\s+derecha', r'a\s+la\s+derecha', r'derecha',
                # French
                r'tourne\s+à\s+droite', r'à\s+droite', r'droite',
                # German
                r'nach\s+rechts', r'rechts',
                # Italian
                r'gira\s+a\s+destra', r'destra',
                # Portuguese
                r'vire\s+à\s+direita', r'direita',
                # Russian
                r'направо', r'вправо',
                # Japanese
                r'右', r'右に',
                # Chinese
                r'右转', r'向右'
            ],
            'stop': [
                # English
                r'\bstop\b', r'halt',
                # Spanish
                r'para', r'detente', r'alto',
                # French
                r'arrête', r'stop',
                # German
                r'halt', r'stopp',
                # Italian
                r'fermati', r'stop',
                # Portuguese
                r'pare', r'para',
                # Russian
                r'стой', r'остановись',
                # Japanese
                r'止まれ', r'ストップ',
                # Chinese
                r'停止', r'停'
            ]
        }
        
        for direction, pats in patterns.items():
            for pat in pats:
                if re.search(pat, text):
                    dist = 0.5
                    # Distance modifiers in multiple languages
                    if any(word in text for word in ['little', 'bit', 'poco', 'un peu', 'wenig', 'piccolo', 'pouco', 'немного', '少し', '一点']):
                        dist = 0.2
                    elif any(word in text for word in ['far', 'lot', 'much', 'mucho', 'beaucoup', 'viel', 'molto', 'muito', 'много', 'たくさん', '很多']):
                        dist = 1.0
                    
                    speed = 'medium'
                    # Speed modifiers in multiple languages
                    if any(word in text for word in ['slow', 'slowly', 'lento', 'lentement', 'langsam', 'lentamente', 'devagar', 'медленно', 'ゆっくり', '慢']):
                        speed = 'slow'
                    elif any(word in text for word in ['fast', 'quickly', 'rápido', 'vite', 'schnell', 'veloce', 'rápido', 'быстро', '速く', '快']):
                        speed = 'fast'
                    
                    return {'direction': direction, 'distance': dist, 'speed': speed}
        
        return None
