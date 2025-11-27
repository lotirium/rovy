"""
Cloud AI Assistant - Using Local Models
Uses llama.cpp with Gemma for text and LLaVA/Phi for vision
Ported from original rovy/dual_model_assistant.py
"""
import os
import re
import gc
import time
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np

logger = logging.getLogger('Assistant')

# Try to import llama-cpp-python
LLAMA_CPP_AVAILABLE = False
try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    logger.warning("llama-cpp-python not available. Install with: pip install llama-cpp-python")

# Try to import PIL for image handling
PIL_AVAILABLE = False
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    logger.warning("Pillow not available")

# Try to import OpenCV
CV2_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    pass


class CloudAssistant:
    """
    AI Assistant using local LLM models via llama.cpp
    - Gemma-2-2B for fast text/chat tasks
    - LLaVA or Phi-3-Vision for image understanding
    
    Ported from rovy/dual_model_assistant.py
    """
    
    def __init__(
        self,
        text_model_path: str = None,
        vision_model_path: str = None,
        vision_mmproj_path: str = None,
        n_gpu_layers: int = -1,  # -1 = all layers on GPU
        n_ctx: int = 2048,
        lazy_load_vision: bool = True
    ):
        """
        Initialize the assistant with local models.
        
        Args:
            text_model_path: Path to text-only LLM (GGUF format)
            vision_model_path: Path to VLM model (GGUF format)
            vision_mmproj_path: Path to vision projector (GGUF format)
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
            n_ctx: Context window size
            lazy_load_vision: If True, only load vision model when needed
        """
        self.text_model_path = text_model_path
        self.vision_model_path = vision_model_path
        self.vision_mmproj_path = vision_mmproj_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.lazy_load_vision = lazy_load_vision
        
        # Model instances
        self.text_llm = None
        self.vision_llm = None
        self.vision_chat_handler = None
        
        # Find models if not specified
        self._find_models()
        
        logger.info(f"Text model: {self.text_model_path}")
        logger.info(f"Vision model: {self.vision_model_path}")
        logger.info(f"Vision projector: {self.vision_mmproj_path}")
        
        if not LLAMA_CPP_AVAILABLE:
            logger.error("llama-cpp-python is required!")
            return
        
        # Free GPU memory before loading
        self._free_gpu_memory()
        
        # Load text model immediately
        self._load_text_model()
        
        # Load vision model now or later
        if not lazy_load_vision:
            self._load_vision_model()
    
    def _find_models(self):
        """Auto-detect model paths"""
        # Common model locations
        model_dirs = [
            os.path.expanduser("~/.cache"),
            os.path.expanduser("~/models"),
            "./models",
            "C:/models",
            "D:/models",
        ]
        
        # Find text model (Gemma, Llama, Mistral, etc.)
        if not self.text_model_path:
            text_patterns = [
                "gemma-2-2b-it-Q4_K_S.gguf",
                "gemma-2-2b-it-Q4_K_M.gguf",
                "gemma-2b*.gguf",
                "llama-2-7b*.gguf",
                "mistral-7b*.gguf",
                "phi-2*.gguf",
            ]
            self.text_model_path = self._search_for_model(model_dirs, text_patterns)
        
        # Find vision model (LLaVA, Phi-3-Vision, etc.)
        if not self.vision_model_path:
            vision_patterns = [
                "llava-phi-3-mini-int4.gguf",
                "llava-v1.5-7b-q4.gguf",
                "llava-v1.5-7b-Q4_K_M.gguf",
                "llava*.gguf",
            ]
            self.vision_model_path = self._search_for_model(model_dirs, vision_patterns)
        
        # Find vision projector
        if not self.vision_mmproj_path and self.vision_model_path:
            # Match projector to model type
            if self.vision_model_path and "phi-3-mini" in self.vision_model_path.lower():
                proj_patterns = ["llava-phi-3-mini-mmproj-f16.gguf"]
            else:
                proj_patterns = [
                    "llava-mmproj-f16.gguf",
                    "mmproj-model-f16.gguf",
                    "mmproj*.gguf",
                ]
            self.vision_mmproj_path = self._search_for_model(model_dirs, proj_patterns)
    
    def _search_for_model(self, dirs: list, patterns: list) -> Optional[str]:
        """Search for model file in directories"""
        import glob
        
        for directory in dirs:
            if not os.path.exists(directory):
                continue
            
            for pattern in patterns:
                matches = glob.glob(os.path.join(directory, pattern))
                for match in matches:
                    if os.path.isfile(match) and os.path.getsize(match) > 1000000:  # > 1MB
                        return match
        
        return None
    
    def _free_gpu_memory(self):
        """Force garbage collection and free GPU memory"""
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("🧹 Cleared GPU memory")
        except ImportError:
            pass
    
    def _load_text_model(self):
        """Load the lightweight text-only model"""
        if self.text_llm is not None:
            return
        
        if not self.text_model_path or not os.path.exists(self.text_model_path):
            logger.error(f"Text model not found: {self.text_model_path}")
            return
        
        logger.info("Loading text model...")
        start = time.time()
        
        try:
            self.text_llm = Llama(
                model_path=self.text_model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                n_threads=os.cpu_count() or 4,
                n_batch=512,
                verbose=False,
            )
            elapsed = time.time() - start
            logger.info(f"✅ Text model loaded in {elapsed:.2f}s")
        except Exception as e:
            logger.error(f"❌ Failed to load text model: {e}")
    
    def _load_vision_model(self):
        """Load vision model using llama.cpp"""
        if self.vision_llm is not None:
            return
        
        if not self.vision_model_path or not os.path.exists(self.vision_model_path):
            logger.error(f"Vision model not found: {self.vision_model_path}")
            return
        
        if not self.vision_mmproj_path or not os.path.exists(self.vision_mmproj_path):
            logger.error(f"Vision projector not found: {self.vision_mmproj_path}")
            return
        
        model_type = "Phi-3-Mini Vision" if "phi" in self.vision_model_path.lower() else "LLaVA"
        logger.info(f"Loading {model_type}...")
        start = time.time()
        
        try:
            self.vision_chat_handler = Llava15ChatHandler(
                clip_model_path=self.vision_mmproj_path,
                verbose=False
            )
            
            self.vision_llm = Llama(
                model_path=self.vision_model_path,
                chat_handler=self.vision_chat_handler,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=2048,
                n_threads=os.cpu_count() or 4,
                n_batch=512,
                logits_all=True,
                verbose=False,
            )
            
            elapsed = time.time() - start
            logger.info(f"✅ {model_type} loaded in {elapsed:.2f}s")
        except Exception as e:
            logger.error(f"❌ Failed to load vision model: {e}")
    
    def ask(self, question: str, max_tokens: int = 150, temperature: float = 0.7) -> str:
        """
        Ask a text-only question (fast path using lightweight model).
        
        Args:
            question: Question text
            max_tokens: Maximum response length
            temperature: Sampling temperature
            
        Returns:
            str: Model's response
        """
        if self.text_llm is None:
            self._load_text_model()
            if self.text_llm is None:
                return "Text model not available. Please configure model path."
        
        logger.info(f"Text query: {question}")
        start = time.time()
        
        try:
            # Add time context for time-related questions
            time_context = ""
            question_lower = question.lower()
            if any(phrase in question_lower for phrase in ['what time', 'time is it', 'what date', 'today']):
                time_context = f"Current time: {datetime.now().strftime('%I:%M %p, %A %B %d, %Y')}. "
            
            response = self.text_llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": f"You are Rovy, a helpful robot assistant. {time_context}Give direct, concise answers. Keep responses under 50 words for natural conversation."
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9,
            )
            
            elapsed = time.time() - start
            answer = response['choices'][0]['message']['content'].strip()
            
            # Clean up response
            answer = self._clean_response(answer)
            
            logger.info(f"⚡ Text response in {elapsed:.2f}s")
            return answer
            
        except Exception as e:
            logger.error(f"❌ Text query failed: {e}")
            return f"Error: {str(e)}"
    
    def ask_with_vision(self, question: str, image, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """
        Ask a question about an image (vision model).
        
        Args:
            question: Question about the image
            image: Can be:
                   - numpy array (OpenCV image)
                   - bytes (JPEG/PNG data)
                   - PIL Image
                   - file path string
            max_tokens: Maximum response length
            temperature: Sampling temperature
            
        Returns:
            str: Model's response
        """
        # Load vision model if needed
        if self.vision_llm is None:
            self._load_vision_model()
            if self.vision_llm is None:
                return "Vision model not available. Please configure model paths."
        
        logger.info(f"Vision query: {question}")
        start = time.time()
        
        # Reset KV cache
        self.vision_llm.reset()
        
        try:
            # Convert image to PIL
            pil_image = self._to_pil_image(image)
            if pil_image is None:
                return "Could not process image."
            
            # Convert to base64 data URI
            import io
            import base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG", quality=70)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            data_uri = f"data:image/jpeg;base64,{img_str}"
            
            response = self.vision_llm.create_chat_completion(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": question}
                    ]
                }],
                max_tokens=max_tokens,
                temperature=max(0.1, temperature),
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.2,
                frequency_penalty=0.2,
            )
            
            # Extract response
            answer = ""
            if isinstance(response, dict) and 'choices' in response and response['choices']:
                choice = response['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    answer = choice['message']['content']
                elif 'text' in choice:
                    answer = choice['text']
            
            answer = answer.strip() if answer else ""
            
            # Filter junk responses
            if answer and len(set(answer.replace('\n', '').replace(' ', ''))) <= 2:
                logger.warning("⚠️ Filtered junk output")
                answer = ""
            
            # Clean up
            answer = self._clean_response(answer)
            
            elapsed = time.time() - start
            logger.info(f"⚡ Vision response in {elapsed:.2f}s")
            return answer if answer else "I couldn't understand what I'm seeing."
            
        except Exception as e:
            logger.error(f"❌ Vision query failed: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def _to_pil_image(self, image) -> Optional['Image.Image']:
        """Convert various image formats to PIL Image"""
        if not PIL_AVAILABLE:
            return None
        
        try:
            # Already PIL Image
            if isinstance(image, Image.Image):
                return image
            
            # File path
            if isinstance(image, str) and os.path.exists(image):
                return Image.open(image)
            
            # Bytes (JPEG/PNG)
            if isinstance(image, bytes):
                import io
                return Image.open(io.BytesIO(image))
            
            # Numpy array (OpenCV)
            if isinstance(image, np.ndarray):
                if CV2_AVAILABLE:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(image_rgb)
                else:
                    return Image.fromarray(image)
            
            return None
        except Exception as e:
            logger.error(f"Image conversion error: {e}")
            return None
    
    def _clean_response(self, text: str) -> str:
        """Clean up model response for TTS"""
        # Remove artifacts
        text = text.replace('#', '').replace('</s>', '').strip()
        text = text.replace('[end of text]', '').replace('[End of text]', '')
        
        # Remove emojis (don't work well with TTS)
        text = re.sub(r'[\U0001F300-\U0001F9FF]', '', text)
        text = re.sub(r'[\U0001F600-\U0001F64F]', '', text)
        text = re.sub(r'[\U00002600-\U000027BF]', '', text)
        
        # Remove markdown
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'^\s*[\*\-•]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Clean whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_movement(self, response: str, original_query: str) -> Optional[Dict[str, Any]]:
        """
        Extract movement commands from query or response.
        
        Args:
            response: The assistant's response
            original_query: The user's original query
            
        Returns:
            dict with 'direction', 'distance', 'speed' or None
        """
        text = f"{original_query} {response}".lower()
        
        movement_patterns = {
            'forward': [r'go\s+forward', r'move\s+forward', r'moving\s+forward', r'ahead', r'go\s+straight'],
            'backward': [r'go\s+back', r'move\s+back', r'moving\s+back', r'reverse', r'backward'],
            'left': [r'turn\s+left', r'go\s+left', r'turning\s+left', r'rotate\s+left'],
            'right': [r'turn\s+right', r'go\s+right', r'turning\s+right', r'rotate\s+right'],
            'stop': [r'\bstop\b', r'halt', r'freeze', r'don\'t\s+move']
        }
        
        for direction, patterns in movement_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    distance = 0.5
                    if 'little' in text or 'bit' in text:
                        distance = 0.2
                    elif 'lot' in text or 'far' in text:
                        distance = 1.0
                    
                    speed = 'medium'
                    if 'slow' in text or 'careful' in text:
                        speed = 'slow'
                    elif 'fast' in text or 'quick' in text:
                        speed = 'fast'
                    
                    return {'direction': direction, 'distance': distance, 'speed': speed}
        
        return None
    
    def unload_vision(self):
        """Unload vision model to free memory"""
        if self.vision_llm is not None:
            logger.info("Unloading vision model...")
            self.vision_llm = None
            self.vision_chat_handler = None
            gc.collect()
            logger.info("✅ Vision model unloaded")


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    assistant = CloudAssistant()
    
    # Test text
    print("\n--- Text Test ---")
    response = assistant.ask("Hello! What can you do?")
    print(f"Response: {response}")
    
    # Test movement extraction
    print("\n--- Movement Test ---")
    movement = assistant.extract_movement("I'll move forward now", "go forward")
    print(f"Movement: {movement}")
