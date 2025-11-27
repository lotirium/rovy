"""
Vision Processing Module - Using Local Models
Handles face recognition and image processing
Ported from original rovy/face_recognizer.py
"""
import os
import time
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
import numpy as np

logger = logging.getLogger('Vision')

# Try to import OpenCV
CV2_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    logger.warning("OpenCV not available. Install with: pip install opencv-python")

# Try to import face_recognition (uses dlib under the hood)
FACE_RECOGNITION_AVAILABLE = False
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    logger.info("face_recognition not available. Install with: pip install face-recognition")

# Try to import PIL
PIL_AVAILABLE = False
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    logger.info("Pillow not available. Install with: pip install Pillow")


class VisionProcessor:
    """
    Handles vision processing including face recognition.
    Uses CNN model for face detection (GPU accelerated if available).
    
    Ported from rovy/face_recognizer.py
    """
    
    def __init__(self, known_faces_dir: str = "known_faces", tolerance: float = 0.6):
        """
        Initialize vision processor.
        
        Args:
            known_faces_dir: Directory containing known face images
            tolerance: Face matching tolerance (lower = stricter, default 0.6)
        """
        self.known_faces_dir = known_faces_dir
        self.tolerance = tolerance
        
        # Known face encodings and names
        self.known_encodings: List[np.ndarray] = []
        self.known_names: List[str] = []
        
        # Load known faces
        if FACE_RECOGNITION_AVAILABLE:
            self._load_known_faces()
        else:
            logger.warning("Face recognition not available")
    
    def _load_known_faces(self):
        """Load known faces from directory - ported from rovy/face_recognizer.py"""
        faces_dir = Path(self.known_faces_dir)
        
        if not faces_dir.exists():
            logger.warning(f"Known faces directory not found: {faces_dir}")
            faces_dir.mkdir(parents=True, exist_ok=True)
            return
        
        logger.info(f"Loading known faces from {faces_dir}...")
        
        for image_path in faces_dir.iterdir():
            if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    # Load image
                    image = face_recognition.load_image_file(str(image_path))
                    
                    # Get face encoding
                    encodings = face_recognition.face_encodings(image)
                    
                    if encodings:
                        self.known_encodings.append(encodings[0])
                        self.known_names.append(image_path.stem)
                        logger.info(f"  ✅ Loaded: {image_path.stem}")
                    else:
                        logger.warning(f"  ⚠️ No face found in: {image_path.name}")
                        
                except Exception as e:
                    logger.error(f"  ❌ Error loading {image_path.name}: {e}")
        
        logger.info(f"Loaded {len(self.known_names)} known faces")
    
    def decode_image(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Decode image bytes to numpy array.
        
        Args:
            image_bytes: JPEG/PNG image data
            
        Returns:
            numpy array (BGR format) or None
        """
        if not CV2_AVAILABLE:
            return None
        
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            logger.error(f"Image decode error: {e}")
            return None
    
    def recognize_faces(self, image, is_rgb: bool = False) -> List[Dict[str, Any]]:
        """
        Detect and recognize faces in an image.
        Ported from rovy/face_recognizer.py with optimizations.
        
        Args:
            image: numpy array (BGR or RGB) or bytes
            is_rgb: If True, image is already in RGB format
            
        Returns:
            List of face detections: [{'name': str, 'confidence': float, 'bbox': tuple}, ...]
        """
        if not FACE_RECOGNITION_AVAILABLE or not CV2_AVAILABLE:
            return []
        
        if not self.known_encodings:
            logger.debug("No known faces loaded")
            return []
        
        total_start = time.time()
        
        try:
            # Handle bytes input
            if isinstance(image, bytes):
                image = self.decode_image(image)
                if image is None:
                    return []
                is_rgb = False  # OpenCV decodes to BGR
            
            # Convert to RGB if needed
            if is_rgb:
                rgb_frame = image
            else:
                rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # OPTIMIZATION 1: Downsample to 0.5x for faster processing
            scale_factor = 0.5
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=scale_factor, fy=scale_factor)
            
            # OPTIMIZATION 2: Use CNN model for detection (GPU accelerated)
            # CNN at 0.5x scale is faster than HOG
            detect_start = time.time()
            face_locations = face_recognition.face_locations(
                small_frame, 
                model="cnn",  # Use CNN for GPU acceleration
                number_of_times_to_upsample=0
            )
            detect_time = time.time() - detect_start
            
            if not face_locations:
                logger.debug(f"No faces found (detection took {detect_time:.2f}s)")
                return []
            
            logger.info(f"Found {len(face_locations)} face(s) (detection: {detect_time:.2f}s)")
            
            # OPTIMIZATION 3: Fast encoding (num_jitters=0)
            encode_start = time.time()
            face_encodings = face_recognition.face_encodings(
                small_frame, 
                face_locations, 
                num_jitters=0  # Fastest, slightly less accurate
            )
            encode_time = time.time() - encode_start
            logger.debug(f"Encoding took {encode_time:.2f}s")
            
            results = []
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = "Unknown"
                confidence = 0.0
                
                # OPTIMIZATION 4: Vectorized distance calculation
                distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                
                if len(distances) > 0:
                    best_idx = np.argmin(distances)
                    distance = distances[best_idx]
                    
                    logger.debug(f"Best match: {self.known_names[best_idx]} (distance: {distance:.3f})")
                    
                    # Distance < tolerance is a match (lower is better)
                    if distance < self.tolerance:
                        name = self.known_names[best_idx]
                        confidence = 1.0 - distance
                        logger.info(f"✅ Recognized: {name}")
                
                # Scale bbox back to original size
                top = int(top / scale_factor)
                right = int(right / scale_factor)
                bottom = int(bottom / scale_factor)
                left = int(left / scale_factor)
                
                results.append({
                    'name': name,
                    'confidence': confidence,
                    'bbox': (left, top, right, bottom),
                    'center': ((left + right) // 2, (top + bottom) // 2)
                })
            
            total_time = time.time() - total_start
            logger.debug(f"Total face recognition: {total_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Face recognition error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def add_known_face(self, name: str, image) -> bool:
        """
        Add a new known face to the database.
        
        Args:
            name: Name of the person
            image: Image containing their face (numpy array or bytes)
            
        Returns:
            bool: Success
        """
        if not FACE_RECOGNITION_AVAILABLE or not CV2_AVAILABLE:
            return False
        
        try:
            # Handle bytes input
            if isinstance(image, bytes):
                image = self.decode_image(image)
                if image is None:
                    return False
            
            # Convert to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get face encoding
            encodings = face_recognition.face_encodings(rgb_image)
            
            if not encodings:
                logger.warning(f"No face found in image for {name}")
                return False
            
            # Save image
            faces_dir = Path(self.known_faces_dir)
            faces_dir.mkdir(parents=True, exist_ok=True)
            
            image_path = faces_dir / f"{name}.jpg"
            cv2.imwrite(str(image_path), image)
            
            # Add to known faces
            self.known_encodings.append(encodings[0])
            self.known_names.append(name)
            
            logger.info(f"✅ Added known face: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding face: {e}")
            return False
    
    def analyze_scene(self, image) -> Dict[str, Any]:
        """
        Basic scene analysis (brightness, blur detection, etc.)
        
        Args:
            image: numpy array or bytes
            
        Returns:
            dict with scene properties
        """
        if not CV2_AVAILABLE:
            return {}
        
        try:
            # Handle bytes input
            if isinstance(image, bytes):
                image = self.decode_image(image)
                if image is None:
                    return {}
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Brightness (mean of grayscale)
            brightness = np.mean(gray)
            
            # Contrast (standard deviation)
            contrast = np.std(gray)
            
            # Blur detection (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            is_blurry = laplacian_var < 100
            
            return {
                'brightness': float(brightness),
                'contrast': float(contrast),
                'blur_score': float(laplacian_var),
                'is_blurry': is_blurry,
                'resolution': (image.shape[1], image.shape[0])
            }
            
        except Exception as e:
            logger.error(f"Scene analysis error: {e}")
            return {}


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    processor = VisionProcessor()
    
    print(f"\nKnown faces: {processor.known_names}")
    
    # Test with a sample image if available
    test_image = Path("test_image.jpg")
    if test_image.exists():
        with open(test_image, "rb") as f:
            image_bytes = f.read()
        
        faces = processor.recognize_faces(image_bytes)
        print(f"\nDetected faces: {faces}")
        
        scene = processor.analyze_scene(image_bytes)
        print(f"\nScene analysis: {scene}")
