#!/usr/bin/env python3
"""
Face Recognition Module
Recognizes known faces from camera frames
"""
import face_recognition
import cv2
import os
import numpy as np

class FaceRecognizer:
    def __init__(self, known_dir="known-faces"):
        self.known_encodings = []
        self.known_names = []
        self.load_known_faces(known_dir)

    def load_known_faces(self, folder):
        """Load all known faces and their names"""
        if not os.path.exists(folder):
            print(f"⚠️  Warning: Known faces directory '{folder}' not found")
            return
            
        print(f"\n[Face Recognition] Loading known faces from {folder}...")
        for filename in os.listdir(folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(folder, filename)
                name = os.path.splitext(filename)[0]
                try:
                    image = face_recognition.load_image_file(path)
                    encoding = face_recognition.face_encodings(image)
                    if encoding:
                        self.known_encodings.append(encoding[0])
                        self.known_names.append(name)
                        print(f"  ✅ Loaded: {name}")
                    else:
                        print(f"  ⚠️  No face found in: {filename}")
                except Exception as e:
                    print(f"  ❌ Error loading {filename}: {e}")
        
        print(f"[Face Recognition] Loaded {len(self.known_names)} known faces\n")

    def recognize_face(self, frame, is_rgb=True):
        """Detect and recognize faces from the frame
        
        Args:
            frame: Image from camera (RGB or BGR depending on is_rgb parameter)
            is_rgb: If True, frame is already in RGB format (default: True)
            
        Returns:
            List of tuples: (name, confidence, (left, top, right, bottom))
        """
        if not self.known_encodings:
            return []
            
        # Convert to RGB if needed
        if is_rgb:
            rgb_frame = frame
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find faces
        face_locations = face_recognition.face_locations(rgb_frame)
        if not face_locations:
            print(f"     [Debug] No face locations found in frame")
            return []
        
        print(f"     [Debug] Found {len(face_locations)} face(s) in frame")
            
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        matches = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            confidence = 0.0
            
            # Compare with known faces
            distances = face_recognition.face_distance(self.known_encodings, face_encoding)
            if len(distances) > 0:
                best_match_index = np.argmin(distances)
                distance = distances[best_match_index]
                
                # Debug: Show best match and distance
                print(f"     [Debug] Best match: {self.known_names[best_match_index]} (distance: {distance:.3f})")
                
                # Distance < 0.6 is usually a good match (lower is better)
                # Relaxed threshold from 0.45 to 0.6 for better detection
                if distance < 0.6:
                    name = self.known_names[best_match_index]
                    confidence = 1.0 - distance  # Convert distance to confidence
                    print(f"     [Debug] ✅ Recognized as {name}")
                else:
                    print(f"     [Debug] ⚠️ Distance {distance:.3f} > 0.6 threshold, treating as Unknown")

            matches.append((name, confidence, (left, top, right, bottom)))
            
        return matches

