#!/usr/bin/env python3
"""
Voice Localization Demo (Lightweight)
Demonstrates using ReSpeaker DOA to locate voice and OAK-D to detect person
No LLaVA - for faster testing!
"""
import time
import cv2
from smart_assistant import ReSpeakerInterface, TextToSpeech, WakeWordDetector
from rover_controller import Rover
from oakd_depth_navigator import OakDDepthCamera
from face_recognizer import FaceRecognizer

# ============================================================================
# CALIBRATION CONFIGURATION
# ============================================================================
# Angle offset between ReSpeaker microphone and camera servo mounting positions
# If your microphone is tilted/rotated relative to the camera, adjust this value
# Example: If mic's 90° (left) should map to camera's 120°, set offset to 30
MICROPHONE_TO_CAMERA_OFFSET = -121  # Degrees to add to DOA angle before converting to servo angle
# Manual calibration based on two test points:
# LEFT: Mic ~203° → Camera -90° (LEFT) ✅
# RIGHT: Mic ~31° → Camera 90° (RIGHT) - adjusted offset to -121°

def main():
    print("\n" + "="*70)
    print("🎯 Voice Localization + Person Detection Demo")
    print("="*70)
    print("\nThis demo will:")
    print("1. Use ReSpeaker's DOA to detect where your voice is coming from")
    print("2. Automatically move the camera to point towards your voice")
    print("3. Use OAK-D's person detection to verify you're there")
    print("\nMake sure you:")
    print("  - Have ReSpeaker Mic Array v2.0 connected")
    print("  - Have OAK-D camera connected")
    print("  - Rover controller is connected")
    print("="*70 + "\n")
    
    # Initialize components
    print("[Setup] Initializing rover controller...")
    rover = Rover()
    
    print("[Setup] Initializing OAK-D camera with YOLOv8 person detection...")
    camera = OakDDepthCamera(resolution=(640, 352), enable_person_detection=True)  # Match YOLOv8 model resolution
    camera.start()
    
    print("[Setup] Initializing ReSpeaker...")
    from smart_assistant import ReSpeakerInterface, TextToSpeech, WakeWordDetector
    
    respeaker = ReSpeakerInterface(use_whisper=False)
    tts = TextToSpeech(print_only=False)
    wake_detector = WakeWordDetector(device_index=respeaker.device_index)
    
    print("[Setup] Initializing face recognition...")
    face_recognizer = FaceRecognizer(known_dir="known-faces")
    
    print("\n✅ Setup complete!\n")
    
    # Initialize gimbal - IMPORTANT: Must unlock servos first
    print("[Camera] Initializing gimbal...")
    print("[Camera] Unlocking servos (T:135)...")
    rover.gimbal_unlock()
    time.sleep(0.5)
    
    print("[Camera] Centering camera to forward position...")
    rover.gimbal_ctrl_move(0, 0, input_speed_x=500, input_speed_y=500)
    time.sleep(2.0)
    
    try:
        # Main demo loop
        print("\n" + "="*70)
        print("DEMO INSTRUCTIONS:")
        print("="*70)
        print("1. Stand somewhere around the rover")
        print("2. Say 'Hey Jarvis' and keep talking for 1-2 seconds")
        print("   (e.g., 'Hey Jarvis, hello there!')")
        print("3. The rover will detect your voice direction")
        print("4. The camera will turn towards you")
        print("5. The rover will say 'Yes, I see you!'")
        print("\nPress Ctrl+C to exit")
        print("="*70 + "\n")
        
        while True:
            print("\n👂 Say 'Hey Jarvis' to activate voice localization...")
            
            # Listen for wake word
            wake_result = wake_detector.listen_for_wake_word(timeout=30)
            if wake_result:
                print("✅ Wake word detected! Capturing voice direction...")
                tts.speak("Yes?")
                
                # Get voice direction - listen a bit to capture their voice direction
                doa = respeaker.get_voice_direction(listen_duration=0.5)
                
                if doa is not None:
                    # Use the built-in locate_and_detect_person with voice-guided direction
                    print(f"🎯 Voice detected at {doa}°, turning camera and scanning for person...")
                    
                    # Apply calibration offset for microphone mounting position
                    calibrated_doa = (doa + MICROPHONE_TO_CAMERA_OFFSET) % 360
                    if calibrated_doa != doa:
                        print(f"   📐 Applying {MICROPHONE_TO_CAMERA_OFFSET}° offset: {doa}° → {calibrated_doa}°")
                    
                    # Get pan angle from voice direction
                    servo_angles = respeaker.doa_to_servo_angles(calibrated_doa, tilt_angle=0)
                    
                    if servo_angles:
                        # Turn towards voice
                        pan = servo_angles['pan']  # Use direct servo angle (no inversion)
                        print(f"   Turning to direction: {pan}°")
                        
                        # Try multiple tilt angles to find person with YOLOv8
                        person_found = False
                        found_tilt = 0
                        tilt_angles = [0, 15, 30, 40]  # Start straight, then scan up
                        
                        for tilt in tilt_angles:
                            print(f"   Scanning at tilt={tilt}°...")
                            
                            # Send movement command with timestamp
                            start_time = time.time()
                            print(f"   🔵 APPLE - Camera starting to move NOW")
                            rover.gimbal_ctrl_move(pan, tilt, input_speed_x=500, input_speed_y=500)
                            
                            # Calculate wait time based on angle distance (for first move from center)
                            if tilt == tilt_angles[0]:
                                angle_distance = abs(pan)
                                wait_time = 7.0 if angle_distance > 90 else 5.0
                                print(f"   🟡 BANANA - Waiting {wait_time}s for pan movement ({angle_distance}°)...")
                                time.sleep(wait_time)
                                print(f"   🟢 CHERRY - Camera should be stopped after {wait_time}s wait")
                            else:
                                print(f"   🟡 BANANA - Waiting 1.5s for tilt change...")
                                time.sleep(1.5)  # Subsequent tilt changes are smaller
                                print(f"   🟢 CHERRY - Tilt should be complete")
                            
                            # Fast detection with 3 frame checks (camera warm-up)
                            print(f"   🟣 DRAGON - Stabilizing 1.5s...")
                            time.sleep(1.5)  # Longer stabilization
                            print(f"   🔴 EAGLE - Checking 3 frames...")
                            
                            # Try 3 frames (camera needs warm-up after movement)
                            detections = None
                            for check in range(3):
                                time.sleep(0.3)  # Quick wait between frames
                                detections = camera.detect_person(debug=(check == 0))
                                if detections:
                                    print(f"      ✓ Detected in frame {check+1}/3")
                                    break
                            
                            # Debug: Show what we found
                            if detections:
                                print(f"   [Debug] Found {len(detections)} detection(s):")
                                for i, d in enumerate(detections):
                                    depth_str = f"{d['depth']}mm" if d['depth'] else "no depth"
                                    print(f"     #{i+1}: conf={d['confidence']:.2f}, {depth_str}")
                            else:
                                print(f"   No detections, trying next angle...")
                            
                            if detections:
                                # Accept ANY person detection - no filtering!
                                print(f"   ✅ Accepting detection without filtering")
                                best = detections[0]
                                person_found = True
                                found_tilt = tilt
                                conf = int(best['confidence'] * 100)
                                elapsed = time.time() - start_time
                                if best['depth']:
                                    depth_m = best['depth'] / 1000
                                    print(f"   ✅ [{time.strftime('%H:%M:%S')}] Person found! Confidence: {conf}%, Distance: {depth_m:.1f}m")
                                else:
                                    print(f"   ✅ [{time.strftime('%H:%M:%S')}] Person found! Confidence: {conf}%")
                                print(f"   ⏱️  Total time for this angle: {elapsed:.1f}s")
                                
                                # Try face recognition on the detected person region
                                print(f"   🔍 Recognizing face...")
                                try:
                                    rgb_frame, _ = camera.capture_frames()
                                    # Convert RGB to BGR (face_recognition lib via capstone expects BGR)
                                    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                                    
                                    # Get person bounding box and crop it
                                    bbox = best['bbox']  # (x, y, w, h) in normalized coords (0-1)
                                    height, width = bgr_frame.shape[:2]
                                    x1 = int(bbox[0] * width)
                                    y1 = int(bbox[1] * height)
                                    x2 = int((bbox[0] + bbox[2]) * width)
                                    y2 = int((bbox[1] + bbox[3]) * height)
                                    
                                    # Crop person region
                                    person_roi = bgr_frame[y1:y2, x1:x2]
                                    
                                    if person_roi.size > 0:  # Make sure crop is valid
                                        print(f"     [Debug] Cropped person ROI: {person_roi.shape}")
                                        # Pass BGR frame to face_recognition (is_rgb=False)
                                        recognized_faces = face_recognizer.recognize_face(person_roi, is_rgb=False)
                                        if recognized_faces:
                                            name, face_conf, face_bbox = recognized_faces[0]
                                            if name != "Unknown":
                                                print(f"   ✅ Recognized: {name} (confidence: {face_conf:.2f})")
                                                tts.speak(f"Yes, I see you {name}!")
                                            else:
                                                print(f"   ⚠️  Face detected but not recognized (confidence too low)")
                                                tts.speak("Yes, I see you!")
                                        else:
                                            print(f"   ⚠️  No face detected in person ROI")
                                            tts.speak("Yes, I see you!")
                                    else:
                                        print(f"   ⚠️  Invalid person crop")
                                        tts.speak("Yes, I see you!")
                                except Exception as e:
                                    print(f"   ⚠️  Face recognition error: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    tts.speak("Yes, I see you!")
                                break
                            
                            if person_found:
                                break
                        
                        if not person_found:
                            print("   ⚠️  No person visually detected, but I heard your voice!")
                            tts.speak("I heard you!")
                        
                        # Show results
                        print("\n" + "-"*70)
                        print("RESULTS:")
                        print("-"*70)
                        print(f"✅ Voice Direction: {doa}° (0°=front, 90°=left, 180°=back, 270°=right)")
                        
                        angle_map = {
                            -180: "FAR LEFT", -135: "BACK LEFT", -90: "LEFT", -45: "FRONT LEFT",
                            0: "CENTER", 45: "FRONT RIGHT", 90: "RIGHT", 135: "BACK RIGHT", 180: "FAR RIGHT"
                        }
                        pan_desc = angle_map.get(pan, f"{pan}°")
                        
                        print(f"✅ Camera panned to: {pan_desc} ({pan}°)")
                        if person_found:
                            print(f"✅ Person detected at tilt={found_tilt}°")
                        else:
                            print(f"⚠️  No person detected (scanned tilts: {tilt_angles}) - Acknowledged with 'I heard you!'")
                        print("-"*70)
                    else:
                        print("❌ Could not calculate servo position")
                else:
                    print("❌ Could not determine voice direction")
                
                # Return camera to center
                print("\n[Camera] Returning to center position...")
                rover.gimbal_ctrl_move(0, 0, input_speed_x=500, input_speed_y=500)
                time.sleep(1.5)
                
                # Ask if user wants to try again
                choice = input("\n🔄 Try again? (y/n): ").lower()
                if choice != 'y':
                    break
            else:
                print("⏰ No wake word detected, listening again...")
    
    except KeyboardInterrupt:
        print("\n\n[Demo] Interrupted by user")
    
    finally:
        # Cleanup
        print("\n[Cleanup] Shutting down...")
        
        # Turn off flash lights
        print("[Cleanup] Turning off flash lights...")
        rover.lights_ctrl(0, 0)
        
        # Return camera to center
        rover.gimbal_ctrl_move(0, 0, input_speed_x=500, input_speed_y=500)
        time.sleep(0.5)
        
        # Close camera
        if camera:
            camera.close()
        
        print("[Cleanup] Done!")
        print("\n" + "="*70)
        print("👋 Thanks for using the voice localization demo!")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()

