#!/usr/bin/env python3
"""
Automatic Microphone-Camera Angle Calibration
Automatically calculates the offset by asking you to speak from known positions.
"""
import time
from smart_assistant import ReSpeakerInterface, WakeWordDetector, TextToSpeech
from rover_controller import Rover

def main():
    print("\n" + "="*70)
    print("🎯 AUTOMATIC ANGLE OFFSET CALIBRATION")
    print("="*70)
    print("\nThis tool will AUTOMATICALLY calculate the offset!")
    print("\nHow it works:")
    print("  1. I'll ask you to stand at specific positions (front, right, back)")
    print("  2. You say 'Hey Jarvis' from each position")
    print("  3. I'll measure the angles and calculate the offset automatically")
    print("\nReady? Let's start!")
    print("="*70 + "\n")
    
    input("Press ENTER to begin...")
    
    # Initialize
    print("\n[Setup] Initializing...")
    rover = Rover()
    respeaker = ReSpeakerInterface(use_whisper=False)
    wake_detector = WakeWordDetector(device_index=respeaker.device_index)
    tts = TextToSpeech(print_only=False)
    
    print("[Camera] Centering camera...")
    rover.gimbal_unlock()
    time.sleep(0.5)
    rover.gimbal_ctrl_move(0, 0, input_speed_x=300, input_speed_y=300)
    time.sleep(2.0)
    
    print("\n✅ Ready!\n")
    
    # Test positions: (name, expected_camera_angle, instructions)
    test_positions = [
        ("FRONT", 0, "Stand directly in FRONT of the rover (where camera points now)"),
        ("RIGHT", 90, "Move to the RIGHT side of the rover (90° from front)"),
        ("BACK", 180, "Move to the BACK of the rover (directly behind it)"),
    ]
    
    measurements = []
    
    try:
        for position_name, expected_camera_angle, instructions in test_positions:
            print("\n" + "="*70)
            print(f"📍 POSITION {len(measurements)+1}/3: {position_name}")
            print("="*70)
            print(f"\n{instructions}")
            print("\nOnce you're in position:")
            print("  Say 'Hey Jarvis' clearly")
            print("="*70 + "\n")
            
            tts.speak(f"Please move to the {position_name.lower()} position")
            time.sleep(1)
            
            input("Press ENTER when you're ready to speak...")
            
            # Listen for wake word
            print(f"\n👂 Listening for 'Hey Jarvis' from {position_name}...")
            wake_result = wake_detector.listen_for_wake_word(timeout=30)
            
            if wake_result:
                print("✅ Detected! Measuring angle...")
                
                # Get voice direction
                doa = respeaker.get_voice_direction(listen_duration=1.0)
                
                if doa is not None:
                    print(f"📊 Measured microphone angle: {doa}°")
                    print(f"📊 Expected camera angle: {expected_camera_angle}°")
                    
                    # Calculate what offset would be needed
                    # We need: (doa + offset) should map to expected_camera_angle
                    # But we need to account for the servo conversion
                    
                    measurements.append({
                        'position': position_name,
                        'measured_doa': doa,
                        'expected_camera': expected_camera_angle
                    })
                    
                    print(f"✅ Recorded measurement {len(measurements)}/3\n")
                    tts.speak("Got it")
                else:
                    print("❌ Could not detect direction")
                    retry = input("Try again? (y/n): ")
                    if retry.lower() == 'y':
                        continue
                    else:
                        print("Skipping this position...")
            else:
                print("⏰ No wake word detected")
                retry = input("Try again? (y/n): ")
                if retry.lower() == 'y':
                    continue
        
        # Calculate offset from measurements
        if len(measurements) >= 2:
            print("\n" + "="*70)
            print("🧮 CALCULATING OFFSET...")
            print("="*70 + "\n")
            
            print("Measurements collected:")
            for m in measurements:
                print(f"  {m['position']}: Mic={m['measured_doa']}°, Expected Camera={m['expected_camera']}°")
            
            # Calculate offsets needed for each measurement
            # Camera pan works like: pan = -doa (inverted)
            # So if we want camera at 90° (right), we need pan = -90
            # Which comes from DOA = 270° (with inversion)
            # 
            # The relationship: expected_camera = -calibrated_doa
            # So: calibrated_doa = -expected_camera
            # And: calibrated_doa = (measured_doa + offset) % 360
            # Therefore: offset = calibrated_doa - measured_doa
            
            offsets = []
            for m in measurements:
                # We want: -calibrated_doa = expected_camera
                # So: calibrated_doa = -expected_camera
                target_doa = (-m['expected_camera']) % 360
                
                # Calculate offset needed
                offset = (target_doa - m['measured_doa'])
                
                # Normalize to -180 to 180
                if offset > 180:
                    offset -= 360
                elif offset < -180:
                    offset += 360
                
                offsets.append(offset)
                print(f"\n  {m['position']}: Need DOA={target_doa}°, Got={m['measured_doa']}° → Offset={offset}°")
            
            # Average the offsets
            avg_offset = int(round(sum(offsets) / len(offsets)))
            
            print("\n" + "="*70)
            print("✅ CALIBRATION COMPLETE!")
            print("="*70)
            print(f"\n📐 Calculated offset: {avg_offset}°")
            print(f"\nIndividual offsets: {offsets}")
            print(f"Average: {avg_offset}°")
            
            print("\n" + "="*70)
            print("TO APPLY THIS CALIBRATION:")
            print("="*70)
            print("\n1. Open: voice_localization_demo.py")
            print("2. Find line: MICROPHONE_TO_CAMERA_OFFSET = 0")
            print(f"3. Change to: MICROPHONE_TO_CAMERA_OFFSET = {avg_offset}")
            print("4. Save and run!")
            print("\n" + "="*70)
            
            tts.speak("Calibration complete!")
            
            # Test the calibration
            print("\n\nWould you like to TEST this calibration now?")
            test = input("Test it? (y/n): ")
            
            if test.lower() == 'y':
                print("\n" + "="*70)
                print("🧪 TESTING CALIBRATION")
                print("="*70)
                print("\nStand anywhere and say 'Hey Jarvis'")
                print("The camera should point at you!\n")
                
                wake_result = wake_detector.listen_for_wake_word(timeout=30)
                if wake_result:
                    doa = respeaker.get_voice_direction(listen_duration=1.0)
                    if doa is not None:
                        calibrated_doa = (doa + avg_offset) % 360
                        print(f"\n📍 Measured: {doa}° → Calibrated: {calibrated_doa}°")
                        
                        servo_angles = respeaker.doa_to_servo_angles(calibrated_doa, tilt_angle=0)
                        if servo_angles:
                            pan = -servo_angles['pan']
                            print(f"🎥 Moving camera to: {pan}°")
                            rover.gimbal_ctrl_move(pan, 0, input_speed_x=300, input_speed_y=300)
                            time.sleep(2.5)
                            tts.speak("Is it pointing at you?")
                            print("\n✅ Check if camera is pointing at you!")
        else:
            print("\n❌ Not enough measurements to calculate offset")
            print("Need at least 2 successful measurements")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    
    finally:
        rover.gimbal_ctrl_move(0, 0, input_speed_x=300, input_speed_y=300)
        time.sleep(0.5)
        print("\n👋 Done!\n")


if __name__ == "__main__":
    main()

