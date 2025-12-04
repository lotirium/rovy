#!/usr/bin/env python3
"""
Test if RPLIDAR C1 motor starts automatically when scanning begins.
Some RPLIDAR models control the motor automatically via software.
"""
import sys
sys.path.insert(0, '/home/rovy/rovy_client')

from lidar_scan import RPLIDAR
import time

def main():
    print("=" * 60)
    print("RPLIDAR C1 Auto Motor Test")
    print("=" * 60)
    print("\nThis test will try to start scanning WITHOUT GPIO motor control.")
    print("Watch the motor - it might start automatically when scanning begins.\n")
    
    lidar = RPLIDAR(motor_pin=None)  # No GPIO pin
    
    if not lidar.connect():
        sys.exit(1)
    
    try:
        # Get device info
        print("Getting device information...")
        info = lidar.get_info()
        if info:
            print(f"  Model ID: {info['model_id']}")
            print(f"  Firmware: {info['firmware']}")
        
        # Reset
        print("\nResetting device...")
        lidar.reset()
        
        # Try software motor control command
        print("\nTrying software motor control command...")
        try:
            lidar.ser.reset_input_buffer()
            lidar.ser.reset_output_buffer()
            # Motor PWM command: 0xA5 0xF0 [PWM value]
            motor_cmd = bytes([0xA5, 0xF0, 0xFF])  # Full speed
            lidar.ser.write(motor_cmd)
            time.sleep(0.5)
            print("  ✓ Software motor command sent")
            print("  Watch the motor - did it start spinning?")
            time.sleep(3)
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        # Try starting scan (motor might start automatically)
        print("\nStarting scan (motor may start automatically)...")
        if lidar.start_scan():
            print("  ✓ Scan command sent")
            print("  Watch the motor - did it start spinning now?")
            print("  Waiting 5 seconds...")
            time.sleep(5)
            
            # Try to read some data
            print("\nReading scan data...")
            points = lidar.read_scan_data(max_points=5)
            if points:
                print(f"  ✓ Received {len(points)} scan points")
                print("  If you see scan data, the LiDAR is working!")
                print("  The motor should be spinning if it's working correctly.")
            else:
                print("  ⚠ No scan data received")
        else:
            print("  ✗ Failed to start scan")
        
        # Stop
        print("\nStopping...")
        lidar.stop()
        
        print("\n" + "=" * 60)
        print("Summary:")
        print("  - If motor started: It's controlled automatically or via software")
        print("  - If motor didn't start: Check physical MOTOCTL connection")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        lidar.disconnect()

if __name__ == "__main__":
    main()

