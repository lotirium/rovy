#!/usr/bin/env python3
"""
Test ALL available GPIO pins to find the motor control pin.
This will systematically test every GPIO pin.
"""
import sys
import time
import gpiod

def test_all_pins():
    """Test all GPIO pins on gpiochip0."""
    print("=" * 60)
    print("RPLIDAR C1 - Test ALL GPIO Pins")
    print("=" * 60)
    print("\nThis will test every GPIO pin systematically.")
    print("Watch your LiDAR motor - it should start spinning when the correct pin is activated.")
    print("\nPress Ctrl+C to stop at any time.\n")
    
    try:
        chip = gpiod.Chip('/dev/gpiochip0')
        info = chip.get_info()
        num_lines = info.num_lines
        print(f"Found {num_lines} GPIO lines on gpiochip0\n")
        
        tested = []
        for pin in range(num_lines):
            try:
                # Get line info
                line_info = chip.get_line_info(pin)
                line_name = line_info.name if line_info.name else f"GPIO{pin}"
                
                # Skip if line is already in use
                if line_info.used:
                    print(f"Pin {pin:2d} ({line_name:12s}): SKIPPED (already in use)")
                    continue
                
                print(f"Pin {pin:2d} ({line_name:12s}): Testing...", end=" ", flush=True)
                
                # Try to control the pin
                config = gpiod.LineSettings()
                config.direction = gpiod.line.Direction.OUTPUT
                config.output_value = gpiod.line.Value.ACTIVE
                
                lines = chip.request_lines(
                    consumer='rplidar_test',
                    config={pin: config}
                )
                
                lines.set_values({pin: gpiod.line.Value.ACTIVE})
                print("HIGH", end="", flush=True)
                time.sleep(1)
                
                lines.set_values({pin: gpiod.line.Value.INACTIVE})
                print(" -> LOW")
                
                lines.release()
                tested.append(pin)
                
                # Ask user
                response = input(f"  Did the motor spin? (y/n/q): ").lower().strip()
                if response == 'y':
                    print(f"\n{'='*60}")
                    print(f"✓ FOUND IT! GPIO pin {pin} ({line_name}) controls the motor!")
                    print(f"{'='*60}")
                    print(f"\nUse this pin in lidar_scan.py:")
                    print(f"  python3 lidar_scan.py --motor-pin {pin}")
                    chip.close()
                    return pin
                elif response == 'q':
                    break
                    
            except Exception as e:
                print(f"ERROR: {e}")
                continue
        
        chip.close()
        
        print(f"\n{'='*60}")
        print(f"Tested {len(tested)} GPIO pins")
        print(f"{'='*60}")
        print("\nIf none of the pins made the motor spin:")
        print("  1. Check physical wiring - MOTOCTL pin may not be connected")
        print("  2. Verify power supply (5V)")
        print("  3. Check if motor needs PWM instead of simple HIGH/LOW")
        print("  4. Consult RPLIDAR C1 documentation for wiring diagram")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        test_all_pins()
    except KeyboardInterrupt:
        print("\n\nStopped by user")

