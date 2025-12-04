#!/usr/bin/env python3
"""
Test script to help identify the correct GPIO pin for RPLIDAR C1 motor control.
This will cycle through common GPIO pins to help you find which one controls the motor.
"""
import sys
import time

# Try gpiod first (works on newer Pi models like Pi 5)
GPIO_AVAILABLE = False
GPIO_LIB = None
try:
    import gpiod
    GPIO_AVAILABLE = True
    GPIO_LIB = 'gpiod'
except ImportError:
    try:
        import RPi.GPIO as GPIO
        GPIO_AVAILABLE = True
        GPIO_LIB = 'RPi'
    except ImportError:
        print("✗ No GPIO library available. Install gpiod or RPi.GPIO")
        print("  For Raspberry Pi 5: pip install gpiod")
        print("  For older Pi models: pip install RPi.GPIO")
        sys.exit(1)

# Common GPIO pins used for motor control
COMMON_PINS = [18, 19, 12, 13, 5, 6, 16, 20, 21, 26]

def test_pin(pin):
    """Test a GPIO pin by setting it high."""
    try:
        if GPIO_LIB == 'gpiod':
            # Try different ways to open the chip
            chip = None
            for chip_name in ['/dev/gpiochip0', 'gpiochip0', 'gpiochip4']:
                try:
                    chip = gpiod.Chip(chip_name)
                    break
                except:
                    continue
            
            if chip is None:
                raise Exception("Could not open gpiochip. Check permissions (user should be in 'gpio' group)")
            
            # New gpiod API (v2.x) uses request_lines
            config = gpiod.LineSettings()
            config.direction = gpiod.line.Direction.OUTPUT
            config.output_value = gpiod.line.Value.ACTIVE
            
            lines = chip.request_lines(
                consumer='rplidar_test',
                config={pin: config}
            )
            
            lines.set_values({pin: gpiod.line.Value.ACTIVE})
            print(f"  ✓ Set GPIO {pin} HIGH (motor should spin if this is the correct pin)")
            time.sleep(2)
            lines.set_values({pin: gpiod.line.Value.INACTIVE})
            lines.release()
            chip.close()
        else:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.HIGH)
            print(f"  ✓ Set GPIO {pin} HIGH (motor should spin if this is the correct pin)")
            time.sleep(2)
            GPIO.output(pin, GPIO.LOW)
            GPIO.cleanup()
        return True
    except PermissionError as e:
        print(f"  ✗ Permission denied for GPIO {pin}")
        print(f"     Try: sudo usermod -a -G gpio $USER")
        print(f"     Then log out and back in, or run with: sudo python3 test_motor_pin.py")
        return False
    except Exception as e:
        print(f"  ✗ Error with GPIO {pin}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("RPLIDAR C1 Motor Pin Tester")
    print("=" * 60)
    
    # Check permissions
    import os
    if GPIO_LIB == 'gpiod' and not os.access('/dev/gpiochip0', os.R_OK | os.W_OK):
        print("\n⚠ WARNING: May not have permission to access GPIO")
        print("  If you get permission errors, run:")
        print("    sudo usermod -a -G gpio $USER")
        print("  Then log out and back in, or run with sudo\n")
    
    print("\nThis script will test common GPIO pins.")
    print("Watch your LiDAR motor - it should start spinning when the correct pin is activated.")
    print("\nPress Ctrl+C to stop at any time.\n")
    
    if len(sys.argv) > 1:
        # Test specific pin
        try:
            pin = int(sys.argv[1])
            print(f"Testing GPIO pin {pin}...")
            test_pin(pin)
            print(f"\n✓ Tested GPIO {pin}")
            print("Did the motor start spinning? If yes, use this pin number in lidar_scan.py")
        except ValueError:
            print(f"Invalid pin number: {sys.argv[1]}")
    else:
        # Test common pins
        print("Testing common GPIO pins (2 seconds each)...")
        print("Watch for the motor to start spinning!\n")
        
        for pin in COMMON_PINS:
            print(f"Testing GPIO {pin}...")
            test_pin(pin)
            time.sleep(0.5)
            
            response = input("  Did the motor spin? (y/n/q to quit): ").lower()
            if response == 'y':
                print(f"\n✓ Found it! Use GPIO {pin} as the motor pin.")
                print(f"  Run: python3 lidar_scan.py --motor-pin {pin}")
                break
            elif response == 'q':
                break
        
        print("\n✓ Testing complete")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        if GPIO_AVAILABLE:
            if GPIO_LIB == 'RPi':
                try:
                    GPIO.cleanup()
                except:
                    pass
            # gpiod doesn't need cleanup - lines are released individually

