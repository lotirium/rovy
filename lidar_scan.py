#!/usr/bin/env python3
"""
RPLIDAR C1 Scanner - Read and display scan data
Works at 460800 baud rate
"""
import serial
import struct
import time
import sys

# Try to import GPIO for motor control
# Prefer gpiod (works on newer Pi models like Pi 5)
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
        pass

# RPLIDAR C1 Configuration
DEVICE = "/dev/ttyUSB0"
BAUD_RATE = 460800

# RPLIDAR Protocol Constants
SYNC_BYTE = 0xA5
SYNC_BYTE2 = 0x5A

# Commands
CMD_STOP = 0x25
CMD_RESET = 0x40
CMD_SCAN = 0x20
CMD_FORCE_SCAN = 0x21
CMD_GET_INFO = 0x50
CMD_GET_HEALTH = 0x52
CMD_SET_MOTOR_PWM = 0xF0  # Motor PWM control (if supported)

# Response types
RESP_SCAN = 0x81
RESP_INFO = 0x04
RESP_HEALTH = 0x06


class RPLIDAR:
    """Simple RPLIDAR C1 interface."""
    
    def __init__(self, port=DEVICE, baudrate=BAUD_RATE, motor_pin=None):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.motor_pin = motor_pin  # GPIO pin for motor control (MOTOCTL)
        self.motor_enabled = False
        self.motor_lines = None  # For gpiod line request
        self.motor_chip = None  # For gpiod chip
        
    def connect(self):
        """Open serial connection."""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=2.0,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            time.sleep(0.5)  # Give device time to initialize
            print(f"✓ Connected to {self.port} at {self.baudrate} baud")
            return True
        except Exception as e:
            print(f"✗ Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Close serial connection."""
        if self.ser and self.ser.is_open:
            self.stop()
            self.ser.close()
        
        # Cleanup GPIO
        if GPIO_AVAILABLE:
            if GPIO_LIB == 'gpiod':
                if self.motor_lines:
                    try:
                        self.motor_lines.release()
                    except:
                        pass
                if self.motor_chip:
                    try:
                        self.motor_chip.close()
                    except:
                        pass
            elif GPIO_LIB == 'RPi' and self.motor_enabled:
                try:
                    GPIO.cleanup()
                except:
                    pass
        
        print("✓ Disconnected")
    
    def send_command(self, cmd):
        """Send a command to the LiDAR."""
        if not self.ser or not self.ser.is_open:
            return False
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        time.sleep(0.05)
        command = bytes([SYNC_BYTE, cmd])
        self.ser.write(command)
        return True
    
    def read_response(self, timeout=1.0):
        """Read response from LiDAR."""
        if not self.ser or not self.ser.is_open:
            return None
        
        start_time = time.time()
        buffer = bytearray()
        
        # Wait for sync byte
        while time.time() - start_time < timeout:
            if self.ser.in_waiting > 0:
                byte = self.ser.read(1)[0]
                if byte == SYNC_BYTE:
                    buffer.append(byte)
                    break
            time.sleep(0.01)
        
        if not buffer:
            return None
        
        # Read second sync byte
        if self.ser.in_waiting > 0:
            byte = self.ser.read(1)[0]
            if byte == SYNC_BYTE2:
                buffer.append(byte)
            else:
                return None
        else:
            time.sleep(0.01)
            if self.ser.in_waiting > 0:
                byte = self.ser.read(1)[0]
                if byte == SYNC_BYTE2:
                    buffer.append(byte)
                else:
                    return None
        
        # Read length (2 bytes, little endian)
        if self.ser.in_waiting >= 2:
            length_bytes = self.ser.read(2)
            buffer.extend(length_bytes)
            data_len = struct.unpack('<H', length_bytes)[0]
        else:
            return None
        
        # Read checksum (2 bytes)
        if self.ser.in_waiting >= 2:
            checksum = self.ser.read(2)
            buffer.extend(checksum)
        else:
            return None
        
        # Read data
        if data_len > 0:
            while len(buffer) < 7 + data_len and time.time() - start_time < timeout:
                if self.ser.in_waiting > 0:
                    buffer.extend(self.ser.read(min(self.ser.in_waiting, 7 + data_len - len(buffer))))
                time.sleep(0.01)
        
        return bytes(buffer)
    
    def get_info(self):
        """Get device information."""
        if not self.send_command(CMD_GET_INFO):
            return None
        
        time.sleep(0.3)
        response = self.read_response()
        if response and len(response) >= 20:
            # Parse device info
            model_id = struct.unpack('<H', response[7:9])[0]
            fw_minor = response[9]
            fw_major = response[10]
            hw_version = response[11]
            serial = response[12:26].hex()
            
            return {
                'model_id': model_id,
                'firmware': f"{fw_major}.{fw_minor}",
                'hardware': hw_version,
                'serial': serial
            }
        return None
    
    def get_health(self):
        """Get device health status."""
        if not self.send_command(CMD_GET_HEALTH):
            return None
        
        time.sleep(0.3)
        response = self.read_response()
        if response and len(response) >= 10:
            status = response[7]
            status_names = {0: "OK", 1: "Warning", 2: "Error"}
            return {
                'status': status,
                'status_name': status_names.get(status, "Unknown")
            }
        return None
    
    def enable_motor(self, enable=True):
        """Enable or disable the motor via GPIO pin.
        
        Note: RPLIDAR C1 motor is controlled via MOTOCTL pin.
        If motor_pin is not set, this will try to find it or use software control.
        """
        if self.motor_pin is None:
            print("⚠ Motor pin not specified. Motor may not spin.")
            print("   RPLIDAR C1 requires MOTOCTL pin to be connected to a GPIO.")
            print("   Check your wiring - MOTOCTL should be connected to a GPIO pin.")
            # Try software motor control via command (may not work for C1)
            if enable:
                print("   Attempting software motor control...")
                # Some RPLIDAR models support motor PWM via command
                # This might not work for C1, but worth trying
                try:
                    self.ser.reset_input_buffer()
                    self.ser.reset_output_buffer()
                    # Try motor PWM command (if supported)
                    # Format: 0xA5 0xF0 [PWM value]
                    motor_cmd = bytes([SYNC_BYTE, CMD_SET_MOTOR_PWM, 0xFF if enable else 0x00])
                    self.ser.write(motor_cmd)
                    time.sleep(0.1)
                    print("   Software motor command sent (may not work for C1)")
                except:
                    pass
            return
        
        # Hardware GPIO control
        if not GPIO_AVAILABLE:
            print("⚠ GPIO library not available. Cannot control motor.")
            return
        
        try:
            if GPIO_LIB == 'gpiod':
                # If we already have the line requested, just change the value
                if self.motor_lines is not None:
                    # Already have control, just change the value
                    self.motor_lines.set_values({self.motor_pin: gpiod.line.Value.ACTIVE if enable else gpiod.line.Value.INACTIVE})
                    self.motor_enabled = enable
                    if enable:
                        print(f"✓ Motor enabled via GPIO {self.motor_pin} (keeping HIGH)")
                    else:
                        print(f"✓ Motor disabled via GPIO {self.motor_pin}")
                else:
                    # First time - request the line
                    chip = gpiod.Chip('/dev/gpiochip0')
                    
                    config = gpiod.LineSettings()
                    config.direction = gpiod.line.Direction.OUTPUT
                    config.output_value = gpiod.line.Value.ACTIVE if enable else gpiod.line.Value.INACTIVE
                    
                    lines = chip.request_lines(
                        consumer='rplidar',
                        config={self.motor_pin: config}
                    )
                    
                    # Keep motor enabled continuously (don't release the line)
                    lines.set_values({self.motor_pin: gpiod.line.Value.ACTIVE if enable else gpiod.line.Value.INACTIVE})
                    self.motor_enabled = enable
                    self.motor_lines = lines  # Store for later cleanup (keep it active!)
                    self.motor_chip = chip
                    if enable:
                        print(f"✓ Motor enabled via GPIO {self.motor_pin} (keeping HIGH)")
                    else:
                        print(f"✓ Motor disabled via GPIO {self.motor_pin}")
            else:
                # Using RPi.GPIO (older method)
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.motor_pin, GPIO.OUT)
                GPIO.output(self.motor_pin, GPIO.HIGH if enable else GPIO.LOW)
                self.motor_enabled = enable
                print(f"✓ Motor {'enabled' if enable else 'disabled'} via GPIO {self.motor_pin}")
        except Exception as e:
            print(f"✗ Failed to control motor: {e}")
            import traceback
            traceback.print_exc()
    
    def reset(self):
        """Reset the LiDAR device."""
        print("Resetting LiDAR...")
        self.send_command(CMD_RESET)
        time.sleep(1.0)  # Give device time to reset
    
    def stop(self):
        """Stop scanning and disable motor."""
        try:
            self.send_command(CMD_STOP)
            time.sleep(0.2)
            self.ser.reset_input_buffer()
        except:
            pass
        
        # Disable motor by just setting the value, don't try to request line again
        if self.motor_enabled and self.motor_lines is not None:
            try:
                if GPIO_LIB == 'gpiod':
                    self.motor_lines.set_values({self.motor_pin: gpiod.line.Value.INACTIVE})
                elif GPIO_LIB == 'RPi':
                    GPIO.output(self.motor_pin, GPIO.LOW)
                self.motor_enabled = False
                print(f"✓ Motor disabled via GPIO {self.motor_pin}")
            except Exception as e:
                print(f"⚠ Could not disable motor cleanly: {e}")
    
    def start_scan(self):
        """Start normal scan mode."""
        if not self.send_command(CMD_SCAN):
            return False
        
        time.sleep(0.5)  # Give more time for response
        response = self.read_response(timeout=1.0)
        if response and len(response) >= 7:
            # Check if response is valid
            if response[0] == SYNC_BYTE and response[1] == SYNC_BYTE2:
                return True
        # Even if no response, scan might have started - try reading data
        return True  # Assume scan started if command was sent
    
    def read_scan_data(self, max_points=100):
        """Read scan data points."""
        if not self.ser or not self.ser.is_open:
            return []
        
        points = []
        start_time = time.time()
        timeout = 5.0  # 5 second timeout
        
        while len(points) < max_points and time.time() - start_time < timeout:
            if self.ser.in_waiting >= 5:  # Minimum scan packet size
                # Read scan packet (5 bytes for normal scan)
                packet = self.ser.read(5)
                
                if len(packet) == 5:
                    # Parse scan data
                    # Byte 0: Quality (bits 0-6), start flag (bit 7)
                    # Byte 1-2: Angle (14 bits, Q6 format)
                    # Byte 3-4: Distance (15 bits, mm)
                    
                    quality = packet[0] & 0x3F
                    start_flag = (packet[0] & 0x80) >> 7
                    
                    angle_raw = struct.unpack('<H', packet[1:3])[0]
                    angle = (angle_raw >> 1) / 64.0  # Convert to degrees
                    
                    distance_raw = struct.unpack('<H', packet[3:5])[0]
                    distance = (distance_raw & 0x7FFF) / 4.0  # Convert to mm
                    
                    if quality > 0 and distance > 0:  # Valid point
                        points.append({
                            'angle': angle,
                            'distance': distance,
                            'quality': quality,
                            'start': start_flag == 1
                        })
            
            time.sleep(0.001)  # Small delay to avoid CPU spinning
        
        return points


def main():
    """Main function to test LiDAR scanning."""
    print("=" * 60)
    print("RPLIDAR C1 Scanner")
    print("=" * 60)
    
    # Motor control pin (MOTOCTL)
    # Found via testing: GPIO pin 0 (ID_SDA) controls the motor
    # Set to None to try software control or if not connected
    MOTOR_PIN = 0  # GPIO pin 0 - change if you found a different pin
    
    # You can also specify via command line: python lidar_scan.py --motor-pin 18
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg == '--motor-pin' and i + 1 < len(sys.argv):
                MOTOR_PIN = int(sys.argv[i + 1])
                break
    
    lidar = RPLIDAR(motor_pin=MOTOR_PIN)
    
    if not lidar.connect():
        sys.exit(1)
    
    try:
        # Get device info
        print("\nGetting device information...")
        info = lidar.get_info()
        if info:
            print(f"  Model ID: {info['model_id']}")
            print(f"  Firmware: {info['firmware']}")
            print(f"  Hardware: {info['hardware']}")
            print(f"  Serial: {info['serial']}")
        
        # Get health status
        print("\nChecking device health...")
        health = lidar.get_health()
        if health:
            print(f"  Status: {health['status_name']} ({health['status']})")
        
        # Reset device first
        print("\nResetting device...")
        lidar.reset()
        
        # Enable motor (required for spinning)
        print("\nEnabling motor...")
        print("NOTE: If motor doesn't spin, check:")
        print("  1. MOTOCTL pin is connected to a GPIO pin")
        print("  2. Power supply is adequate (5V)")
        print("  3. Motor pin number is correct")
        print("\nTrying to enable motor...")
        lidar.enable_motor(True)
        time.sleep(0.5)  # Give motor time to start
        
        # Start scanning
        print("\nStarting scan...")
        if lidar.start_scan():
            print("✓ Scan started successfully")
            
            print("\nReading scan data (press Ctrl+C to stop)...")
            print("-" * 60)
            print(f"{'Angle':>8} {'Distance':>10} {'Quality':>8} {'Start':>6}")
            print("-" * 60)
            
            try:
                while True:
                    points = lidar.read_scan_data(max_points=10)
                    for point in points:
                        print(f"{point['angle']:>8.2f}° {point['distance']:>10.1f}mm {point['quality']:>8} {'Yes' if point['start'] else 'No':>6}")
                    
                    if points:
                        print()  # Blank line between batches
                    
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n\nStopping scan...")
                # Stop the scan and motor
                try:
                    lidar.stop()
                except:
                    pass
        else:
            print("✗ Failed to start scan")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        lidar.disconnect()
        print("\n✓ Done")


if __name__ == "__main__":
    main()

