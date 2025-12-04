#!/usr/bin/env python3
"""
Check if RPLIDAR C1 is connected and working at /dev/ttyUSB0
Test various RPLIDAR protocol commands and signals
"""
import os
import sys
import time
import struct

def check_device_exists():
    """Check if the device file exists."""
    device = "/dev/ttyUSB0"
    if os.path.exists(device):
        print(f"✓ Device {device} exists")
        # Check permissions
        stat_info = os.stat(device)
        print(f"  Permissions: {oct(stat_info.st_mode)[-3:]}")
        return True
    else:
        print(f"✗ Device {device} does not exist")
        return False

def format_bytes(data):
    """Format bytes for display."""
    if not data:
        return "None"
    hex_str = ' '.join(f'{b:02X}' for b in data)
    ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data)
    return f"{hex_str}  ({ascii_str})"

def send_command(ser, cmd_byte, description, wait_time=0.2, read_bytes=100):
    """Send a RPLIDAR command and read response."""
    print(f"\n  → {description}")
    print(f"    Command: 0xA5 0x{cmd_byte:02X}")
    
    # Clear buffers first
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(0.05)
    
    # Send command (RPLIDAR protocol: 0xA5 + command byte)
    command = bytes([0xA5, cmd_byte])
    ser.write(command)
    
    # Wait for response
    time.sleep(wait_time)
    
    # Read available data
    if ser.in_waiting > 0:
        data = ser.read(min(ser.in_waiting, read_bytes))
        print(f"    ✓ Response: {len(data)} bytes")
        print(f"    Data: {format_bytes(data)}")
        
        # Try to parse response header if we have enough bytes
        if len(data) >= 7:
            # RPLIDAR response format: 
            # Byte 0: 0xA5 (sync byte)
            # Byte 1: Response type
            # Byte 2-3: Data length (little endian)
            # Byte 4-5: Checksum
            # Byte 6+: Data
            sync = data[0]
            resp_type = data[1]
            data_len = struct.unpack('<H', data[2:4])[0]
            checksum = struct.unpack('<H', data[4:6])[0]
            print(f"    Parsed: sync=0x{sync:02X}, type=0x{resp_type:02X}, len={data_len}, checksum=0x{checksum:04X}")
        
        return data
    else:
        print(f"    ⚠ No response received")
        return None

def test_spontaneous_data(ser, duration=2.0):
    """Check if device sends data spontaneously."""
    print(f"\n  → Checking for spontaneous data (waiting {duration}s)...")
    ser.reset_input_buffer()
    start_time = time.time()
    data_received = []
    
    while time.time() - start_time < duration:
        if ser.in_waiting > 0:
            chunk = ser.read(ser.in_waiting)
            data_received.append(chunk)
            print(f"    Received {len(chunk)} bytes: {format_bytes(chunk[:50])}")
        time.sleep(0.1)
    
    if data_received:
        total = b''.join(data_received)
        print(f"    ✓ Total: {len(total)} bytes received")
        return total
    else:
        print(f"    ⚠ No spontaneous data")
        return None

def test_rplidar_commands():
    """Test various RPLIDAR protocol commands."""
    device = "/dev/ttyUSB0"
    
    try:
        import serial
    except ImportError:
        print("✗ pyserial not installed. Install with: pip install pyserial")
        return False
    
    # Try different baud rates - RPLIDAR C1 might use different rates
    baud_rates = [115200, 256000, 230400, 460800, 921600, 9600]
    
    for baud_rate in baud_rates:
        try:
            print(f"\n{'='*60}")
            print(f"Testing at {baud_rate} baud")
            print(f"{'='*60}")
            
            ser = serial.Serial(
                port=device,
                baudrate=baud_rate,
                timeout=2.0,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            print(f"✓ Serial connection opened at {baud_rate} baud")
            
            # Give device time to initialize
            time.sleep(0.5)
            
            # First, check if device sends data spontaneously
            spontaneous = test_spontaneous_data(ser, duration=1.0)
            if spontaneous:
                print(f"    ✓ Device is sending data! This baud rate works.")
                ser.close()
                # Reopen for command testing
                ser = serial.Serial(port=device, baudrate=baud_rate, timeout=2.0)
                time.sleep(0.3)
            
            # RPLIDAR Protocol Commands:
            # 0x20 - Start scan (normal)
            # 0x21 - Start scan (express)
            # 0x25 - Stop
            # 0x40 - Reset
            # 0x50 - Get device info
            # 0x52 - Get device health
            # 0x59 - Get sample rate
            
            print(f"\n{'='*60}")
            print("Testing RPLIDAR Protocol Commands")
            print(f"{'='*60}")
            
            # 1. Stop command (safe to send first)
            send_command(ser, 0x25, "STOP - Stop scanning", wait_time=0.3)
            
            # 2. Get device info
            info_data = send_command(ser, 0x50, "GET_DEVICE_INFO - Get device information", wait_time=0.5, read_bytes=20)
            found_working = False
            if info_data and len(info_data) >= 7 and info_data[0] == 0xA5:
                # Device info structure (20 bytes):
                # Bytes 0-1: Model ID
                # Bytes 2-3: Firmware version (minor.major)
                # Bytes 4-5: Hardware version
                # Bytes 6-19: Serial number (14 bytes)
                if len(info_data) >= 11:
                    model_id = struct.unpack('<H', info_data[7:9])[0] if len(info_data) >= 9 else 0
                    fw_minor = info_data[9] if len(info_data) > 9 else 0
                    fw_major = info_data[10] if len(info_data) > 10 else 0
                    print(f"    Device Model ID: {model_id}")
                    print(f"    Firmware: {fw_major}.{fw_minor}")
                found_working = True
            
            # 3. Get device health
            health_data = send_command(ser, 0x52, "GET_DEVICE_HEALTH - Get device health status", wait_time=0.5, read_bytes=10)
            if health_data and len(health_data) >= 7 and health_data[0] == 0xA5:
                # Health status: 0=OK, 1=Warning, 2=Error
                status = health_data[7] if len(health_data) > 7 else -1
                status_names = {0: "OK", 1: "Warning", 2: "Error"}
                print(f"    Health Status: {status_names.get(status, 'Unknown')} ({status})")
                found_working = True
            
            # 4. Get sample rate
            rate_data = send_command(ser, 0x59, "GET_SAMPLE_RATE - Get sample rate", wait_time=0.5, read_bytes=10)
            if rate_data and len(rate_data) >= 7 and rate_data[0] == 0xA5:
                found_working = True
            
            # 5. Try to start scan (express mode)
            scan_data = send_command(ser, 0x21, "START_SCAN_EXPRESS - Start express scan mode", wait_time=0.5, read_bytes=50)
            if scan_data and len(scan_data) >= 7 and scan_data[0] == 0xA5:
                found_working = True
            
            # If scan started, try to read a few scan packets
            if scan_data:
                print(f"\n  → Reading scan data packets...")
                time.sleep(0.3)
                for i in range(5):
                    if ser.in_waiting > 0:
                        packet = ser.read(min(ser.in_waiting, 100))
                        print(f"    Packet {i+1}: {len(packet)} bytes - {format_bytes(packet[:20])}")
                        time.sleep(0.1)
                    else:
                        break
            
            # 6. Stop scan
            send_command(ser, 0x25, "STOP - Stop scanning", wait_time=0.3)
            
            # 7. Try normal scan mode
            scan_normal = send_command(ser, 0x20, "START_SCAN_NORMAL - Start normal scan mode", wait_time=0.5, read_bytes=50)
            if scan_normal and len(scan_normal) >= 7 and scan_normal[0] == 0xA5:
                found_working = True
            
            # 8. Stop again
            try:
                send_command(ser, 0x25, "STOP - Stop scanning", wait_time=0.3)
            except:
                pass  # Ignore errors when stopping
            
            # 9. Try alternative command format (some devices need different format)
            print(f"\n  → Trying alternative command formats...")
            
            # Try without sync byte
            print(f"    Testing: Direct command 0x50 (no sync byte)")
            ser.reset_input_buffer()
            ser.write(bytes([0x50]))
            time.sleep(0.3)
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                print(f"    ✓ Response: {format_bytes(data)}")
            
            # Try with different sync byte
            print(f"    Testing: 0xAA 0x50 (alternative sync)")
            ser.reset_input_buffer()
            ser.write(bytes([0xAA, 0x50]))
            time.sleep(0.3)
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                print(f"    ✓ Response: {format_bytes(data)}")
            
            # Try reading continuously for a bit
            print(f"\n  → Continuous read test (2 seconds)...")
            ser.reset_input_buffer()
            start = time.time()
            packets = []
            while time.time() - start < 2.0:
                if ser.in_waiting > 0:
                    pkt = ser.read(min(ser.in_waiting, 100))
                    packets.append(pkt)
                    print(f"    Packet: {len(pkt)} bytes - {format_bytes(pkt[:30])}")
                time.sleep(0.05)
            
            if packets:
                print(f"    ✓ Received {len(packets)} packets during continuous read")
            
            try:
                ser.close()
            except:
                pass
            
            # If we got any responses at this baud rate, it's likely correct
            if found_working or spontaneous or packets:
                print(f"\n{'='*60}")
                print(f"✓ Found working baud rate: {baud_rate}")
                print(f"{'='*60}")
                return True
            else:
                print(f"  ⚠ No responses at {baud_rate} baud, trying next...")
                continue
                
        except serial.SerialException as e:
            print(f"  ✗ Serial error at {baud_rate}: {e}")
            continue
        except Exception as e:
            print(f"  ✗ Error at {baud_rate}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("⚠ No responses received at any baud rate")
    print(f"{'='*60}")
    print("\nPossible reasons:")
    print("1. Device may need specific initialization sequence")
    print("2. Device might be in a different mode")
    print("3. Device might use a different protocol")
    print("4. Check if device documentation specifies different commands")
    return False

def check_usb_info():
    """Check USB device information."""
    device = "/dev/ttyUSB0"
    
    try:
        import subprocess
        # Get USB device info using udevadm
        result = subprocess.run(
            ['udevadm', 'info', '--name=/dev/ttyUSB0', '--query=all'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("\nUSB Device Information:")
            for line in result.stdout.split('\n'):
                if 'ID_VENDOR' in line or 'ID_MODEL' in line or 'ID_SERIAL' in line:
                    print(f"  {line}")
    except Exception as e:
        print(f"Could not get USB info: {e}")

def main():
    print("=" * 60)
    print("RPLIDAR C1 Connection Check")
    print("=" * 60)
    
    # Check if device exists
    if not check_device_exists():
        sys.exit(1)
    
    # Check USB info
    check_usb_info()
    
    # Test RPLIDAR commands
    if test_rplidar_commands():
        print("\n" + "=" * 60)
        print("✓ RPLIDAR C1 communication test completed!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("✗ Could not establish communication with RPLIDAR C1")
        print("=" * 60)
        print("\nTroubleshooting:")
        print("1. Check if device is properly connected")
        print("2. Check permissions: sudo usermod -a -G dialout $USER")
        print("3. Try running with sudo: sudo python3 check_lidar.py")
        print("4. Check if another process is using the device")
        print("5. Verify the device is an RPLIDAR C1 (not a different model)")
        sys.exit(1)

if __name__ == "__main__":
    main()

