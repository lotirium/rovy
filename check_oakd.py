#!/usr/bin/env python3
"""
Check if OAK-D camera is connected and working.
Tests DepthAI SDK connection and device enumeration.
"""
import sys
import subprocess
import os

def check_usb_devices():
    """Check for OAK-D devices via USB enumeration."""
    print("🔌 Checking USB devices...")
    try:
        result = subprocess.run(
            ["lsusb"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.lower()
        # OAK-D devices show up as "Intel Movidius MyriadX" with vendor ID 03e7:2485
        if "luxonis" in lines or "oak" in lines or "depthai" in lines or "movidius" in lines or "03e7:2485" in result.stdout:
            print("✓ Found OAK-D related device in USB enumeration:")
            for line in result.stdout.split("\n"):
                if any(keyword in line.lower() for keyword in ["luxonis", "oak", "depthai", "movidius"]) or "03e7:2485" in line:
                    print(f"  {line}")
            return True
        else:
            print("✗ No OAK-D device found in USB enumeration (lsusb)")
            return False
    except FileNotFoundError:
        print("⚠ lsusb command not available")
        return None
    except Exception as e:
        print(f"⚠ Error checking USB devices: {e}")
        return None

def check_v4l_devices():
    """Check for OAK-D devices via V4L2."""
    print("\n📹 Checking V4L2 devices...")
    v4l_path = "/dev/v4l/by-id"
    if not os.path.exists(v4l_path):
        print(f"⚠ {v4l_path} does not exist")
        return False
    
    try:
        devices = os.listdir(v4l_path)
        oak_devices = [
            d for d in devices
            if any(keyword in d.lower() for keyword in ["oak", "depthai", "luxonis"])
        ]
        
        if oak_devices:
            print(f"✓ Found {len(oak_devices)} OAK-D device(s) in V4L2:")
            for dev in oak_devices:
                full_path = os.path.join(v4l_path, dev)
                target = os.readlink(full_path)
                print(f"  {dev} -> {target}")
            return True
        else:
            print("✗ No OAK-D device found in V4L2 devices")
            return False
    except Exception as e:
        print(f"⚠ Error checking V4L2 devices: {e}")
        return False

def check_depthai_installed():
    """Check if depthai package is installed."""
    try:
        import depthai as dai
        print("✓ DepthAI package is installed")
        try:
            version = getattr(dai, "__version__", "unknown")
            print(f"  Version: {version}")
        except:
            pass
        return True, dai
    except ImportError:
        print("✗ DepthAI package is not installed")
        print("  Install with: pip install depthai")
        return False, None

def check_devices(dai):
    """Check for available OAK-D devices."""
    if dai is None:
        return False
    
    try:
        print("\n🔍 Searching for OAK-D devices...")
        available = dai.Device.getAllAvailableDevices()
        
        if len(available) == 0:
            print("✗ No OAK-D devices found")
            print("  Make sure the OAK-D is:")
            print("    - Connected via USB")
            print("    - Powered on")
            print("    - Not being used by another process")
            return False
        
        print(f"✓ Found {len(available)} OAK-D device(s):")
        for i, dev_info in enumerate(available, 1):
            mxid = dev_info.getMxId()
            state = dev_info.getState()
            print(f"  Device {i}:")
            print(f"    MXID: {mxid}")
            print(f"    State: {state}")
            
            # Try to get more info if device is available
            if state == dai.XLinkDeviceState.X_LINK_UNBOOTED:
                try:
                    device = dai.Device(dev_info)
                    usb_speed = device.getUsbSpeed()
                    print(f"    USB Speed: {usb_speed}")
                    
                    # Get camera info
                    cameras = device.getConnectedCameras()
                    print(f"    Connected Cameras: {cameras}")
                    
                    device.close()
                    print(f"    ✓ Device {i} is accessible and working")
                except Exception as e:
                    print(f"    ⚠ Device {i} found but cannot be accessed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error checking for devices: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_capture(dai):
    """Test capturing a frame from OAK-D."""
    if dai is None:
        return False
    
    try:
        print("\n📸 Testing frame capture...")
        
        # Create a simple pipeline
        pipeline = dai.Pipeline()
        
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("preview")
        cam_rgb.preview.link(xout_rgb.input)
        
        # Connect to device
        device = dai.Device(pipeline)
        queue = device.getOutputQueue(name="preview", maxSize=1, blocking=False)
        
        print("  Waiting for frame...")
        import time
        for attempt in range(10):
            frame_data = queue.tryGet()
            if frame_data is not None:
                frame = frame_data.getCvFrame()
                if frame is not None:
                    height, width = frame.shape[:2]
                    print(f"✓ Successfully captured frame: {width}x{height}")
                    device.close()
                    return True
            time.sleep(0.1)
        
        print("✗ Failed to capture frame after 10 attempts")
        device.close()
        return False
        
    except Exception as e:
        print(f"✗ Error testing capture: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("=" * 60)
    print("  OAK-D Connection Check")
    print("=" * 60)
    
    # Check USB devices first (doesn't require Python packages)
    usb_found = check_usb_devices()
    v4l_found = check_v4l_devices()
    
    # Check if DepthAI is installed
    dai_installed, dai = check_depthai_installed()
    
    if not dai_installed:
        print("\n" + "=" * 60)
        if usb_found or v4l_found:
            print("⚠️  OAK-D device detected but DepthAI SDK not installed")
            print("   Install with: pip install depthai")
            print("   Or install all requirements: pip install -r cloud/requirements.txt")
        else:
            print("❌ OAK-D is NOT connected")
            print("   - No device found in USB enumeration")
            print("   - No device found in V4L2")
        sys.exit(1)
    
    # Check for devices using DepthAI SDK
    devices_found = check_devices(dai)
    if not devices_found:
        print("\n" + "=" * 60)
        print("❌ OAK-D is NOT connected or not accessible")
        if usb_found or v4l_found:
            print("   Device detected but DepthAI SDK cannot access it")
            print("   - Device may be in use by another process")
            print("   - Try unplugging and replugging the device")
        sys.exit(1)
    
    # Test capture
    capture_works = test_capture(dai)
    
    print("\n" + "=" * 60)
    if capture_works:
        print("✅ OAK-D is connected and working!")
        sys.exit(0)
    else:
        print("⚠️  OAK-D found but frame capture failed")
        sys.exit(1)

if __name__ == "__main__":
    main()

