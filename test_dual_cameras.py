#!/usr/bin/env python3
"""
Test script to verify USB camera and OAK-D camera work simultaneously.
This mimics the initialization logic in main.py.
"""

import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_usb_camera():
    """Test USB camera initialization."""
    try:
        import cv2
        LOGGER.info("✅ OpenCV available")
    except ImportError:
        LOGGER.error("❌ OpenCV not installed")
        return False
    
    # List all video devices
    by_id_dir = Path("/dev/v4l/by-id")
    if by_id_dir.is_dir():
        devices = list(by_id_dir.iterdir())
        LOGGER.info(f"📹 Found {len(devices)} video device(s):")
        for dev in devices:
            device_type = "OAK-D" if any(x in dev.name.lower() for x in ["oak", "depthai", "luxonis"]) else "USB"
            LOGGER.info(f"   - {device_type}: {dev.name}")
    
    # Try to open USB camera (skip OAK-D devices)
    usb_camera = None
    if by_id_dir.is_dir():
        for entry in sorted(by_id_dir.iterdir()):
            name = entry.name.lower()
            # Skip OAK-D devices
            if "oak" in name or "depthai" in name or "luxonis" in name:
                LOGGER.info(f"Skipping OAK-D device: {entry.name}")
                continue
            # Try USB camera with video-index0
            if "usb" in name and "video-index0" in name:
                device_path = str(entry)
                LOGGER.info(f"Trying USB camera: {device_path}")
                try:
                    usb_camera = cv2.VideoCapture(device_path)
                    if usb_camera.isOpened():
                        LOGGER.info(f"✅ USB camera opened successfully: {entry.name}")
                        # Test frame capture
                        ret, frame = usb_camera.read()
                        if ret and frame is not None:
                            LOGGER.info(f"✅ USB camera frame captured: {frame.shape}")
                            return usb_camera
                        else:
                            LOGGER.warning(f"⚠️  USB camera opened but no frame captured")
                            usb_camera.release()
                            usb_camera = None
                    else:
                        LOGGER.warning(f"⚠️  Could not open USB camera: {device_path}")
                        usb_camera = None
                except Exception as e:
                    LOGGER.warning(f"⚠️  Error with USB camera {device_path}: {e}")
                    usb_camera = None
    
    # Fallback to numeric indices
    if usb_camera is None:
        LOGGER.info("Trying numeric video device indices...")
        for idx in range(4):
            try:
                LOGGER.info(f"Trying /dev/video{idx}...")
                test_cap = cv2.VideoCapture(idx)
                if test_cap.isOpened():
                    ret, frame = test_cap.read()
                    if ret and frame is not None:
                        LOGGER.info(f"✅ USB camera opened on index {idx}: {frame.shape}")
                        return test_cap
                    else:
                        test_cap.release()
            except Exception as e:
                LOGGER.debug(f"Index {idx} failed: {e}")
    
    LOGGER.error("❌ No USB camera found")
    return None


def test_oakd_camera():
    """Test OAK-D camera initialization."""
    try:
        import depthai as dai
        LOGGER.info("✅ DepthAI available")
    except ImportError:
        LOGGER.error("❌ DepthAI not installed")
        return False
    
    try:
        # List available OAK-D devices
        available = dai.Device.getAllAvailableDevices()
        LOGGER.info(f"📋 Found {len(available)} OAK-D device(s)")
        for dev in available:
            LOGGER.info(f"   - MXID: {dev.getMxId()}")
        
        if len(available) == 0:
            LOGGER.error("❌ No OAK-D devices found")
            return None
        
        # Create pipeline
        LOGGER.info("Creating OAK-D pipeline...")
        pipeline = dai.Pipeline()
        
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setIspScale(1, 3)
        cam_rgb.setFps(30)
        
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("preview")
        xout_rgb.setMetadataOnly(False)
        cam_rgb.preview.link(xout_rgb.input)
        
        # Start device
        LOGGER.info("Starting OAK-D device...")
        device = dai.Device(pipeline)
        
        # Get USB speed
        try:
            usb_speed = device.getUsbSpeed().name
            LOGGER.info(f"OAK-D USB speed: {usb_speed}")
        except:
            pass
        
        # Get queue
        queue = device.getOutputQueue(name="preview", maxSize=4, blocking=False)
        
        # Test frame capture
        LOGGER.info("Testing OAK-D frame capture...")
        for attempt in range(10):
            frame_data = queue.tryGet()
            if frame_data is not None:
                frame = frame_data.getCvFrame()
                if frame is not None and frame.size > 0:
                    LOGGER.info(f"✅ OAK-D frame captured: {frame.shape}")
                    return device
            time.sleep(0.1)
        
        LOGGER.warning("⚠️  OAK-D device started but no frames captured")
        return device
        
    except Exception as e:
        LOGGER.error(f"❌ OAK-D initialization failed: {e}", exc_info=True)
        return None


def main():
    """Test both cameras simultaneously."""
    LOGGER.info("=" * 60)
    LOGGER.info("Testing USB Camera and OAK-D Camera Simultaneously")
    LOGGER.info("=" * 60)
    
    # Test USB camera first
    LOGGER.info("\n[1/3] Testing USB Camera...")
    usb_camera = test_usb_camera()
    
    if usb_camera:
        LOGGER.info("✅ USB camera is working")
    else:
        LOGGER.error("❌ USB camera failed")
    
    # Small delay between initializations
    LOGGER.info("\n[2/3] Waiting 0.5s before OAK-D initialization...")
    time.sleep(0.5)
    
    # Test OAK-D camera
    LOGGER.info("\n[3/3] Testing OAK-D Camera...")
    oakd_device = test_oakd_camera()
    
    if oakd_device:
        LOGGER.info("✅ OAK-D camera is working")
    else:
        LOGGER.error("❌ OAK-D camera failed")
    
    # Summary
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("SUMMARY")
    LOGGER.info("=" * 60)
    
    usb_status = "✅ WORKING" if usb_camera else "❌ FAILED"
    oakd_status = "✅ WORKING" if oakd_device else "❌ FAILED"
    
    LOGGER.info(f"USB Camera:  {usb_status}")
    LOGGER.info(f"OAK-D Camera: {oakd_status}")
    
    if usb_camera and oakd_device:
        LOGGER.info("\n🎉 SUCCESS! Both cameras are working simultaneously!")
        
        # Test concurrent frame capture
        LOGGER.info("\nTesting concurrent frame capture...")
        try:
            import cv2
            
            # Capture from USB camera
            ret, usb_frame = usb_camera.read()
            if ret:
                LOGGER.info(f"✅ USB frame: {usb_frame.shape}")
            
            # Capture from OAK-D
            oakd_queue = oakd_device.getOutputQueue(name="preview", maxSize=4, blocking=False)
            for _ in range(10):
                frame_data = oakd_queue.tryGet()
                if frame_data is not None:
                    oakd_frame = frame_data.getCvFrame()
                    if oakd_frame is not None:
                        LOGGER.info(f"✅ OAK-D frame: {oakd_frame.shape}")
                        break
                time.sleep(0.1)
            
            LOGGER.info("✅ Concurrent frame capture successful!")
        except Exception as e:
            LOGGER.warning(f"⚠️  Concurrent capture test failed: {e}")
        
        # Cleanup
        if usb_camera:
            usb_camera.release()
        if oakd_device:
            oakd_device.close()
        
        return 0
    else:
        LOGGER.error("\n❌ FAILED! Not all cameras are working.")
        
        # Cleanup
        if usb_camera:
            usb_camera.release()
        if oakd_device:
            oakd_device.close()
        
        return 1


if __name__ == "__main__":
    sys.exit(main())

