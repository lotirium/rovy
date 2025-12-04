#!/usr/bin/env python3
"""
Simple dual camera test:
- USB camera streams continuously
- OAK-D captures ONE snapshot
- Sends to cloud AI with "What do you see?"
- Gets and displays answer
"""

import sys
import time
import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "cloud"))

# Stats
stats = {
    "usb_frames": 0,
    "oakd_captured": False,
    "ai_response": None,
    "start_time": None
}


async def stream_usb_continuously(duration=20):
    """Stream USB camera continuously for specified duration."""
    try:
        import cv2
        LOGGER.info("🎥 Starting USB camera continuous stream...")
        
        # Open USB camera using stable path
        cap = None
        by_id_dir = Path("/dev/v4l/by-id")
        
        if by_id_dir.is_dir():
            for entry in sorted(by_id_dir.iterdir()):
                name = entry.name.lower()
                if "oak" in name or "depthai" in name or "luxonis" in name:
                    continue
                if "usb" in name and "video-index0" in name:
                    device = str(entry)
                    cap = cv2.VideoCapture(device)
                    if cap.isOpened():
                        LOGGER.info(f"✅ USB camera opened: {entry.name}")
                        break
        
        # Fallback to indices
        if cap is None or not cap.isOpened():
            for idx in [0, 1, 2, 3]:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    LOGGER.info(f"✅ USB camera opened on index {idx}")
                    break
        
        if cap is None or not cap.isOpened():
            LOGGER.error("❌ Could not open USB camera")
            return
        
        # Stream continuously
        end_time = time.time() + duration
        while time.time() < end_time:
            ret, frame = cap.read()
            if ret and frame is not None:
                stats["usb_frames"] += 1
                # Log every 30 frames
                if stats["usb_frames"] % 30 == 0:
                    elapsed = time.time() - stats["start_time"]
                    fps = stats["usb_frames"] / elapsed if elapsed > 0 else 0
                    LOGGER.info(f"📹 USB streaming: {stats['usb_frames']} frames ({fps:.1f} fps)")
            
            await asyncio.sleep(0.033)  # ~30 fps
        
        cap.release()
        LOGGER.info(f"✅ USB stream complete: {stats['usb_frames']} total frames")
        
    except Exception as e:
        LOGGER.error(f"❌ USB stream error: {e}", exc_info=True)


async def capture_oakd_and_query_cloud():
    """Capture ONE frame from OAK-D and query cloud AI."""
    try:
        import depthai as dai
        import cv2
        
        # Wait a bit for USB camera to start first
        await asyncio.sleep(2)
        
        LOGGER.info("📸 Initializing OAK-D for single capture...")
        
        # Create pipeline using SAME code as main.py (known working)
        pipeline = dai.Pipeline()
        
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setIspScale(1, 3)  # Downscale to 640x360
        cam_rgb.setFps(30)
        
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("preview")
        xout_rgb.setMetadataOnly(False)
        cam_rgb.preview.link(xout_rgb.input)
        
        # Start device
        LOGGER.info("🔧 Starting OAK-D device...")
        device = dai.Device(pipeline)
        queue = device.getOutputQueue(name="preview", maxSize=4, blocking=False)
        
        LOGGER.info("✅ OAK-D ready, capturing frame...")
        
        # Capture frame (try multiple times)
        frame = None
        for attempt in range(20):
            frame_data = queue.get() if attempt < 10 else queue.tryGet()
            if frame_data is not None:
                frame = frame_data.getCvFrame()
                if frame is not None and frame.size > 0:
                    stats["oakd_captured"] = True
                    LOGGER.info(f"✅ OAK-D frame captured: {frame.shape}")
                    break
            await asyncio.sleep(0.1)
        
        if frame is None or not stats["oakd_captured"]:
            LOGGER.error("❌ Failed to capture OAK-D frame")
            device.close()
            return
        
        # Send to cloud AI
        LOGGER.info("🤖 Sending frame to cloud AI...")
        try:
            from ai import CloudAssistant
            
            # Initialize assistant
            LOGGER.info("Initializing CloudAssistant...")
            assistant = CloudAssistant()
            
            # Validate frame before encoding
            if frame is None or frame.size == 0:
                LOGGER.error("❌ Invalid frame")
                device.close()
                return
            
            LOGGER.info(f"Frame info: shape={frame.shape}, dtype={frame.dtype}")
            
            # Encode frame to JPEG with validation
            LOGGER.info("Encoding frame to JPEG...")
            success, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success or encoded is None:
                LOGGER.error("❌ Failed to encode frame")
                device.close()
                return
            
            image_bytes = encoded.tobytes()
            if len(image_bytes) == 0:
                LOGGER.error("❌ Encoded image is empty")
                device.close()
                return
            
            LOGGER.info(f"📤 Sending {len(image_bytes)} bytes to AI...")
            
            # Query AI with retry logic (in thread to not block)
            question = "What do you see in this image?"
            loop = asyncio.get_event_loop()
            
            # Run with explicit error handling
            try:
                LOGGER.info("Calling ask_with_vision...")
                response = await loop.run_in_executor(
                    None,
                    assistant.ask_with_vision,
                    question,
                    image_bytes
                )
                
                LOGGER.info(f"Received response: {response}")
                
                # Check if response indicates an error
                if response.startswith("Error:") or response.startswith("Vision query failed"):
                    LOGGER.error(f"❌ AI returned error: {response}")
                    stats["ai_response"] = None
                else:
                    stats["ai_response"] = response
                    
                    # Display result
                    print("\n" + "="*70)
                    print("🤖 AI VISION ANALYSIS")
                    print("="*70)
                    print(f"Question: {question}")
                    print(f"Answer:   {response}")
                    print("="*70 + "\n")
                    
                    LOGGER.info("✅ AI analysis complete!")
                    
            except Exception as query_error:
                LOGGER.error(f"❌ Vision query execution failed: {query_error}", exc_info=True)
                stats["ai_response"] = None
            
        except ImportError as e:
            LOGGER.error(f"❌ CloudAssistant not available: {e}", exc_info=True)
            stats["ai_response"] = None
        except Exception as e:
            LOGGER.error(f"❌ AI setup failed: {e}", exc_info=True)
            stats["ai_response"] = None
        
        device.close()
        LOGGER.info("🔒 OAK-D device closed")
        
    except Exception as e:
        LOGGER.error(f"❌ OAK-D error: {e}", exc_info=True)


async def main():
    """Run the test."""
    print("="*70)
    print("DUAL CAMERA TEST: USB Stream + OAK-D AI Vision")
    print("="*70)
    print("- USB camera will stream continuously")
    print("- OAK-D will capture ONE snapshot")
    print("- AI will analyze the OAK-D frame")
    print("="*70)
    print()
    
    stats["start_time"] = time.time()
    
    # Run both tasks concurrently
    LOGGER.info("🚀 Starting test...")
    
    try:
        await asyncio.gather(
            stream_usb_continuously(duration=20),
            capture_oakd_and_query_cloud()
        )
    except KeyboardInterrupt:
        LOGGER.info("\n⚠️  Test interrupted by user")
    
    # Final stats
    elapsed = time.time() - stats["start_time"]
    usb_fps = stats["usb_frames"] / elapsed if elapsed > 0 else 0
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Duration:      {elapsed:.1f}s")
    print(f"USB frames:    {stats['usb_frames']} ({usb_fps:.1f} fps)")
    print(f"OAK-D capture: {'✅ Success' if stats['oakd_captured'] else '❌ Failed'}")
    print(f"AI response:   {'✅ Received' if stats['ai_response'] else '❌ Not received'}")
    print("="*70)
    
    # Verdict
    if stats["usb_frames"] > 100 and stats["oakd_captured"] and stats["ai_response"]:
        print("\n🎉 SUCCESS! USB streamed continuously while OAK-D captured and AI analyzed!")
        return 0
    else:
        print("\n❌ Test incomplete.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

