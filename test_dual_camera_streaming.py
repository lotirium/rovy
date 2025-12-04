#!/usr/bin/env python3
"""
Test USB streaming + OAK-D capture + Cloud AI vision simultaneously.
- USB camera streams continuously
- OAK-D captures snapshots periodically
- Sends OAK-D frames to cloud AI for analysis
"""

import sys
import time
import asyncio
import base64
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

# Stats tracking
stats = {
    "usb_frames": 0,
    "oakd_frames": 0,
    "ai_queries": 0,
    "errors": 0,
    "start_time": None
}


async def stream_usb_camera(stop_event):
    """Continuously stream USB camera in background."""
    try:
        import cv2
        LOGGER.info("🎥 Starting USB camera stream...")
        
        # Try to open USB camera (skip OAK-D devices)
        by_id_dir = Path("/dev/v4l/by-id")
        cap = None
        
        if by_id_dir.is_dir():
            for entry in sorted(by_id_dir.iterdir()):
                name = entry.name.lower()
                if "oak" in name or "depthai" in name or "luxonis" in name:
                    continue
                if "usb" in name and "video-index0" in name:
                    device = str(entry)
                    LOGGER.info(f"Trying USB camera: {device}")
                    cap = cv2.VideoCapture(device)
                    if cap.isOpened():
                        LOGGER.info(f"✅ Opened USB camera: {entry.name}")
                        break
        
        # Fallback to numeric indices
        if cap is None or not cap.isOpened():
            for idx in [1, 2, 0, 3]:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    LOGGER.info(f"✅ Opened USB camera on index {idx}")
                    break
        
        if cap is None or not cap.isOpened():
            LOGGER.error("❌ Could not open USB camera")
            return
        
        # Stream continuously
        while not stop_event.is_set():
            ret, frame = cap.read()
            if ret and frame is not None:
                stats["usb_frames"] += 1
                if stats["usb_frames"] % 30 == 0:  # Log every 30 frames
                    LOGGER.info(f"📹 USB streaming: {stats['usb_frames']} frames ({frame.shape})")
            else:
                LOGGER.warning("⚠️  USB frame read failed")
                stats["errors"] += 1
            
            await asyncio.sleep(0.033)  # ~30 FPS
        
        cap.release()
        LOGGER.info("🛑 USB camera stream stopped")
        
    except Exception as e:
        LOGGER.error(f"❌ USB stream error: {e}", exc_info=True)
        stats["errors"] += 1


async def capture_oakd_and_query_ai(interval=5):
    """Periodically capture from OAK-D and query cloud AI."""
    try:
        import depthai as dai
        import cv2
        
        LOGGER.info("🔧 Initializing OAK-D camera...")
        
        # Create pipeline
        pipeline = dai.Pipeline()
        
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setIspScale(1, 3)  # Downscale for speed
        cam_rgb.setFps(30)
        
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("preview")
        xout_rgb.setMetadataOnly(False)
        cam_rgb.preview.link(xout_rgb.input)
        
        # Start device
        device = dai.Device(pipeline)
        queue = device.getOutputQueue(name="preview", maxSize=4, blocking=False)
        
        LOGGER.info("✅ OAK-D camera initialized")
        
        # Import AI assistant
        try:
            from ai import CloudAssistant
            assistant = CloudAssistant()
            LOGGER.info("✅ Cloud AI assistant loaded")
        except Exception as e:
            LOGGER.warning(f"⚠️  Cloud AI not available: {e}")
            assistant = None
        
        # Capture and query periodically
        query_count = 0
        while True:
            await asyncio.sleep(interval)
            
            # Capture OAK-D frame
            frame_data = queue.tryGet()
            if frame_data is None:
                # Wait a bit and retry
                await asyncio.sleep(0.1)
                frame_data = queue.tryGet()
            
            if frame_data is not None:
                frame = frame_data.getCvFrame()
                if frame is not None and frame.size > 0:
                    stats["oakd_frames"] += 1
                    LOGGER.info(f"📸 OAK-D captured frame {stats['oakd_frames']}: {frame.shape}")
                    
                    # Send to AI for analysis
                    if assistant:
                        try:
                            # Encode frame to JPEG
                            success, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            if success:
                                image_bytes = encoded.tobytes()
                                
                                # Query AI
                                query_count += 1
                                question = "What do you see in this image? Describe it briefly."
                                LOGGER.info(f"🤖 Asking AI (query #{query_count}): {question}")
                                
                                # Run in thread to not block
                                loop = asyncio.get_event_loop()
                                response = await loop.run_in_executor(
                                    None,
                                    assistant.ask_with_vision,
                                    question,
                                    image_bytes
                                )
                                
                                stats["ai_queries"] += 1
                                LOGGER.info(f"💬 AI Response #{query_count}: {response}")
                                print(f"\n{'='*60}")
                                print(f"🤖 AI VISION ANALYSIS #{query_count}")
                                print(f"{'='*60}")
                                print(f"Question: {question}")
                                print(f"Answer:   {response}")
                                print(f"{'='*60}\n")
                        except Exception as e:
                            LOGGER.error(f"❌ AI query failed: {e}")
                            stats["errors"] += 1
            else:
                LOGGER.warning("⚠️  No OAK-D frame available")
                stats["errors"] += 1
        
    except Exception as e:
        LOGGER.error(f"❌ OAK-D error: {e}", exc_info=True)
        stats["errors"] += 1


async def print_stats():
    """Print statistics periodically."""
    while True:
        await asyncio.sleep(10)
        if stats["start_time"]:
            elapsed = time.time() - stats["start_time"]
            usb_fps = stats["usb_frames"] / elapsed if elapsed > 0 else 0
            print(f"\n{'='*60}")
            print(f"📊 STATISTICS (elapsed: {elapsed:.1f}s)")
            print(f"{'='*60}")
            print(f"USB frames:    {stats['usb_frames']} ({usb_fps:.1f} fps)")
            print(f"OAK-D frames:  {stats['oakd_frames']}")
            print(f"AI queries:    {stats['ai_queries']}")
            print(f"Errors:        {stats['errors']}")
            print(f"{'='*60}\n")


async def main():
    """Run the dual camera test."""
    print("="*60)
    print("DUAL CAMERA STREAMING + AI VISION TEST")
    print("="*60)
    print("USB Camera: Continuous streaming")
    print("OAK-D:      Periodic capture + AI analysis")
    print("="*60)
    print()
    
    stats["start_time"] = time.time()
    
    # Create stop event for USB stream
    stop_event = asyncio.Event()
    
    # Start tasks
    tasks = [
        asyncio.create_task(stream_usb_camera(stop_event)),
        asyncio.create_task(capture_oakd_and_query_ai(interval=5)),
        asyncio.create_task(print_stats())
    ]
    
    try:
        # Run for 30 seconds
        LOGGER.info("🚀 Starting test (will run for 30 seconds)...")
        await asyncio.sleep(30)
        
        LOGGER.info("⏱️  Test duration reached, stopping...")
        stop_event.set()
        
        # Wait a bit for cleanup
        await asyncio.sleep(2)
        
    except KeyboardInterrupt:
        LOGGER.info("\n⚠️  Interrupted by user")
        stop_event.set()
    finally:
        # Cancel tasks
        for task in tasks:
            task.cancel()
        
        # Wait for tasks to finish
        await asyncio.gather(*tasks, return_exceptions=True)
    
    # Final stats
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    elapsed = time.time() - stats["start_time"]
    usb_fps = stats["usb_frames"] / elapsed if elapsed > 0 else 0
    print(f"Duration:      {elapsed:.1f}s")
    print(f"USB frames:    {stats['usb_frames']} ({usb_fps:.1f} fps)")
    print(f"OAK-D frames:  {stats['oakd_frames']}")
    print(f"AI queries:    {stats['ai_queries']}")
    print(f"Errors:        {stats['errors']}")
    print(f"{'='*60}")
    
    # Verdict
    if stats["usb_frames"] > 0 and stats["oakd_frames"] > 0 and stats["ai_queries"] > 0:
        print("\n🎉 SUCCESS! Both cameras worked simultaneously with AI analysis!")
        return 0
    else:
        print("\n❌ FAILED! Some components didn't work properly.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

