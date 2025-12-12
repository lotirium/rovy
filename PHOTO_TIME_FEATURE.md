# Photo Time Feature

The Photo Time feature allows you to capture photos using the OakD camera and print them directly to a Kodak Mini 2 Era M200 printer via Bluetooth.

## Features

- **Live Camera Stream**: View real-time video feed from the OakD camera
- **Photo Capture**: Take high-quality photos with OakD camera
- **Voice Commands**: Control photo capture and printing with voice
- **Photo Preview**: Review photos before printing
- **Bluetooth Printing**: Print directly to Kodak Mini 2 Era M200 printer
- **Mobile Interface**: Beautiful, intuitive UI in the mobile app

## Setup

### 1. Install Dependencies

#### Mobile App
```bash
cd mobile
npm install
```

The following dependency has been added:
- `@react-native-voice/voice@^3.2.4` - for voice command recognition

#### Robot/Raspberry Pi
```bash
cd robot
pip install -r requirements.txt
```

Make sure DepthAI (OakD camera library) is installed:
```bash
pip install depthai
```

### 2. Kodak Printer Setup

1. **Turn on your Kodak Mini 2 Era M200 printer**
   - Ensure Bluetooth is enabled
   - The printer typically advertises as "PM-210" or similar

2. **Pair with mobile device** (optional but recommended)
   - Go to your device's Bluetooth settings
   - Look for "PM-210" or "Kodak" devices
   - Pair if prompted (some printers don't require pairing)

3. **Load photo paper**
   - Make sure the printer has photo paper loaded
   - The Kodak Mini 2 uses 2.1" x 3.4" (54mm x 86mm) photo paper

### 3. Mobile App Configuration

The Photo Time tab will automatically appear in the mobile app after rebuilding:

```bash
cd mobile
# For Android
npx expo run:android

# For iOS
npx expo run:ios
```

## Usage

### Using the Mobile App

1. **Open Photo Time Tab**
   - Launch the mobile app
   - Navigate to the "Photo Time" tab (camera icon)

2. **View Live Stream**
   - The OakD camera stream will appear automatically
   - If not connected, check that the robot is online

3. **Take a Photo**
   - **Button**: Tap the large "Take Photo" button
   - **Voice**: Tap the "Voice" button and say "Take picture" or "Take photo"

4. **Review Photo**
   - Photo preview will appear automatically
   - Review the captured image

5. **Print Photo**
   - **Button**: Tap the "Print" button in preview
   - **Voice**: Say "Print" or "Print it"
   - Wait for the printer to process and print

6. **Retake Photo** (if needed)
   - Tap "Retake" button or say "Retake"

### Voice Commands

The following voice commands are supported:

| Command | Action |
|---------|--------|
| "Take picture" | Capture a photo from OakD camera |
| "Take photo" | Capture a photo from OakD camera |
| "Print" | Print the captured photo |
| "Print it" | Print the captured photo |
| "Retake" | Discard and take a new photo |
| "Take another" | Discard and take a new photo |
| "Cancel" | Close preview without printing |

### Using Voice Commands via Cloud AI

You can also use voice commands through the main voice chat feature:

1. Go to "Voice Chat" in the app
2. Say "Hey Jarvis" (or your configured wake word)
3. Say "Take a picture"
4. The robot will capture and confirm the photo

## API Endpoints

### Capture Photo (uses existing `/shot` endpoint)
```http
GET http://{robot_ip}:8000/shot
```

**Response:**
- Content-Type: `image/jpeg`
- Raw JPEG image bytes from OakD camera

**Note:** This endpoint was already part of the system and is used for AI vision. The Photo Time feature reuses this existing endpoint.

### Print Photo (Note: Currently handled client-side)
```http
POST http://{robot_ip}:8000/photo/print
Content-Type: application/json

{
  "image": "base64_encoded_jpeg_image"
}
```

**Note:** This is a placeholder endpoint. Actual printing is handled by the mobile app's Bluetooth connection to the Kodak printer.

## Architecture

### Components

1. **Mobile App** (`/mobile/app/(tabs)/photo-time.tsx`)
   - Camera stream display via WebSocket
   - Photo capture UI
   - Voice command integration
   - Photo preview modal
   - Print controls

2. **Kodak Printer Manager** (`/mobile/services/kodak-printer.ts`)
   - Bluetooth Low Energy (BLE) communication
   - Printer device scanning and connection
   - Image data transfer to printer
   - Handles Kodak Mini 2 Era M200 protocol

3. **Robot API** (`/robot/main_api.py`)
   - Uses existing `/shot` endpoint for OakD photo capture
   - `/photo/print` endpoint (placeholder for future server-side printing)
   - OakD camera pipeline managed by existing robot camera system

4. **Cloud Voice Handler** (`/cloud/app/main.py`)
   - Voice command recognition for "take picture"
   - Integration with robot photo capture
   - TTS feedback to user

### Data Flow

```
User -> Voice Command -> Cloud Server -> Robot /shot API -> OakD Camera
                                                                 |
                                                                 v
User <- TTS Feedback <- Cloud Server <----------------------- [JPEG Data]

User -> Mobile App -> "Take Photo" Button -> Robot /shot API -> OakD Camera
                                                                     |
                                                                     v
User <- Photo Preview <- JPEG Data (converted to base64) <------- [JPEG Data]

User -> Mobile App -> "Print" Button -> Kodak Printer (BLE) -> [Printed Photo]
```

## Troubleshooting

### Camera Not Streaming
- Check robot is online and connected
- Verify OakD camera is connected to Raspberry Pi
- Check WebSocket connection in browser console
- Restart robot service: `sudo systemctl restart rovy`

### Photo Capture Fails
- Ensure DepthAI library is installed: `pip install depthai`
- Check OakD camera USB connection
- Verify camera permissions
- Check robot logs: `journalctl -u rovy -f`

### Printer Not Found
- Make sure Kodak printer is turned on
- Enable Bluetooth on mobile device
- Check printer is in range (within 10 meters)
- Try scanning again after waiting 10 seconds
- Restart printer

### Printer Won't Print
- Check photo paper is loaded
- Ensure printer has battery power
- Verify Bluetooth connection is stable
- Try reconnecting to printer
- Check printer manual for error indicators

### Voice Commands Not Working
- Ensure microphone permissions are granted
- Check "Voice" button is activated (blue highlight)
- Speak clearly and close to device
- Try button-based capture if voice fails
- Check voice recognition in system settings

## Technical Notes

### OakD Camera Resolution
- Captures at 4K resolution for maximum quality
- Images are JPEG compressed at 95% quality
- Typical image size: 2-4 MB

### Bluetooth Printing
- Uses BLE (Bluetooth Low Energy) for connection
- Images sent in 512-byte chunks
- Print time: ~30-60 seconds depending on image complexity
- Kodak Mini 2 expects images ideally at 1280x1920 pixels

### Performance
- Camera stream: ~10 FPS over WebSocket
- Photo capture: ~1-2 seconds
- Image transfer to mobile: ~1-2 seconds
- Bluetooth printing: ~30-60 seconds

## Future Enhancements

- [ ] Server-side Bluetooth printing (direct from Raspberry Pi)
- [ ] Photo filters and effects
- [ ] Multiple photo capture (photo booth mode)
- [ ] Photo gallery to view captured photos
- [ ] Social sharing integration
- [ ] QR code printing for sharing
- [ ] Collage and multi-photo layouts
- [ ] Automatic photo adjustment (brightness, contrast)

## Credits

- OakD Camera: Luxonis DepthAI
- Kodak Mini 2 Era M200: Kodak
- Voice Recognition: @react-native-voice/voice
- BLE Communication: react-native-ble-plx

## License

Part of the ROVY robot project.

