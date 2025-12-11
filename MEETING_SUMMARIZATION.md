# Meeting Summarization System

## Overview

A complete meeting recording, transcription, and summarization system integrated into the Rovy robot platform. Users can record meetings via the mobile app or voice commands, and the system automatically transcribes and summarizes them using Whisper and OpenAI GPT-4.

## Architecture

```
┌─────────────┐
│ Mobile App  │ ◄─── Upload audio files
│             │ ◄─── View summaries
└──────┬──────┘
       │
       ▼
┌─────────────┐      ┌──────────────┐
│   Robot     │ ───► │ Cloud Server │
│  (Pi 5)     │      │   (PC)       │
│             │      │              │
│ - Record    │      │ - Whisper    │
│   meetings  │      │ - OpenAI     │
│ - Voice cmd │      │ - Storage    │
└─────────────┘      └──────────────┘
```

## Features

### 1. **Recording Methods**

#### A. Voice Command (Robot)
Say "start meeting recording" to the robot:
```
User: "Hey Rovy, start meeting recording"
Robot: *starts recording*
User: "Stop meeting recording"
Robot: *uploads to cloud for processing*
```

#### B. API Endpoint (Robot)
```bash
# Start recording
POST http://robot-ip:8000/meeting/record
{
  "action": "start",
  "title": "Team Standup",
  "meeting_type": "meeting"  # meeting, lecture, conversation, note
}

# Stop recording
POST http://robot-ip:8000/meeting/record
{
  "action": "stop"
}

# Check status
GET http://robot-ip:8000/meeting/status
```

#### C. Mobile App Upload (Future)
Upload pre-recorded audio files directly from the mobile app.

### 2. **Processing Pipeline**

1. **Audio Capture**: Robot records audio using its microphone
2. **Upload**: Audio sent to cloud server as WAV file
3. **Transcription**: Whisper (base model) transcribes audio to text
4. **Summarization**: OpenAI GPT-4 generates concise summary
5. **Storage**: Meeting stored in JSON database with metadata
6. **Retrieval**: Mobile app fetches and displays summaries

### 3. **Cloud API Endpoints**

#### Upload Meeting
```bash
POST /meetings/upload
Content-Type: multipart/form-data

Fields:
  - audio: WAV file (required)
  - title: Meeting title (optional)
  - meeting_type: Type of meeting (optional, default: "meeting")

Response:
{
  "success": true,
  "meeting_id": "uuid",
  "message": "Meeting processed successfully"
}
```

#### Get All Meetings
```bash
GET /meetings

Response:
{
  "summaries": [
    {
      "id": "uuid",
      "title": "Team Standup",
      "type": "meeting",
      "content": "Summary text...",
      "transcript": "Full transcript...",
      "date": "2025-01-15T10:30:00Z",
      "duration": 300.5,
      "audio_filename": "meeting.wav"
    }
  ],
  "count": 1
}
```

#### Get Specific Meeting
```bash
GET /meetings/{meeting_id}

Response:
{
  "id": "uuid",
  "title": "Team Standup",
  ...
}
```

#### Delete Meeting
```bash
DELETE /meetings/{meeting_id}

Response:
{
  "status": "success",
  "message": "Meeting deleted"
}
```

### 4. **Mobile App Integration**

The mobile app's "Summaries" tab now:
- Fetches real meeting data from cloud API
- Displays meetings with icons based on type
- Shows loading, error, and empty states
- Supports pull-to-refresh
- Auto-formats dates

### 5. **Storage**

Meetings are stored in:
- **Database**: `cloud/meetings/meetings.json` (metadata)
- **Audio Files**: `cloud/meetings/{meeting_id}.wav` (original recordings)

## Usage Examples

### Example 1: Record Meeting via Voice

```
1. Say "Hey Rovy, start meeting recording"
2. Robot confirms and starts recording
3. Have your meeting
4. Say "Stop meeting recording"
5. Robot uploads and processes
6. View summary in mobile app
```

### Example 2: Record Meeting via API

```python
import requests
import time

robot_ip = "192.168.1.100"

# Start recording
response = requests.post(
    f"http://{robot_ip}:8000/meeting/record",
    json={
        "action": "start",
        "title": "Project Planning",
        "meeting_type": "meeting"
    }
)

# Meeting happens...
time.sleep(600)  # 10 minutes

# Stop recording
response = requests.post(
    f"http://{robot_ip}:8000/meeting/record",
    json={"action": "stop"}
)

print(response.json())
# {"status": "success", "meeting_id": "...", "duration": 600.5}
```

### Example 3: Fetch Meetings in Mobile App

The mobile app automatically fetches meetings when the Summaries tab is opened.

```typescript
// Already implemented in mobile/app/(tabs)/status.tsx
const response = await cloudApi.getMeetings();
setSummaries(response.summaries);
```

## Configuration

### Cloud Server (PC)

1. Ensure OpenAI API key is set:
```bash
export OPENAI_API_KEY="sk-..."
```

2. Whisper model is loaded automatically (base model by default)

3. Meeting service initializes on startup

### Robot (Pi 5)

1. Configure cloud API URL in `robot/config.py`:
```python
CLOUD_API_URL = "http://100.72.107.106:8001"  # Your cloud PC IP
```

2. Ensure audio device is configured:
```python
SAMPLE_RATE = 16000
CHANNELS = 1
```

## File Structure

```
cloud/
├── app/
│   ├── main.py                    # FastAPI app with meeting endpoints
│   ├── models.py                  # Pydantic models for meetings
│   └── meeting_service.py         # Meeting processing logic
├── meetings/
│   ├── meetings.json              # Meeting metadata database
│   └── {uuid}.wav                 # Audio recordings
├── speech.py                      # Whisper transcription
└── ai.py                          # OpenAI summarization

robot/
└── main_api.py                    # Robot API with recording endpoints

mobile/
├── services/
│   └── cloud-api.ts               # Cloud API client with meeting methods
└── app/(tabs)/
    └── status.tsx                 # Summaries screen with real data
```

## Technical Details

### Audio Format
- **Sample Rate**: 16 kHz (resampled if needed)
- **Channels**: Mono (1 channel)
- **Bit Depth**: 16-bit
- **Format**: WAV (PCM)

### Transcription
- **Model**: Whisper base (English only)
- **Processing**: CPU-based on cloud server
- **Accuracy**: Optimized with beam search and temperature settings

### Summarization
- **Model**: OpenAI GPT-4o
- **Prompt**: Focuses on key points, decisions, and action items
- **Length**: 2-3 sentences (configurable)
- **Temperature**: 0.5 (balanced creativity/consistency)

### Title Generation
- Automatic if not provided
- Generated by GPT-4o based on summary
- Format: "Meeting - Jan 15, 2025" (fallback)

## Performance

- **Recording**: Real-time, no lag
- **Upload**: ~1-2 seconds for 10-minute meeting
- **Transcription**: ~30 seconds for 10-minute meeting (Whisper base)
- **Summarization**: ~5 seconds (GPT-4o)
- **Total**: ~40 seconds for 10-minute meeting

## Future Enhancements

1. **Speaker Diarization**: Identify different speakers
2. **Real-time Transcription**: Live captions during meeting
3. **Action Item Extraction**: Automatically detect tasks
4. **Meeting Search**: Full-text search across transcripts
5. **Export Options**: PDF, DOCX, email summaries
6. **Multi-language Support**: Transcribe in multiple languages
7. **Meeting Analytics**: Duration, speaker time, sentiment analysis

## Troubleshooting

### Issue: "Meeting service not available"
**Solution**: Ensure OpenAI API key is set and cloud server is running

### Issue: "Transcription returned empty"
**Solution**: Check audio quality, ensure microphone is working, increase recording volume

### Issue: "Upload failed"
**Solution**: Check network connection between robot and cloud server

### Issue: "No meetings showing in app"
**Solution**: Verify cloud API URL in mobile app, check network connectivity

## API Reference

See the FastAPI documentation at `http://cloud-server:8001/docs` for interactive API documentation.

## License

Part of the Rovy robot platform.

