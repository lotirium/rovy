"""
Meeting Summarization Service
Handles audio upload, transcription, summarization, and storage.
"""
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)

# Storage directory for meetings
MEETINGS_DIR = Path(__file__).parent.parent / "meetings"
MEETINGS_DB = MEETINGS_DIR / "meetings.json"


class MeetingService:
    """Service for managing meeting recordings and summaries."""
    
    def __init__(self, speech_processor=None, assistant=None):
        """
        Initialize meeting service.
        
        Args:
            speech_processor: SpeechProcessor instance for transcription
            assistant: CloudAssistant instance for summarization
        """
        self.speech_processor = speech_processor
        self.assistant = assistant
        
        # Ensure meetings directory exists
        MEETINGS_DIR.mkdir(exist_ok=True)
        
        # Load meetings database
        self.meetings = self._load_meetings()
        
        logger.info(f"✅ Meeting service initialized ({len(self.meetings)} meetings)")
    
    def _load_meetings(self) -> dict:
        """Load meetings from JSON database."""
        if MEETINGS_DB.exists():
            try:
                with open(MEETINGS_DB, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load meetings database: {e}")
        return {}
    
    def _save_meetings(self):
        """Save meetings to JSON database."""
        try:
            with open(MEETINGS_DB, 'w') as f:
                json.dump(self.meetings, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save meetings database: {e}")
    
    async def process_audio(
        self,
        audio_bytes: bytes,
        filename: str = "recording.wav",
        title: Optional[str] = None,
        meeting_type: str = "meeting"
    ) -> dict:
        """
        Process audio file: transcribe and summarize.
        
        Args:
            audio_bytes: Raw audio data
            filename: Original filename
            title: Optional meeting title
            meeting_type: Type of meeting (meeting, lecture, conversation, note)
        
        Returns:
            dict with meeting data
        """
        meeting_id = str(uuid.uuid4())
        logger.info(f"Processing meeting {meeting_id}: {filename}")
        
        # Save audio file
        audio_path = MEETINGS_DIR / f"{meeting_id}.wav"
        try:
            with open(audio_path, 'wb') as f:
                f.write(audio_bytes)
            logger.info(f"Saved audio to {audio_path}")
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return {
                "success": False,
                "meeting_id": meeting_id,
                "message": f"Failed to save audio: {e}"
            }
        
        # Transcribe audio
        transcript = None
        if self.speech_processor:
            try:
                logger.info("Transcribing audio...")
                # Detect audio format and sample rate
                import wave
                try:
                    with wave.open(str(audio_path), 'rb') as wav:
                        sample_rate = wav.getframerate()
                        logger.info(f"Audio sample rate: {sample_rate} Hz")
                except:
                    sample_rate = 16000  # Default
                
                transcript = self.speech_processor.transcribe(audio_bytes, sample_rate=sample_rate)
                
                if transcript:
                    logger.info(f"Transcription complete: {len(transcript)} chars")
                else:
                    logger.warning("Transcription returned empty")
                    transcript = "[Transcription failed - audio may be silent or unclear]"
            except Exception as e:
                logger.error(f"Transcription error: {e}")
                transcript = f"[Transcription error: {e}]"
        else:
            transcript = "[Speech processor not available]"
            logger.warning("Speech processor not configured")
        
        # Generate summary
        summary = None
        if self.assistant and transcript and not transcript.startswith("["):
            try:
                logger.info("Generating summary with OpenAI...")
                
                # Create summarization prompt
                prompt = f"""Please summarize the following {meeting_type} transcript concisely. 
Focus on key points, decisions, and action items. Keep the summary to 2-3 sentences.

Transcript:
{transcript}

Summary:"""
                
                summary = self.assistant.ask(prompt, max_tokens=200, temperature=0.5, disable_tools=True)
                logger.info(f"Summary generated: {len(summary)} chars")
            except Exception as e:
                logger.error(f"Summarization error: {e}")
                summary = f"Summary generation failed: {e}"
        else:
            # Fallback: use beginning of transcript
            if transcript and not transcript.startswith("["):
                summary = transcript[:200] + "..." if len(transcript) > 200 else transcript
            else:
                summary = transcript or "No summary available"
        
        # Generate title if not provided
        if not title:
            if self.assistant and summary and not summary.startswith("["):
                try:
                    title_prompt = f"Generate a short 3-5 word title for this {meeting_type}: {summary[:200]}"
                    title = self.assistant.ask(title_prompt, max_tokens=20, temperature=0.3, disable_tools=True)
                    title = title.strip('"').strip("'")
                except:
                    title = f"{meeting_type.title()} - {datetime.now().strftime('%b %d, %Y')}"
            else:
                title = f"{meeting_type.title()} - {datetime.now().strftime('%b %d, %Y')}"
        
        # Calculate duration (approximate from audio file size)
        duration = len(audio_bytes) / (16000 * 2)  # Assuming 16kHz, 16-bit
        
        # Store meeting
        meeting_data = {
            "id": meeting_id,
            "title": title,
            "type": meeting_type,
            "content": summary,
            "transcript": transcript,
            "date": datetime.now().isoformat(),
            "duration": round(duration, 2),
            "audio_filename": filename
        }
        
        self.meetings[meeting_id] = meeting_data
        self._save_meetings()
        
        logger.info(f"✅ Meeting {meeting_id} processed and stored")
        
        return {
            "success": True,
            "meeting_id": meeting_id,
            "message": "Meeting processed successfully",
            "meeting": meeting_data
        }
    
    def get_all_meetings(self) -> List[dict]:
        """Get all meetings sorted by date (newest first)."""
        meetings = list(self.meetings.values())
        meetings.sort(key=lambda x: x.get("date", ""), reverse=True)
        return meetings
    
    def get_meeting(self, meeting_id: str) -> Optional[dict]:
        """Get a specific meeting by ID."""
        return self.meetings.get(meeting_id)
    
    def delete_meeting(self, meeting_id: str) -> bool:
        """Delete a meeting and its audio file."""
        if meeting_id in self.meetings:
            # Delete audio file
            audio_path = MEETINGS_DIR / f"{meeting_id}.wav"
            try:
                if audio_path.exists():
                    audio_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete audio file: {e}")
            
            # Remove from database
            del self.meetings[meeting_id]
            self._save_meetings()
            
            logger.info(f"Deleted meeting {meeting_id}")
            return True
        return False

