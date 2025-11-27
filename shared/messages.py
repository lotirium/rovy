"""
Message Protocol for Rovy Cloud System
Defines all message types exchanged between server and client
"""
import json
import base64
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class MessageType(str, Enum):
    """Types of messages exchanged between server and client"""
    # Client -> Server
    AUDIO_DATA = "audio_data"           # Raw audio for speech recognition
    IMAGE_DATA = "image_data"           # Camera frame for vision
    SENSOR_DATA = "sensor_data"         # Battery, IMU, etc.
    TEXT_QUERY = "text_query"           # Direct text input
    WAKE_WORD_DETECTED = "wake_word"    # Wake word triggered
    
    # Server -> Client
    SPEAK = "speak"                     # TTS output to play
    MOVE = "move"                       # Movement command
    GIMBAL = "gimbal"                   # Camera gimbal control
    LIGHTS = "lights"                   # LED control
    DISPLAY = "display"                 # OLED display text
    STATUS_REQUEST = "status_request"  # Request sensor data
    
    # Bidirectional
    PING = "ping"
    PONG = "pong"
    ERROR = "error"
    ACK = "ack"


@dataclass
class BaseMessage:
    """Base message structure"""
    type: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    message_id: Optional[str] = None
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, data: str) -> 'BaseMessage':
        return cls(**json.loads(data))


@dataclass
class AudioMessage(BaseMessage):
    """Audio data from microphone"""
    type: str = MessageType.AUDIO_DATA.value
    audio_base64: str = ""  # Base64 encoded audio bytes
    sample_rate: int = 16000
    channels: int = 1
    format: str = "int16"  # Audio format
    
    @classmethod
    def from_bytes(cls, audio_bytes: bytes, sample_rate: int = 16000) -> 'AudioMessage':
        return cls(
            audio_base64=base64.b64encode(audio_bytes).decode('utf-8'),
            sample_rate=sample_rate
        )
    
    def get_audio_bytes(self) -> bytes:
        return base64.b64decode(self.audio_base64)


@dataclass
class ImageMessage(BaseMessage):
    """Image data from camera"""
    type: str = MessageType.IMAGE_DATA.value
    image_base64: str = ""  # Base64 encoded JPEG
    width: int = 640
    height: int = 480
    format: str = "jpeg"
    query: Optional[str] = None  # Optional question about the image
    
    @classmethod
    def from_bytes(cls, image_bytes: bytes, width: int, height: int, query: str = None) -> 'ImageMessage':
        return cls(
            image_base64=base64.b64encode(image_bytes).decode('utf-8'),
            width=width,
            height=height,
            query=query
        )
    
    def get_image_bytes(self) -> bytes:
        return base64.b64decode(self.image_base64)


@dataclass
class SensorMessage(BaseMessage):
    """Sensor readings from rover"""
    type: str = MessageType.SENSOR_DATA.value
    battery_voltage: Optional[float] = None
    battery_percent: Optional[int] = None
    temperature: Optional[float] = None
    imu_roll: Optional[float] = None
    imu_pitch: Optional[float] = None
    imu_yaw: Optional[float] = None
    voice_direction: Optional[int] = None  # DOA angle


@dataclass
class TextQueryMessage(BaseMessage):
    """Text-based query to assistant"""
    type: str = MessageType.TEXT_QUERY.value
    text: str = ""
    include_vision: bool = False  # Whether to include current camera frame


@dataclass
class SpeakMessage(BaseMessage):
    """Command to speak text"""
    type: str = MessageType.SPEAK.value
    text: str = ""
    audio_base64: Optional[str] = None  # Pre-synthesized audio
    priority: int = 1  # 0=low, 1=normal, 2=high (interrupts)


@dataclass
class MoveMessage(BaseMessage):
    """Movement command"""
    type: str = MessageType.MOVE.value
    direction: str = "stop"  # forward, backward, left, right, stop
    distance: float = 0.0  # meters
    speed: str = "medium"  # slow, medium, fast
    duration: Optional[float] = None  # seconds (alternative to distance)


@dataclass
class GimbalMessage(BaseMessage):
    """Gimbal/camera control"""
    type: str = MessageType.GIMBAL.value
    pan: float = 0  # -180 to 180
    tilt: float = 0  # -30 to 90
    speed: int = 200
    action: str = "move"  # move, nod, shake, reset


@dataclass
class LightsMessage(BaseMessage):
    """LED control"""
    type: str = MessageType.LIGHTS.value
    front: int = 0  # 0-255
    back: int = 0   # 0-255
    mode: str = "solid"  # solid, breath, off


@dataclass
class DisplayMessage(BaseMessage):
    """OLED display control"""
    type: str = MessageType.DISPLAY.value
    lines: List[str] = field(default_factory=list)  # Up to 4 lines


@dataclass
class ErrorMessage(BaseMessage):
    """Error notification"""
    type: str = MessageType.ERROR.value
    error: str = ""
    code: Optional[int] = None


def parse_message(data: str) -> BaseMessage:
    """Parse a JSON message into the appropriate message type"""
    parsed = json.loads(data)
    msg_type = parsed.get('type', '')
    
    type_map = {
        MessageType.AUDIO_DATA.value: AudioMessage,
        MessageType.IMAGE_DATA.value: ImageMessage,
        MessageType.SENSOR_DATA.value: SensorMessage,
        MessageType.TEXT_QUERY.value: TextQueryMessage,
        MessageType.SPEAK.value: SpeakMessage,
        MessageType.MOVE.value: MoveMessage,
        MessageType.GIMBAL.value: GimbalMessage,
        MessageType.LIGHTS.value: LightsMessage,
        MessageType.DISPLAY.value: DisplayMessage,
        MessageType.ERROR.value: ErrorMessage,
    }
    
    msg_class = type_map.get(msg_type, BaseMessage)
    
    # Filter out unknown fields
    valid_fields = {f.name for f in msg_class.__dataclass_fields__.values()} if hasattr(msg_class, '__dataclass_fields__') else set()
    filtered = {k: v for k, v in parsed.items() if k in valid_fields or not valid_fields}
    
    return msg_class(**filtered) if valid_fields else BaseMessage(**parsed)

