"""
Assistant Management System
Handles timers, reminders, meetings, notes, tasks, and other assistant functions.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class TimerStatus(str, Enum):
    """Timer status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ReminderStatus(str, Enum):
    """Reminder status."""
    PENDING = "pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class AssistantManager:
    """Manages timers, reminders, meetings, notes, tasks, and other assistant functions."""
    
    def __init__(self, data_file: Optional[str] = None):
        """
        Initialize assistant manager.
        
        Args:
            data_file: Path to JSON file for storing data. Defaults to assistant_data.json in cloud directory.
        """
        if data_file is None:
            project_root = Path(__file__).parent.parent
            data_file = str(project_root / "assistant_data.json")
        
        self.data_file = Path(data_file)
        self.data: Dict[str, Any] = {
            "timers": [],
            "reminders": [],
            "meetings": [],
            "notes": [],
            "tasks": [],
            "last_updated": None
        }
        self._load_data()
    
    def _load_data(self):
        """Load data from JSON file."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                logger.info(f"Loaded assistant data from {self.data_file}")
            except Exception as e:
                logger.error(f"Failed to load assistant data: {e}")
                self.data = {
                    "timers": [],
                    "reminders": [],
                    "meetings": [],
                    "notes": [],
                    "tasks": [],
                    "last_updated": None
                }
        else:
            logger.info(f"Assistant data file not found, starting fresh: {self.data_file}")
            self._save_data()
    
    def _save_data(self):
        """Save data to JSON file."""
        try:
            self.data["last_updated"] = datetime.now().isoformat()
            self.data_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved assistant data to {self.data_file}")
        except Exception as e:
            logger.error(f"Failed to save assistant data: {e}")
    
    # ============================================================================
    # Timer Management
    # ============================================================================
    
    def create_timer(self, duration_seconds: int, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new timer.
        
        Args:
            duration_seconds: Timer duration in seconds
            name: Optional timer name
        
        Returns:
            Timer dictionary
        """
        timer = {
            "id": f"timer_{len(self.data['timers']) + 1}_{int(datetime.now().timestamp())}",
            "name": name or f"Timer {len(self.data['timers']) + 1}",
            "duration_seconds": duration_seconds,
            "remaining_seconds": duration_seconds,
            "status": TimerStatus.PENDING,
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None
        }
        self.data["timers"].append(timer)
        self._save_data()
        logger.info(f"Created timer: {timer['name']} ({duration_seconds}s)")
        return timer
    
    def start_timer(self, timer_id: str) -> Optional[Dict[str, Any]]:
        """Start a timer."""
        timer = self._find_timer(timer_id)
        if not timer:
            return None
        
        if timer["status"] != TimerStatus.PENDING:
            logger.warning(f"Timer {timer_id} cannot be started (status: {timer['status']})")
            return None
        
        timer["status"] = TimerStatus.RUNNING
        timer["started_at"] = datetime.now().isoformat()
        self._save_data()
        logger.info(f"Started timer: {timer['name']}")
        return timer
    
    def cancel_timer(self, timer_id: str) -> Optional[Dict[str, Any]]:
        """Cancel a timer."""
        timer = self._find_timer(timer_id)
        if not timer:
            return None
        
        timer["status"] = TimerStatus.CANCELLED
        self._save_data()
        logger.info(f"Cancelled timer: {timer['name']}")
        return timer
    
    def get_timers(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all timers, optionally filtered by status."""
        timers = self.data["timers"]
        if status:
            timers = [t for t in timers if t["status"] == status]
        return timers
    
    def _find_timer(self, timer_id: str) -> Optional[Dict[str, Any]]:
        """Find a timer by ID."""
        for timer in self.data["timers"]:
            if timer["id"] == timer_id:
                return timer
        return None
    
    # ============================================================================
    # Reminder Management
    # ============================================================================
    
    def create_reminder(
        self,
        message: str,
        reminder_time: str,  # ISO format datetime
        name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new reminder.
        
        Args:
            message: Reminder message
            reminder_time: When to remind (ISO format datetime string)
            name: Optional reminder name
        
        Returns:
            Reminder dictionary
        """
        reminder = {
            "id": f"reminder_{len(self.data['reminders']) + 1}_{int(datetime.now().timestamp())}",
            "name": name or message[:50],
            "message": message,
            "reminder_time": reminder_time,
            "status": ReminderStatus.PENDING,
            "created_at": datetime.now().isoformat(),
            "completed_at": None
        }
        self.data["reminders"].append(reminder)
        self._save_data()
        logger.info(f"Created reminder: {reminder['name']} at {reminder_time}")
        return reminder
    
    def get_due_reminders(self) -> List[Dict[str, Any]]:
        """Get reminders that are due now or overdue."""
        now = datetime.now()
        due = []
        for reminder in self.data["reminders"]:
            if reminder["status"] == ReminderStatus.PENDING:
                try:
                    reminder_dt = datetime.fromisoformat(reminder["reminder_time"])
                    if reminder_dt <= now:
                        due.append(reminder)
                except Exception as e:
                    logger.warning(f"Invalid reminder time format: {e}")
        return due
    
    def complete_reminder(self, reminder_id: str) -> Optional[Dict[str, Any]]:
        """Mark a reminder as completed."""
        reminder = self._find_reminder(reminder_id)
        if not reminder:
            return None
        
        reminder["status"] = ReminderStatus.COMPLETED
        reminder["completed_at"] = datetime.now().isoformat()
        self._save_data()
        logger.info(f"Completed reminder: {reminder['name']}")
        return reminder
    
    def get_reminders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all reminders, optionally filtered by status."""
        reminders = self.data["reminders"]
        if status:
            reminders = [r for r in reminders if r["status"] == status]
        return reminders
    
    def _find_reminder(self, reminder_id: str) -> Optional[Dict[str, Any]]:
        """Find a reminder by ID."""
        for reminder in self.data["reminders"]:
            if reminder["id"] == reminder_id:
                return reminder
        return None
    
    # ============================================================================
    # Meeting Management
    # ============================================================================
    
    def create_meeting(
        self,
        title: str,
        start_time: str,  # ISO format datetime
        duration_minutes: int = 60,
        participants: Optional[List[str]] = None,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new meeting.
        
        Args:
            title: Meeting title
            start_time: Meeting start time (ISO format datetime string)
            duration_minutes: Meeting duration in minutes
            participants: List of participant names
            notes: Optional meeting notes
        
        Returns:
            Meeting dictionary
        """
        meeting = {
            "id": f"meeting_{len(self.data['meetings']) + 1}_{int(datetime.now().timestamp())}",
            "title": title,
            "start_time": start_time,
            "duration_minutes": duration_minutes,
            "participants": participants or [],
            "notes": notes,
            "summary": None,
            "created_at": datetime.now().isoformat(),
            "completed": False
        }
        self.data["meetings"].append(meeting)
        self._save_data()
        logger.info(f"Created meeting: {title} at {start_time}")
        return meeting
    
    def get_upcoming_meetings(self, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """Get meetings scheduled within the next N hours."""
        now = datetime.now()
        cutoff = now + timedelta(hours=hours_ahead)
        upcoming = []
        
        for meeting in self.data["meetings"]:
            if meeting.get("completed", False):
                continue
            try:
                start_dt = datetime.fromisoformat(meeting["start_time"])
                if now <= start_dt <= cutoff:
                    upcoming.append(meeting)
            except Exception as e:
                logger.warning(f"Invalid meeting time format: {e}")
        
        # Sort by start time
        upcoming.sort(key=lambda m: m["start_time"])
        return upcoming
    
    def summarize_meeting(self, meeting_id: str, summary: str) -> Optional[Dict[str, Any]]:
        """Add a summary to a meeting."""
        meeting = self._find_meeting(meeting_id)
        if not meeting:
            return None
        
        meeting["summary"] = summary
        meeting["completed"] = True
        self._save_data()
        logger.info(f"Summarized meeting: {meeting['title']}")
        return meeting
    
    def get_meetings(self, include_completed: bool = True) -> List[Dict[str, Any]]:
        """Get all meetings."""
        meetings = self.data["meetings"]
        if not include_completed:
            meetings = [m for m in meetings if not m.get("completed", False)]
        return meetings
    
    def _find_meeting(self, meeting_id: str) -> Optional[Dict[str, Any]]:
        """Find a meeting by ID."""
        for meeting in self.data["meetings"]:
            if meeting["id"] == meeting_id:
                return meeting
        return None
    
    # ============================================================================
    # Notes Management
    # ============================================================================
    
    def create_note(self, title: str, content: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create a new note.
        
        Args:
            title: Note title
            content: Note content
            tags: Optional tags
        
        Returns:
            Note dictionary
        """
        note = {
            "id": f"note_{len(self.data['notes']) + 1}_{int(datetime.now().timestamp())}",
            "title": title,
            "content": content,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        self.data["notes"].append(note)
        self._save_data()
        logger.info(f"Created note: {title}")
        return note
    
    def get_notes(self, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all notes, optionally filtered by tag."""
        notes = self.data["notes"]
        if tag:
            notes = [n for n in notes if tag in n.get("tags", [])]
        return notes
    
    def update_note(self, note_id: str, title: Optional[str] = None, content: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Update a note."""
        note = self._find_note(note_id)
        if not note:
            return None
        
        if title:
            note["title"] = title
        if content:
            note["content"] = content
        note["updated_at"] = datetime.now().isoformat()
        self._save_data()
        logger.info(f"Updated note: {note['title']}")
        return note
    
    def _find_note(self, note_id: str) -> Optional[Dict[str, Any]]:
        """Find a note by ID."""
        for note in self.data["notes"]:
            if note["id"] == note_id:
                return note
        return None
    
    # ============================================================================
    # Tasks Management
    # ============================================================================
    
    def create_task(self, title: str, description: Optional[str] = None, due_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new task.
        
        Args:
            title: Task title
            description: Optional task description
            due_date: Optional due date (ISO format datetime string)
        
        Returns:
            Task dictionary
        """
        task = {
            "id": f"task_{len(self.data['tasks']) + 1}_{int(datetime.now().timestamp())}",
            "title": title,
            "description": description,
            "due_date": due_date,
            "completed": False,
            "created_at": datetime.now().isoformat(),
            "completed_at": None
        }
        self.data["tasks"].append(task)
        self._save_data()
        logger.info(f"Created task: {title}")
        return task
    
    def complete_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Mark a task as completed."""
        task = self._find_task(task_id)
        if not task:
            return None
        
        task["completed"] = True
        task["completed_at"] = datetime.now().isoformat()
        self._save_data()
        logger.info(f"Completed task: {task['title']}")
        return task
    
    def get_tasks(self, include_completed: bool = True) -> List[Dict[str, Any]]:
        """Get all tasks."""
        tasks = self.data["tasks"]
        if not include_completed:
            tasks = [t for t in tasks if not t.get("completed", False)]
        return tasks
    
    def _find_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Find a task by ID."""
        for task in self.data["tasks"]:
            if task["id"] == task_id:
                return task
        return None
    
    # ============================================================================
    # Utility Methods
    # ============================================================================
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all assistant data."""
        return {
            "timers": {
                "total": len(self.data["timers"]),
                "running": len([t for t in self.data["timers"] if t["status"] == TimerStatus.RUNNING]),
                "pending": len([t for t in self.data["timers"] if t["status"] == TimerStatus.PENDING])
            },
            "reminders": {
                "total": len(self.data["reminders"]),
                "pending": len([r for r in self.data["reminders"] if r["status"] == ReminderStatus.PENDING]),
                "due": len(self.get_due_reminders())
            },
            "meetings": {
                "total": len(self.data["meetings"]),
                "upcoming": len(self.get_upcoming_meetings())
            },
            "notes": {
                "total": len(self.data["notes"])
            },
            "tasks": {
                "total": len(self.data["tasks"]),
                "completed": len([t for t in self.data["tasks"] if t.get("completed", False)]),
                "pending": len([t for t in self.data["tasks"] if not t.get("completed", False)])
            }
        }

