/**
 * Cloud AI API - Connects to PC server for AI processing
 * 
 * This is separate from robot-api.ts which connects to the Pi.
 * 
 * Architecture:
 *   Phone App → Pi (robot-api.ts): Camera, rover control, lights, WiFi
 *   Phone App → PC (cloud-api.ts): Chat, vision, STT, TTS
 */

import axios, { AxiosInstance } from "axios";

// Default Cloud PC IP (Tailscale) - FastAPI runs on port 8000
export const DEFAULT_CLOUD_URL = "http://100.121.110.125:8000";

export interface ChatRequest {
  message: string;
  max_tokens?: number;
  temperature?: number;
}

export interface ChatResponse {
  response: string;
  movement?: {
    direction: string;
    distance: number;
    speed: string;
  };
  memory?: {
    facts: string[];
    count: number;
  };
}

export interface VisionRequest {
  question: string;
  image_base64: string;
  max_tokens?: number;
}

export interface VisionResponse {
  response: string;
  movement?: {
    direction: string;
    distance: number;
    speed: string;
  };
}

export interface STTResponse {
  text: string | null;
  success: boolean;
}

export interface CloudHealth {
  ok: boolean;
  assistant_loaded: boolean;
  speech_loaded: boolean;
}

export interface CloudApiOptions {
  baseUrl?: string;
  timeout?: number;
}

export class CloudAPI {
  private baseUrl: string;
  private axiosInstance: AxiosInstance;
  private timeout: number;

  constructor(options: CloudApiOptions = {}) {
    this.baseUrl = (options.baseUrl || DEFAULT_CLOUD_URL).replace(/\/$/, "");
    this.timeout = options.timeout ?? 30000; // Longer timeout for AI processing

    this.axiosInstance = axios.create({
      baseURL: this.baseUrl,
      timeout: this.timeout,
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
    });
  }

  public updateBaseUrl(baseUrl: string) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.axiosInstance.defaults.baseURL = this.baseUrl;
  }

  public getBaseUrl(): string {
    return this.baseUrl;
  }

  /**
   * Check cloud server health
   */
  public async health(): Promise<CloudHealth> {
    const response = await this.axiosInstance.get<CloudHealth>("/health");
    return response.data;
  }

  /**
   * Chat with AI (text only)
   */
  public async chat(request: ChatRequest): Promise<ChatResponse> {
    const response = await this.axiosInstance.post<ChatResponse>("/chat", {
      message: request.message,
      max_tokens: request.max_tokens ?? 150,
      temperature: request.temperature ?? 0.7,
    });
    return response.data;
  }

  /**
   * Ask AI about an image
   */
  public async vision(request: VisionRequest): Promise<VisionResponse> {
    const response = await this.axiosInstance.post<VisionResponse>("/vision", {
      question: request.question,
      image_base64: request.image_base64,
      max_tokens: request.max_tokens ?? 200,
    });
    return response.data;
  }

  /**
   * Convert speech to text
   * @param audioBlob Audio data as Blob or File
   */
  public async speechToText(audioBlob: Blob): Promise<STTResponse> {
    const formData = new FormData();
    formData.append("audio", audioBlob, "audio.wav");

    const response = await this.axiosInstance.post<STTResponse>("/stt", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    return response.data;
  }

  /**
   * Convert text to speech
   * @returns Audio data as ArrayBuffer (WAV format)
   */
  public async textToSpeech(text: string): Promise<ArrayBuffer> {
    const response = await this.axiosInstance.post("/tts", { text }, {
      responseType: "arraybuffer",
    });
    return response.data;
  }

  /**
   * Get WebSocket URL for real-time voice interaction
   */
  public getVoiceWebSocketUrl(): string {
    const wsBase = this.baseUrl.replace(/^http/, "ws");
    return `${wsBase}/voice`;
  }

  /**
   * Get all stored memory (facts and preferences)
   */
  public async getMemory(): Promise<{
    facts: Array<{ fact: string; added_at: string; source?: string }>;
    preferences: Record<string, { value: any; updated_at: string }>;
    conversation_count: number;
    last_updated: string | null;
  }> {
    const response = await this.axiosInstance.get("/memory");
    return response.data;
  }

  /**
   * Clear all stored memory (use with caution)
   */
  public async clearMemory(): Promise<{ status: string; message: string }> {
    const response = await this.axiosInstance.delete("/memory");
    return response.data;
  }

  // ============================================================================
  // Assistant Management (Timers, Reminders, Meetings, Notes, Tasks)
  // ============================================================================

  /**
   * Timer Management
   */
  public async createTimer(durationSeconds: number, name?: string): Promise<any> {
    const response = await this.axiosInstance.post("/assistant/timer", {
      duration_seconds: durationSeconds,
      name,
    });
    return response.data;
  }

  public async getTimers(status?: string): Promise<{ timers: any[]; count: number }> {
    const params = status ? { status } : {};
    const response = await this.axiosInstance.get("/assistant/timers", { params });
    return response.data;
  }

  public async startTimer(timerId: string): Promise<any> {
    const response = await this.axiosInstance.post(`/assistant/timer/${timerId}/start`);
    return response.data;
  }

  public async cancelTimer(timerId: string): Promise<{ status: string; message: string }> {
    const response = await this.axiosInstance.delete(`/assistant/timer/${timerId}`);
    return response.data;
  }

  /**
   * Reminder Management
   */
  public async createReminder(
    message: string,
    reminderTime: string,
    name?: string
  ): Promise<any> {
    const response = await this.axiosInstance.post("/assistant/reminder", {
      message,
      reminder_time: reminderTime,
      name,
    });
    return response.data;
  }

  public async getReminders(
    status?: string,
    dueOnly?: boolean
  ): Promise<{ reminders: any[]; count: number }> {
    const params: any = {};
    if (status) params.status = status;
    if (dueOnly) params.due_only = dueOnly;
    const response = await this.axiosInstance.get("/assistant/reminders", { params });
    return response.data;
  }

  public async completeReminder(reminderId: string): Promise<{ status: string; message: string }> {
    const response = await this.axiosInstance.post(`/assistant/reminder/${reminderId}/complete`);
    return response.data;
  }

  /**
   * Meeting Management
   */
  public async createMeeting(
    title: string,
    startTime: string,
    durationMinutes?: number,
    participants?: string[],
    notes?: string
  ): Promise<any> {
    const response = await this.axiosInstance.post("/assistant/meeting", {
      title,
      start_time: startTime,
      duration_minutes: durationMinutes || 60,
      participants: participants || [],
      notes,
    });
    return response.data;
  }

  public async getMeetings(
    upcomingOnly?: boolean,
    includeCompleted?: boolean
  ): Promise<{ meetings: any[]; count: number }> {
    const params: any = {};
    if (upcomingOnly) params.upcoming_only = upcomingOnly;
    if (includeCompleted !== undefined) params.include_completed = includeCompleted;
    const response = await this.axiosInstance.get("/assistant/meetings", { params });
    return response.data;
  }

  public async summarizeMeeting(meetingId: string, summary: string): Promise<any> {
    const response = await this.axiosInstance.post(`/assistant/meeting/${meetingId}/summarize`, {
      summary,
    });
    return response.data;
  }

  /**
   * Notes Management
   */
  public async createNote(
    title: string,
    content?: string,
    tags?: string[]
  ): Promise<any> {
    const response = await this.axiosInstance.post("/assistant/note", {
      title,
      content: content || "",
      tags: tags || [],
    });
    return response.data;
  }

  public async getNotes(tag?: string): Promise<{ notes: any[]; count: number }> {
    const params = tag ? { tag } : {};
    const response = await this.axiosInstance.get("/assistant/notes", { params });
    return response.data;
  }

  public async updateNote(
    noteId: string,
    title?: string,
    content?: string
  ): Promise<any> {
    const response = await this.axiosInstance.put(`/assistant/note/${noteId}`, {
      title,
      content,
    });
    return response.data;
  }

  /**
   * Tasks Management
   */
  public async createTask(
    title: string,
    description?: string,
    dueDate?: string
  ): Promise<any> {
    const response = await this.axiosInstance.post("/assistant/task", {
      title,
      description,
      due_date: dueDate,
    });
    return response.data;
  }

  public async getTasks(includeCompleted?: boolean): Promise<{ tasks: any[]; count: number }> {
    const params = includeCompleted !== undefined ? { include_completed: includeCompleted } : {};
    const response = await this.axiosInstance.get("/assistant/tasks", { params });
    return response.data;
  }

  public async completeTask(taskId: string): Promise<{ status: string; message: string }> {
    const response = await this.axiosInstance.post(`/assistant/task/${taskId}/complete`);
    return response.data;
  }

  /**
   * Get assistant summary
   */
  public async getAssistantSummary(): Promise<any> {
    const response = await this.axiosInstance.get("/assistant/summary");
    return response.data;
  }
}

/**
 * Create a CloudAPI instance
 */
export const createCloudApi = (baseUrl?: string, timeout?: number) =>
  new CloudAPI({ baseUrl, timeout });

/**
 * Default cloud API instance (uses Tailscale IP)
 */
export const cloudApi = new CloudAPI();
