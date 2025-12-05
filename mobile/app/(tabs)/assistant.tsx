import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  ScrollView,
  StyleSheet,
  RefreshControl,
  Pressable,
  Alert,
  TextInput,
  Modal,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { IconSymbol } from '@/components/ui/icon-symbol';
import { cloudApi, DEFAULT_CLOUD_URL } from '@/services/cloud-api';
import { Colors } from '@/constants/theme';
import { CLOUD_API_BASE_URL } from '@/constants/env';

interface Timer {
  id: string;
  name: string;
  duration_seconds: number;
  remaining_seconds: number;
  status: string;
  created_at: string;
}

interface Reminder {
  id: string;
  name: string;
  message: string;
  reminder_time: string;
  status: string;
}

interface Meeting {
  id: string;
  title: string;
  start_time: string;
  duration_minutes: number;
  participants: string[];
  completed: boolean;
}

interface Note {
  id: string;
  title: string;
  content: string;
  tags: string[];
  created_at: string;
}

interface Task {
  id: string;
  title: string;
  description?: string;
  due_date?: string;
  completed: boolean;
}

export default function AssistantScreen() {
  const [refreshing, setRefreshing] = useState(false);
  const [activeTab, setActiveTab] = useState<'timers' | 'reminders' | 'meetings' | 'notes' | 'tasks'>('timers');
  
  const [timers, setTimers] = useState<Timer[]>([]);
  const [reminders, setReminders] = useState<Reminder[]>([]);
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [notes, setNotes] = useState<Note[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  
  const [summary, setSummary] = useState<any>(null);
  const [modalVisible, setModalVisible] = useState(false);
  const [modalType, setModalType] = useState<'timer' | 'reminder' | 'task' | 'note' | null>(null);
  const [formData, setFormData] = useState<any>({});

  const loadData = useCallback(async () => {
    try {
      // Use environment variable or default
      const baseUrl = CLOUD_API_BASE_URL || DEFAULT_CLOUD_URL;
      if (baseUrl) {
        cloudApi.updateBaseUrl(baseUrl);
      }
      
      // Load all data in parallel
      const [timersRes, remindersRes, meetingsRes, notesRes, tasksRes, summaryRes] = await Promise.all([
        cloudApi.getTimers().catch(() => ({ timers: [], count: 0 })),
        cloudApi.getReminders().catch(() => ({ reminders: [], count: 0 })),
        cloudApi.getMeetings(false, true).catch(() => ({ meetings: [], count: 0 })),
        cloudApi.getNotes().catch(() => ({ notes: [], count: 0 })),
        cloudApi.getTasks(false).catch(() => ({ tasks: [], count: 0 })),
        cloudApi.getAssistantSummary().catch(() => null),
      ]);

      setTimers(timersRes.timers || []);
      setReminders(remindersRes.reminders || []);
      setMeetings(meetingsRes.meetings || []);
      setNotes(notesRes.notes || []);
      setTasks(tasksRes.tasks || []);
      setSummary(summaryRes);
    } catch (error) {
      console.error('Failed to load assistant data:', error);
    }
  }, []);

  useEffect(() => {
    loadData();
    // Refresh every 30 seconds
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, [loadData]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  }, [loadData]);

  const formatTime = (isoString: string) => {
    const date = new Date(isoString);
    return date.toLocaleString();
  };

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    if (hours > 0) return `${hours}h ${minutes}m`;
    if (minutes > 0) return `${minutes}m ${secs}s`;
    return `${secs}s`;
  };

  const handleCreateTimer = async () => {
    if (!formData.duration) {
      Alert.alert('Error', 'Please enter a duration');
      return;
    }
    
    try {
      const durationSeconds = parseInt(formData.duration) * 60; // Convert minutes to seconds
      await cloudApi.createTimer(durationSeconds, formData.name);
      setModalVisible(false);
      setFormData({});
      loadData();
    } catch (error) {
      Alert.alert('Error', 'Failed to create timer');
    }
  };

  const handleCreateReminder = async () => {
    if (!formData.message || !formData.time) {
      Alert.alert('Error', 'Please enter message and time');
      return;
    }
    
    try {
      // Simple time parsing - could be improved
      const reminderTime = new Date(formData.time).toISOString();
      await cloudApi.createReminder(formData.message, reminderTime, formData.name);
      setModalVisible(false);
      setFormData({});
      loadData();
    } catch (error) {
      Alert.alert('Error', 'Failed to create reminder');
    }
  };

  const handleCreateTask = async () => {
    if (!formData.title) {
      Alert.alert('Error', 'Please enter a task title');
      return;
    }
    
    try {
      await cloudApi.createTask(formData.title, formData.description, formData.dueDate);
      setModalVisible(false);
      setFormData({});
      loadData();
    } catch (error) {
      Alert.alert('Error', 'Failed to create task');
    }
  };

  const handleCreateNote = async () => {
    if (!formData.title) {
      Alert.alert('Error', 'Please enter a note title');
      return;
    }
    
    try {
      await cloudApi.createNote(formData.title, formData.content || '', formData.tags?.split(',') || []);
      setModalVisible(false);
      setFormData({});
      loadData();
    } catch (error) {
      Alert.alert('Error', 'Failed to create note');
    }
  };

  const handleCompleteTask = async (taskId: string) => {
    try {
      await cloudApi.completeTask(taskId);
      loadData();
    } catch (error) {
      Alert.alert('Error', 'Failed to complete task');
    }
  };

  const handleCancelTimer = async (timerId: string) => {
    try {
      await cloudApi.cancelTimer(timerId);
      loadData();
    } catch (error) {
      Alert.alert('Error', 'Failed to cancel timer');
    }
  };

  const handleCompleteReminder = async (reminderId: string) => {
    try {
      await cloudApi.completeReminder(reminderId);
      loadData();
    } catch (error) {
      Alert.alert('Error', 'Failed to complete reminder');
    }
  };

  const tabs = [
    { id: 'timers' as const, label: 'Timers', icon: 'timer', count: timers.filter(t => t.status === 'running' || t.status === 'pending').length },
    { id: 'reminders' as const, label: 'Reminders', icon: 'bell', count: reminders.filter(r => r.status === 'pending').length },
    { id: 'meetings' as const, label: 'Meetings', icon: 'calendar', count: meetings.filter(m => !m.completed).length },
    { id: 'notes' as const, label: 'Notes', icon: 'note.text', count: notes.length },
    { id: 'tasks' as const, label: 'Tasks', icon: 'checkmark.circle', count: tasks.filter(t => !t.completed).length },
  ];

  const renderContent = () => {
    switch (activeTab) {
      case 'timers':
        return (
          <View style={styles.list}>
            {timers.length === 0 ? (
              <ThemedText style={styles.emptyText}>No timers</ThemedText>
            ) : (
              timers.map((timer) => (
                <ThemedView key={timer.id} style={styles.item}>
                  <View style={styles.itemHeader}>
                    <ThemedText style={styles.itemTitle}>{timer.name}</ThemedText>
                    <ThemedText style={styles.itemStatus}>{timer.status}</ThemedText>
                  </View>
                  <ThemedText style={styles.itemSubtext}>
                    Duration: {formatDuration(timer.duration_seconds)}
                  </ThemedText>
                  {timer.status === 'running' && (
                    <ThemedText style={styles.itemSubtext}>
                      Remaining: {formatDuration(timer.remaining_seconds)}
                    </ThemedText>
                  )}
                  {timer.status === 'pending' && (
                    <Pressable onPress={() => handleCancelTimer(timer.id)} style={styles.cancelButton}>
                      <ThemedText style={styles.cancelButtonText}>Cancel</ThemedText>
                    </Pressable>
                  )}
                </ThemedView>
              ))
            )}
          </View>
        );
      
      case 'reminders':
        return (
          <View style={styles.list}>
            {reminders.length === 0 ? (
              <ThemedText style={styles.emptyText}>No reminders</ThemedText>
            ) : (
              reminders.map((reminder) => (
                <ThemedView key={reminder.id} style={styles.item}>
                  <View style={styles.itemHeader}>
                    <ThemedText style={styles.itemTitle}>{reminder.name}</ThemedText>
                    {reminder.status === 'pending' && (
                      <Pressable onPress={() => handleCompleteReminder(reminder.id)} style={styles.completeButton}>
                        <ThemedText style={styles.completeButtonText}>Complete</ThemedText>
                      </Pressable>
                    )}
                  </View>
                  <ThemedText style={styles.itemSubtext}>{reminder.message}</ThemedText>
                  <ThemedText style={styles.itemSubtext}>
                    Due: {formatTime(reminder.reminder_time)}
                  </ThemedText>
                </ThemedView>
              ))
            )}
          </View>
        );
      
      case 'meetings':
        return (
          <View style={styles.list}>
            {meetings.length === 0 ? (
              <ThemedText style={styles.emptyText}>No meetings</ThemedText>
            ) : (
              meetings.map((meeting) => (
                <ThemedView key={meeting.id} style={styles.item}>
                  <ThemedText style={styles.itemTitle}>{meeting.title}</ThemedText>
                  <ThemedText style={styles.itemSubtext}>
                    {formatTime(meeting.start_time)} ({meeting.duration_minutes} min)
                  </ThemedText>
                  {meeting.participants.length > 0 && (
                    <ThemedText style={styles.itemSubtext}>
                      Participants: {meeting.participants.join(', ')}
                    </ThemedText>
                  )}
                </ThemedView>
              ))
            )}
          </View>
        );
      
      case 'notes':
        return (
          <View style={styles.list}>
            {notes.length === 0 ? (
              <ThemedText style={styles.emptyText}>No notes</ThemedText>
            ) : (
              notes.map((note) => (
                <ThemedView key={note.id} style={styles.item}>
                  <ThemedText style={styles.itemTitle}>{note.title}</ThemedText>
                  <ThemedText style={styles.itemSubtext}>{note.content}</ThemedText>
                  {note.tags.length > 0 && (
                    <ThemedText style={styles.itemSubtext}>
                      Tags: {note.tags.join(', ')}
                    </ThemedText>
                  )}
                </ThemedView>
              ))
            )}
          </View>
        );
      
      case 'tasks':
        return (
          <View style={styles.list}>
            {tasks.length === 0 ? (
              <ThemedText style={styles.emptyText}>No tasks</ThemedText>
            ) : (
              tasks.map((task) => (
                <ThemedView key={task.id} style={[styles.item, task.completed && styles.completedItem]}>
                  <View style={styles.itemHeader}>
                    <ThemedText style={[styles.itemTitle, task.completed && styles.completedText]}>
                      {task.title}
                    </ThemedText>
                    {!task.completed && (
                      <Pressable onPress={() => handleCompleteTask(task.id)} style={styles.completeButton}>
                        <ThemedText style={styles.completeButtonText}>Complete</ThemedText>
                      </Pressable>
                    )}
                  </View>
                  {task.description && (
                    <ThemedText style={styles.itemSubtext}>{task.description}</ThemedText>
                  )}
                  {task.due_date && (
                    <ThemedText style={styles.itemSubtext}>
                      Due: {formatTime(task.due_date)}
                    </ThemedText>
                  )}
                </ThemedView>
              ))
            )}
          </View>
        );
    }
  };

  const renderModal = () => {
    if (!modalType) return null;

    return (
      <Modal
        visible={modalVisible}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <ThemedView style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <ThemedText style={styles.modalTitle}>
                Create {modalType.charAt(0).toUpperCase() + modalType.slice(1)}
              </ThemedText>
              <Pressable onPress={() => setModalVisible(false)}>
                <IconSymbol name="xmark.circle.fill" size={24} color="#fff" />
              </Pressable>
            </View>

            {modalType === 'timer' && (
              <>
                <TextInput
                  style={styles.input}
                  placeholder="Timer name (optional)"
                  placeholderTextColor="#666"
                  value={formData.name}
                  onChangeText={(text) => setFormData({ ...formData, name: text })}
                />
                <TextInput
                  style={styles.input}
                  placeholder="Duration (minutes)"
                  placeholderTextColor="#666"
                  keyboardType="numeric"
                  value={formData.duration}
                  onChangeText={(text) => setFormData({ ...formData, duration: text })}
                />
                <Pressable style={styles.createButton} onPress={handleCreateTimer}>
                  <ThemedText style={styles.createButtonText}>Create Timer</ThemedText>
                </Pressable>
              </>
            )}

            {modalType === 'reminder' && (
              <>
                <TextInput
                  style={styles.input}
                  placeholder="Reminder name (optional)"
                  placeholderTextColor="#666"
                  value={formData.name}
                  onChangeText={(text) => setFormData({ ...formData, name: text })}
                />
                <TextInput
                  style={styles.input}
                  placeholder="Reminder message"
                  placeholderTextColor="#666"
                  value={formData.message}
                  onChangeText={(text) => setFormData({ ...formData, message: text })}
                />
                <TextInput
                  style={styles.input}
                  placeholder="Time (ISO format: 2024-01-01T12:00:00)"
                  placeholderTextColor="#666"
                  value={formData.time}
                  onChangeText={(text) => setFormData({ ...formData, time: text })}
                />
                <Pressable style={styles.createButton} onPress={handleCreateReminder}>
                  <ThemedText style={styles.createButtonText}>Create Reminder</ThemedText>
                </Pressable>
              </>
            )}

            {modalType === 'task' && (
              <>
                <TextInput
                  style={styles.input}
                  placeholder="Task title"
                  placeholderTextColor="#666"
                  value={formData.title}
                  onChangeText={(text) => setFormData({ ...formData, title: text })}
                />
                <TextInput
                  style={[styles.input, styles.textArea]}
                  placeholder="Description (optional)"
                  placeholderTextColor="#666"
                  multiline
                  value={formData.description}
                  onChangeText={(text) => setFormData({ ...formData, description: text })}
                />
                <TextInput
                  style={styles.input}
                  placeholder="Due date (ISO format, optional)"
                  placeholderTextColor="#666"
                  value={formData.dueDate}
                  onChangeText={(text) => setFormData({ ...formData, dueDate: text })}
                />
                <Pressable style={styles.createButton} onPress={handleCreateTask}>
                  <ThemedText style={styles.createButtonText}>Create Task</ThemedText>
                </Pressable>
              </>
            )}

            {modalType === 'note' && (
              <>
                <TextInput
                  style={styles.input}
                  placeholder="Note title"
                  placeholderTextColor="#666"
                  value={formData.title}
                  onChangeText={(text) => setFormData({ ...formData, title: text })}
                />
                <TextInput
                  style={[styles.input, styles.textArea]}
                  placeholder="Note content"
                  placeholderTextColor="#666"
                  multiline
                  value={formData.content}
                  onChangeText={(text) => setFormData({ ...formData, content: text })}
                />
                <TextInput
                  style={styles.input}
                  placeholder="Tags (comma-separated, optional)"
                  placeholderTextColor="#666"
                  value={formData.tags}
                  onChangeText={(text) => setFormData({ ...formData, tags: text })}
                />
                <Pressable style={styles.createButton} onPress={handleCreateNote}>
                  <ThemedText style={styles.createButtonText}>Create Note</ThemedText>
                </Pressable>
              </>
            )}
          </ThemedView>
        </View>
      </Modal>
    );
  };

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <ThemedView style={styles.header}>
        <ThemedText style={styles.title}>Assistant</ThemedText>
        {summary && (
          <ThemedText style={styles.subtitle}>
            {summary.timers?.running || 0} active • {summary.reminders?.due || 0} due • {summary.tasks?.pending || 0} tasks
          </ThemedText>
        )}
      </ThemedView>

      <View style={styles.tabs}>
        {tabs.map((tab) => (
          <Pressable
            key={tab.id}
            style={[styles.tab, activeTab === tab.id && styles.activeTab]}
            onPress={() => setActiveTab(tab.id)}
          >
            <IconSymbol
              name={tab.icon as any}
              size={20}
              color={activeTab === tab.id ? Colors.light.tint : '#666'}
            />
            <ThemedText style={[styles.tabLabel, activeTab === tab.id && styles.activeTabLabel]}>
              {tab.label}
            </ThemedText>
            {tab.count > 0 && (
              <View style={styles.badge}>
                <ThemedText style={styles.badgeText}>{tab.count}</ThemedText>
              </View>
            )}
          </Pressable>
        ))}
      </View>

      <ScrollView
        style={styles.content}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
      >
        {renderContent()}
      </ScrollView>

      <View style={styles.fabContainer}>
        <Pressable
          style={styles.fab}
          onPress={() => {
            const createOptions = ['Timer', 'Reminder', 'Task', 'Note'];
            Alert.alert('Create', 'What would you like to create?', [
              ...createOptions.map((option) => ({
                text: option,
                onPress: () => {
                  setModalType(option.toLowerCase() as any);
                  setModalVisible(true);
                },
              })),
              { text: 'Cancel', style: 'cancel' },
            ]);
          }}
        >
          <IconSymbol name="plus" size={24} color="#fff" />
        </Pressable>
      </View>

      {renderModal()}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  header: {
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#222',
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 14,
    opacity: 0.7,
  },
  tabs: {
    flexDirection: 'row',
    paddingHorizontal: 10,
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#222',
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 8,
    paddingHorizontal: 4,
    borderRadius: 8,
    marginHorizontal: 2,
  },
  activeTab: {
    backgroundColor: '#222',
  },
  tabLabel: {
    fontSize: 12,
    marginLeft: 4,
    opacity: 0.7,
  },
  activeTabLabel: {
    opacity: 1,
    fontWeight: '600',
  },
  badge: {
    backgroundColor: Colors.light.tint,
    borderRadius: 10,
    paddingHorizontal: 6,
    paddingVertical: 2,
    marginLeft: 4,
  },
  badgeText: {
    color: '#000',
    fontSize: 10,
    fontWeight: 'bold',
  },
  content: {
    flex: 1,
  },
  list: {
    padding: 16,
  },
  item: {
    padding: 16,
    marginBottom: 12,
    borderRadius: 12,
    backgroundColor: '#111',
    borderWidth: 1,
    borderColor: '#222',
  },
  completedItem: {
    opacity: 0.6,
  },
  itemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  itemTitle: {
    fontSize: 16,
    fontWeight: '600',
    flex: 1,
  },
  completedText: {
    textDecorationLine: 'line-through',
    opacity: 0.6,
  },
  itemStatus: {
    fontSize: 12,
    opacity: 0.7,
    textTransform: 'capitalize',
  },
  itemSubtext: {
    fontSize: 14,
    opacity: 0.7,
    marginTop: 4,
  },
  emptyText: {
    textAlign: 'center',
    opacity: 0.5,
    marginTop: 40,
    fontSize: 16,
  },
  completeButton: {
    backgroundColor: Colors.light.tint,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
  },
  completeButtonText: {
    color: '#000',
    fontSize: 12,
    fontWeight: '600',
  },
  cancelButton: {
    backgroundColor: '#ef4444',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
    marginTop: 8,
    alignSelf: 'flex-start',
  },
  cancelButtonText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  fabContainer: {
    position: 'absolute',
    bottom: 20,
    right: 20,
  },
  fab: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: Colors.light.tint,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    width: '90%',
    maxWidth: 400,
    backgroundColor: '#111',
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: '#222',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  input: {
    backgroundColor: '#222',
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
    color: '#fff',
    fontSize: 16,
  },
  textArea: {
    height: 100,
    textAlignVertical: 'top',
  },
  createButton: {
    backgroundColor: Colors.light.tint,
    padding: 14,
    borderRadius: 8,
    marginTop: 8,
  },
  createButtonText: {
    color: '#000',
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
  },
});

