# 🎵💃 Quick Setup: Music + Dance Integration

## TL;DR - Get Dancing with Music in 5 Minutes

### Step 1: Install Dependencies (1 minute)

```bash
cd /home/rovy/rovy_client

# Install YouTube Music API
pip install ytmusicapi

# Install audio player (choose one):
sudo apt-get install mpv              # Recommended
# OR
pip install yt-dlp && sudo apt-get install ffmpeg
```

### Step 2: Authenticate YouTube Music (2 minutes)

Choose one method:

#### Option A: OAuth (Recommended - More Reliable)

```bash
cd robot

# 1. Make sure you have client_secrets.json (Google OAuth credentials)
# 2. Run auth script
python auth_youtube.py

# 3. Open the URL it shows on your phone
# 4. Log in and authorize
# 5. Paste the code back
# 6. Done! Token saved to ytmusic_oauth.json
```

#### Option B: Cookie-Based (Quick Alternative)

```bash
cd robot
python setup_youtube_music.py

# Follow prompts to paste browser cookies
```

### Step 3: Test It! (2 minutes)

```bash
# Test music player
cd robot
python music_player.py

# Should show available genres and let you play a song
```

### Step 4: Say "Hey Rovy, dance!" 🎉

That's it! Now you can:

```
"Hey Rovy, dance!"                  → Dances with music
"Hey Rovy, dance to classical!"     → Classical music dance
"Hey Rovy, play music!"             → Just plays music
"Hey Rovy, stop music!"             → Stops playback
```

## What You Get

✅ **Random Song Selection** - Never the same song twice  
✅ **9 Music Genres** - Dance, classical, jazz, rock, pop, chill, electronic, fun, party  
✅ **Synchronized Dancing** - Robot dances to the music  
✅ **Voice Control** - From phone or robot mic  
✅ **OLED Display** - Shows currently playing song  
✅ **Auto Management** - Music stops when dance ends  

## Troubleshooting

### "YouTube Music not authenticated"

**Fix:**
```bash
cd robot
python auth_youtube.py
```

### "No playback method available"

**Fix:**
```bash
sudo apt-get install mpv
```

### "No songs found"

**Fix:**
- Check internet connection
- Try different genre
- Re-authenticate if needed

## Voice Commands Cheat Sheet

### Music Only
```
"Hey Rovy, play music"              → Random fun music
"Hey Rovy, play dance music"        → Dance/EDM
"Hey Rovy, play classical music"    → Classical pieces
"Hey Rovy, play jazz"               → Jazz standards  
"Hey Rovy, stop music"              → Stops playback
```

### Dance with Music
```
"Hey Rovy, dance!"                  → Party dance with dance music
"Hey Rovy, do a wiggle dance!"      → Wiggle with music
"Hey Rovy, dance to classical!"     → Elegant classical dance
"Hey Rovy, spin dance with rock!"   → Rock spin dance
```

## REST API Examples

### Dance with Music
```bash
curl -X POST http://100.72.107.106:8000/dance \
  -H "Content-Type: application/json" \
  -d '{
    "style": "party",
    "duration": 15,
    "with_music": true,
    "music_genre": "dance"
  }'
```

### Just Play Music
```bash
curl -X POST http://100.72.107.106:8000/music \
  -H "Content-Type: application/json" \
  -d '{
    "action": "play",
    "genre": "jazz"
  }'
```

### Stop Music
```bash
curl -X POST http://100.72.107.106:8000/music \
  -H "Content-Type: application/json" \
  -d '{"action": "stop"}'
```

## Files Reference

- **Setup**: `robot/auth_youtube.py` or `robot/setup_youtube_music.py`
- **Music Player**: `robot/music_player.py`
- **Full Docs**: `MUSIC_INTEGRATION.md`
- **Dance Docs**: `DANCING_MODE.md`

## Next Steps

1. ✅ Set up authentication
2. ✅ Test music playback
3. ✅ Try different genres
4. ✅ Create dance party!
5. 🎉 Have fun!

---

**Tip:** Say "Hey Rovy, dance to classical!" for an elegant performance, or "Hey Rovy, dance!" for a high-energy party! 🎵💃🎉

