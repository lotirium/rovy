#!/bin/bash
# Start script for Cloud 2 - Autonomous Personality Service

# Set environment variables (adjust as needed)
# Set your OpenAI API key here or as environment variable
# IMPORTANT: Set your API key as environment variable or edit this file
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export ROVY_ROBOT_IP="${ROVY_ROBOT_IP:-100.72.107.106}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o}"
export PERSONALITY_NAME="${PERSONALITY_NAME:-Android}"
export PERSONALITY_OBS_INTERVAL="${PERSONALITY_OBS_INTERVAL:-3.0}"
export PERSONALITY_SPEECH_COOLDOWN="${PERSONALITY_SPEECH_COOLDOWN:-2.0}"
export PERSONALITY_VISION="${PERSONALITY_VISION:-true}"

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set!"
    echo "Set it with: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

# Run the personality service
cd "$(dirname "$0")"
python main.py

