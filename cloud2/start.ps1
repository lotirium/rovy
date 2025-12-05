# PowerShell start script for Cloud 2 - Autonomous Personality Service

# Set environment variables (adjust as needed)
# Set your OpenAI API key here or as environment variable
# IMPORTANT: Set your API key as environment variable or edit this file
$env:OPENAI_API_KEY = if ($env:OPENAI_API_KEY) { $env:OPENAI_API_KEY } else { "" }
$env:ROVY_ROBOT_IP = if ($env:ROVY_ROBOT_IP) { $env:ROVY_ROBOT_IP } else { "100.72.107.106" }
$env:OPENAI_MODEL = if ($env:OPENAI_MODEL) { $env:OPENAI_MODEL } else { "gpt-4o" }
$env:PERSONALITY_NAME = if ($env:PERSONALITY_NAME) { $env:PERSONALITY_NAME } else { "Android" }
$env:PERSONALITY_OBS_INTERVAL = if ($env:PERSONALITY_OBS_INTERVAL) { $env:PERSONALITY_OBS_INTERVAL } else { "3.0" }
$env:PERSONALITY_SPEECH_COOLDOWN = if ($env:PERSONALITY_SPEECH_COOLDOWN) { $env:PERSONALITY_SPEECH_COOLDOWN } else { "2.0" }
$env:PERSONALITY_VISION = if ($env:PERSONALITY_VISION) { $env:PERSONALITY_VISION } else { "true" }

# Check for OpenAI API key
if (-not $env:OPENAI_API_KEY) {
    Write-Host "ERROR: OPENAI_API_KEY not set!" -ForegroundColor Red
    Write-Host "Set it with: `$env:OPENAI_API_KEY='your-key-here'" -ForegroundColor Yellow
    exit 1
}

# Run the personality service
Set-Location $PSScriptRoot
python main.py

