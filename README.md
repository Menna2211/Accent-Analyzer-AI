# 🎙️ Accent Analyzer

A Streamlit web application that analyzes English accents from video URLs using AI-powered speech recognition.

## Features

- **Video Support**: YouTube, Loom, direct MP4 links, and other video URLs
- **Accent Detection**: Identifies various English accents (US, UK, Australia, Canada, India, etc.)
- **Confidence Scoring**: Shows prediction confidence with visual progress bar
- **Simple Interface**: Clean, user-friendly web interface

## Prerequisites

Before running the application, ensure you have the following installed:

### System Dependencies
- **Python 3.8+**
- **FFmpeg** (for audio extraction)
- **yt-dlp** (for video downloading)

#### Install FFmpeg:
- **Windows**: Download from https://ffmpeg.org/download.html
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`

#### Install yt-dlp:
```bash
pip install yt-dlp
```
Or download binary from: https://github.com/yt-dlp/yt-dlp/releases

## Installation

1. **Clone or download the project files**
2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the application**:
   ```bash
   streamlit run accent_analyzer.py
   ```

2. **Open your browser** to the URL shown (typically `http://localhost:8501`)

3. **Analyze accents**:
   - Paste a video URL in the input field
   - Click "Analyze" 
   - Wait for processing (download → extract audio → analyze)
   - View results with detected accent and confidence score

## Supported Video Sources

- **Direct video files**: `.mp4`, `.mov`, `.avi`, `.wmv`
- **YouTube**: Any public YouTube video
- **Loom**: Public Loom recordings
- **Other platforms**: Most video URLs supported by yt-dlp

## Supported Accents

The model can identify various English accents including:
- English (England, Scotland, Ireland, Wales)
- English (United States)
- English (Australia, New Zealand)
- English (Canada)
- English (South Africa, India)

## How It Works

1. **Video Download**: Downloads video using direct HTTP or yt-dlp fallback
2. **Audio Extraction**: Uses FFmpeg to extract audio in 16kHz mono WAV format
3. **Accent Analysis**: Uses SpeechBrain's pre-trained accent classification model
4. **Results**: Displays detected accent with confidence percentage

## Troubleshooting

### Common Issues

**"Failed to load accent model"**
- Ensure stable internet connection (model downloads ~500MB on first run)
- Check Python dependencies are installed correctly

**"Failed to download video"**
- Verify the URL is accessible and public
- Check if yt-dlp is installed and up-to-date
- Some platforms may block automated downloads

**"Failed to extract audio"**
- Ensure FFmpeg is installed and in PATH
- Check if the downloaded video file is valid

**"Could not determine accent"**
- Audio quality may be too poor
- Background noise may interfere with analysis
- Try a different video with clearer speech

### Performance Notes

- First run downloads the SpeechBrain model (~500MB)
- Processing time depends on video length and internet speed
- Longer videos may take several minutes to process

## Technical Details

- **Framework**: Streamlit
- **AI Model**: SpeechBrain ECAPA-TDNN accent classifier
- **Audio Processing**: FFmpeg + PyTorch audio
- **Video Download**: yt-dlp with requests fallback

## Requirements

See `requirements.txt` for complete Python dependencies.

## License

This is a proof-of-concept application. Please ensure compliance with video platform terms of service when using.
