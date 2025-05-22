import os
import tempfile
import streamlit as st
import requests
import subprocess
from urllib.parse import urlparse
from speechbrain.inference.classifiers import EncoderClassifier
import glob
import shutil
import torchaudio
import warnings
from pydub import AudioSegment

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configuration
SPEECHBRAIN_MODEL = "Jzuluaga/accent-id-commonaccent_ecapa"

# Configure pydub
AudioSegment.converter = "ffmpeg"
AudioSegment.ffmpeg = "ffmpeg"
AudioSegment.ffprobe = "ffprobe"

# Initialize SpeechBrain model
@st.cache_resource
def load_accent_model():
    """Load SpeechBrain accent identification model"""
    try:
        model = EncoderClassifier.from_hparams(SPEECHBRAIN_MODEL)
        return model
    except Exception as e:
        st.error(f"Failed to load accent model: {e}")
        return None

def download_with_ytdlp(url, save_path):
    """Download video using yt-dlp and copy to save_path"""
    try:
        temp_dir = tempfile.mkdtemp()
        output_template = os.path.join(temp_dir, "download.%(ext)s")

        result = subprocess.run([
            'yt-dlp',
            '-f', 'best[ext=mp4]/best',
            '-o', output_template,
            url
        ], capture_output=True, text=True)

        if result.returncode != 0:
            raise ValueError(f"yt-dlp error: {result.stderr}")

        downloaded_files = glob.glob(os.path.join(temp_dir, "download.*"))
        if not downloaded_files:
            raise ValueError("yt-dlp did not produce any downloadable video")

        shutil.move(downloaded_files[0], save_path)
        shutil.rmtree(temp_dir)
        return True

    except Exception as e:
        st.error(f"yt-dlp download failed: {str(e)}")
        return False

def download_video(url, save_path):
    """Try direct download if direct mp4 URL, else fallback to yt-dlp"""
    try:
        if any(url.lower().endswith(ext) for ext in ['.mp4', '.mov', '.m4v', '.avi', '.wmv']):
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                'Accept': '*/*',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://www.google.com/'
            }

            with requests.get(url, headers=headers, stream=True, timeout=10) as response:
                response.raise_for_status()
                content_type = response.headers.get('Content-Type', '')
                if not any(x in content_type for x in ['video', 'octet-stream']):
                    raise ValueError(f"URL does not point to a video file (Content-Type: {content_type})")

                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                if os.path.getsize(save_path) == 0:
                    os.remove(save_path)
                    raise ValueError("Downloaded file is empty")

            return True
        else:
            return download_with_ytdlp(url, save_path)
    except Exception as e:
        return download_with_ytdlp(url, save_path)

def extract_audio(video_path, audio_path):
    """Audio extraction with multiple fallback methods"""
    try:
        # Method 1: Try pydub first
        try:
            audio = AudioSegment.from_file(video_path)
            audio.set_frame_rate(16000).set_channels(1).export(
                audio_path, 
                format="wav",
                codec="pcm_s16le"
            )
            return True
        except Exception as e:
            st.warning(f"Pydub failed, trying ffmpeg directly: {e}")
            
        # Method 2: Fallback to direct ffmpeg command
        cmd = [
            'ffmpeg',
            '-y',
            '-i', video_path,
            '-ac', '1',
            '-ar', '16000',
            '-acodec', 'pcm_s16le',
            audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(f"FFmpeg failed: {result.stderr}")
            
        return True
        
    except Exception as e:
        st.error(f"Audio extraction failed: {str(e)}")
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return False

def analyze_accent_speechbrain(audio_path, model):
    """Analyze accent using SpeechBrain model"""
    try:
        if model is None:
            return None, None
        
        if not os.path.exists(audio_path):
            raise ValueError("Audio file does not exist")
        
        if os.path.getsize(audio_path) == 0:
            raise ValueError("Audio file is empty")
        
        abs_audio_path = os.path.abspath(audio_path)
        
        try:
            result = model.classify_file(abs_audio_path)
            predicted_accent = result[-1][0]
            
            confidence = None
            if len(result) >= 3:
                try:
                    confidence_tensor = result[1]
                    if hasattr(confidence_tensor, 'item'):
                        confidence = float(confidence_tensor.item()) * 100
                    elif hasattr(confidence_tensor, 'max'):
                        confidence = float(confidence_tensor.max()) * 100
                except:
                    confidence = None
            
            return predicted_accent, confidence
            
        except Exception as e:
            st.error(f"classify_file failed: {str(e)}")
            return None, None
        
    except Exception as e:
        st.error(f"Accent analysis failed: {str(e)}")
        return None, None

def format_accent_name(accent):
    """Format accent name for display"""
    accent_mapping = {
        'england': 'English (England)',
        'scotland': 'English (Scotland)',
        'ireland': 'English (Ireland)', 
        'wales': 'English (Wales)',
        'australia': 'English (Australia)',
        'newzealand': 'English (New Zealand)',
        'canada': 'English (Canada)',
        'us': 'English (United States)',
        'southafrica': 'English (South Africa)',
        'india': 'English (India)'
    }
    
    return accent_mapping.get(accent.lower(), accent.replace('_', ' ').title())

# --- Streamlit UI ---
st.title("🎙️ Accent Analyzer")
st.write("Analyze English accents from video URLs")

# Check for ffmpeg
try:
    subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True)
    st.success("FFmpeg is available")
except:
    st.warning("FFmpeg is not properly installed. Audio processing may fail.")

# Load model
with st.spinner("Loading model..."):
    accent_model = load_accent_model()

if accent_model is None:
    st.error("Failed to load the accent detection model.")
    st.stop()

# Input
url = st.text_input("Video URL:", placeholder="https://example.com/video.mp4")

if st.button("Analyze", type="primary"):
    if not url:
        st.warning("Please enter a video URL")
        st.stop()

    with st.status("Processing...", expanded=True) as status:
        video_path = None
        audio_path = None
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
                video_path = tmp_video.name
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                audio_path = tmp_audio.name

            st.write("Downloading video...")
            if not download_video(url, video_path):
                st.error("Failed to download video")
                st.stop()

            st.write("Extracting audio...")
            if not extract_audio(video_path, audio_path):
                st.error("Failed to extract audio")
                st.stop()

            st.write("Analyzing accent...")
            predicted_accent, confidence = analyze_accent_speechbrain(audio_path, accent_model)

            status.update(label="Analysis complete", state="complete")

            # Display results
            if predicted_accent:
                accent_name = format_accent_name(predicted_accent)
                
                st.success("Analysis Complete!")
                st.write(f"**Detected Accent:** {accent_name}")
                
                if confidence is not None:
                    st.write(f"**Confidence:** {confidence:.1f}%")
                    st.progress(confidence / 100)
                
            else:
                st.error("Could not determine accent from the audio")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
        finally:
            # Clean up temporary files
            for path in [video_path, audio_path]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except:
                        pass
