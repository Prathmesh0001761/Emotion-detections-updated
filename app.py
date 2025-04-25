import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import time
from io import BytesIO
import soundfile as sf
import sounddevice as sd
import scipy.signal
import pandas as pd
import streamlit.components.v1 as components
import json
import uuid

# Color scheme
primary_color = '#4A90E2'  # Blue
accent_color = '#F4A261'   # Orange
text_color = '#FFFFFF'
background_color = '#1F2A44'  # Dark navy
card_background = '#2A3B61'   # Lighter navy
border_color = '#3B4A6B'

# Custom CSS with enhanced chatbot styling
def apply_custom_ui():
    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
            
            :root {{
                --primary-color: {primary_color};
                --accent-color: {accent_color};
                --text-color: {text_color};
                --background-color: {background_color};
                --card-background: {card_background};
                --border-color: {border_color};
            }}

            .stApp {{
                background: var(--background-color);
                font-family: 'Poppins', sans-serif;
                color: var(--text-color);
            }}

            .main-container {{
                background: var(--card-background);
                border-radius: 16px;
                padding: 2rem;
                margin: 1rem auto;
                max-width: 1000px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
                transition: transform 0.3s ease;
            }}

            h1 {{
                color: var(--primary-color);
                text-align: center;
                font-size: 2.5rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }}

            .subheader {{
                color: rgba(255, 255, 255, 0.7);
                text-align: center;
                font-size: 1.2rem;
                font-weight: 300;
                margin-bottom: 2rem;
            }}

            .stTabs {{
                background: transparent;
                border-bottom: 1px solid var(--border-color);
                margin-bottom: 1.5rem;
            }}

            [data-baseweb="tab-list"] {{
                gap: 1rem;
                justify-content: center;
            }}

            [data-baseweb="tab"] {{
                color: rgba(255, 255, 255, 0.6);
                font-size: 1rem;
                padding: 0.8rem 1.5rem;
                border-radius: 8px;
                transition: all 0.3s ease;
            }}

            [data-baseweb="tab"]:hover {{
                color: var(--text-color);
                background: rgba(255, 255, 255, 0.05);
            }}

            [aria-selected="true"] {{
                color: var(--text-color);
                background: var(--primary-color);
                font-weight: 600;
            }}

            .stFileUploader {{
                background: var(--card-background);
                border: 2px dashed var(--border-color);
                border-radius: 12px;
                padding: 2rem;
                transition: all 0.3s ease;
            }}

            .stFileUploader:hover {{
                border-color: var(--primary-color);
                background: rgba(74, 144, 226, 0.1);
            }}

            .stButton>button {{
                background: var(--primary-color);
                color: var(--text-color);
                border-radius: 8px;
                padding: 0.8rem 2rem;
                font-weight: 500;
                border: none;
                transition: all 0.3s ease;
                width: 100%;
            }}

            .stButton>button:hover {{
                background: var(--accent-color);
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
            }}

            .audio-container {{
                background: var(--card-background);
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1rem 0;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            }}

            .result-box {{
                background: var(--card-background);
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1rem 0;
                border: 1px solid var(--border-color);
                text-align: center;
            }}

            .emotion-display {{
                font-size: 2.2rem;
                color: var(--accent-color);
                font-weight: 600;
                margin: 1rem 0;
            }}

            .confidence {{
                color: rgba(255, 255, 255, 0.8);
                font-size: 1.1rem;
            }}

            .stSelectbox > div {{
                background: var(--card-background);
                border-radius: 8px;
                border: 1px solid var(--border-color);
                color: var(--text-color);
            }}

            .stSelectbox > div:hover {{
                border-color: var(--primary-color);
            }}

            .countdown-container {{
                background: var(--card-background);
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1rem 0;
                text-align: center;
            }}

            .countdown-text {{
                font-size: 1.8rem;
                color: var(--primary-color);
                font-weight: 500;
            }}

            .alert-box {{
                background: rgba(244, 162, 97, 0.1);
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1rem 0;
                border-left: 4px solid var(--accent-color);
                color: var(--text-color);
            }}

            .plot-container {{
                background: var(--card-background);
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1rem 0;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            }}

            .file-info {{
                background: var(--card-background);
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1rem 0;
                color: rgba(255, 255, 255, 0.8);
            }}

            .chatbot-container {{
                background: var(--card-background);
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1rem 0;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
                animation: fadeIn 0.5s ease-out;
                position: relative;
            }}

            .typing-indicator {{
                display: none;
                font-size: 0.9rem;
                color: rgba(255, 255, 255, 0.6);
                padding: 10px;
                text-align: left;
            }}

            .typing-indicator.active {{
                display: block;
            }}

            .typing-indicator::before {{
                content: '• • •';
                animation: blink 1s infinite;
            }}

            .custom-spinner {{
                width: 40px;
                height: 40px;
                border: 4px solid rgba(74, 144, 226, 0.3);
                border-top: 4px solid var(--primary-color);
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 2rem auto;
            }}

            @keyframes fadeIn {{
                0% {{ opacity: 0; transform: translateY(10px); }}
                100% {{ opacity: 1; transform: translateY(0); }}
            }}

            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}

            @keyframes blink {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0.3; }}
                100% {{ opacity: 1; }}
            }}

            @media (max-width: 768px) {{
                h1 {{ font-size: 2rem; }}
                .subheader {{ font-size: 1rem; }}
                .main-container {{ padding: 1.5rem; margin: 0.5rem; }}
                .emotion-display {{ font-size: 1.8rem; }}
                .countdown-text {{ font-size: 1.5rem; }}
            }}
        </style>
    """, unsafe_allow_html=True)

# Voice Activity Detection
def trim_silence(audio, sr, threshold_db=-40, frame_length=2048, hop_length=512):
    try:
        threshold = 10 ** (threshold_db / 20)
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        non_silent = rms > threshold
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        non_silent_times = times[non_silent]
        if len(non_silent_times) == 0:
            return audio, sr
        start_time = non_silent_times[0]
        end_time = non_silent_times[-1]
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        trimmed_audio = audio[start_sample:end_sample]
        return trimmed_audio, sr
    except Exception as e:
        st.error(f"Error trimming silence: {str(e)}")
        return audio, sr

# Waveform plot
def wave_plot(data, sampling_rate):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='none')
    librosa.display.waveshow(y=data, sr=sampling_rate, color=primary_color, ax=ax)
    ax.spines['bottom'].set_color(text_color)
    ax.spines['left'].set_color(text_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', colors=text_color, labelsize=10)
    ax.tick_params(axis='y', colors=text_color, labelsize=10)
    ax.set_xlabel("Time (s)", color=text_color, fontsize=12)
    ax.set_ylabel("Amplitude", color=text_color, fontsize=12)
    ax.set_title("Audio Waveform", fontweight="bold", color=text_color, fontsize=14)
    ax.grid(color='#444444', alpha=0.3)
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close()

# Pie chart for emotion distribution
def plot_emotion_pie_chart(emotion_counts):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='none')
    emotions = emotion_counts.index
    counts = emotion_counts.values
    colors = plt.cm.Paired(np.linspace(0, 1, len(emotions)))
    text_kwargs = {'color': text_color, 'fontsize': 12}
    ax.pie(counts, labels=emotions, autopct='%1.1f%%', startangle=140, colors=colors, textprops=text_kwargs)
    ax.set_title("Emotion Distribution", fontweight="bold", color=text_color, fontsize=14)
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close()

# CNN model prediction
def prediction(data, sampling_rate, file_name, is_real_time=False):
    emotion_dict = {
        0: "😐 Neutral",
        1: "😌 Calm",
        2: "😊 Happy",
        3: "😢 Sad",
        4: "😠 Angry",
        5: "😨 Fear",
        6: "🤢 Disgust",
        7: "😲 Surprise"
    }

    try:
        model = load_model("models/CnnModel.h5")
        trimmed_data, sr = trim_silence(data, sampling_rate)
        if len(trimmed_data) == 0:
            st.warning("Audio is entirely silent after trimming. Skipping prediction.")
            return None, None

        mfccs = np.mean(librosa.feature.mfcc(y=trimmed_data, sr=sr, n_mfcc=40,
                                             n_fft=2048, hop_length=512).T, axis=0)
        X_test = np.expand_dims([mfccs], axis=2)

        spinner_placeholder = st.empty()
        spinner_placeholder.markdown('<div class="custom-spinner"></div>', unsafe_allow_html=True)

        predict = model.predict(X_test, verbose=0)
        spinner_placeholder.empty()

        detected_emotion = emotion_dict[np.argmax(predict)]
        confidence = np.max(predict) * 100
        emotion_name = detected_emotion.split(" ")[1]

        source = "Real-time Audio" if is_real_time else file_name
        st.markdown(f"""
            <div class="file-info">
                <strong>Source:</strong> {source}<br>
                <strong>Duration:</strong> {len(trimmed_data)/sr:.2f} seconds<br>
                <strong>Sample Rate:</strong> {sr} Hz
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="result-box">
                <h3 style="color: var(--text-color); text-align: center; font-size: 1.5rem;">Detected Emotion</h3>
                <div class="emotion-display">{detected_emotion}</div>
                <div class="confidence">Confidence: {confidence:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)

        high_risk_emotions = ["Sad", "Angry", "Fear"]
        if emotion_name in high_risk_emotions and confidence > 80:
            st.markdown(f"""
                <div class="alert-box">
                    ⚠️ <strong>High-confidence {emotion_name}</strong> detected ({confidence:.2f}%). Let's talk about how you're feeling.
                </div>
            """, unsafe_allow_html=True)
            # Load and display chatbot with session ID
            session_id = str(uuid.uuid4())
            st.session_state.emotion_data = {
                "emotion": emotion_name,
                "confidence": confidence,
                "session_id": session_id
            }
            with open("chatbot.html", "r") as f:
                html_content = f.read()
            emotion_script = f"""
                <script>
                    window.emotionData = {json.dumps(st.session_state.emotion_data)};
                </script>
            """
            st.markdown('<div class="chatbot-container">', unsafe_allow_html=True)
            components.html(emotion_script + html_content, height=500, scrolling=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.session_state.emotion_data = None

        return emotion_name, confidence

    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None

# Real-time audio recording
def record_audio(duration=10, sample_rate=44100):
    try:
        countdown_placeholder = st.empty()
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        for remaining in range(duration, -1, -1):
            minutes = remaining // 60
            seconds = remaining % 60
            time_str = f"{minutes:02d}:{seconds:02d}" if remaining >= 60 else f"{seconds} seconds"
            countdown_placeholder.markdown(
                f"""
                <div class="countdown-container">
                    <div class="countdown-text">Recording: {time_str}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            time.sleep(1)
        sd.wait()
        audio = recording.flatten()
        countdown_placeholder.empty()
        st.info(f"Recording completed ({duration} seconds).")
        return audio, sample_rate
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
        return None, None

# Main app function
def main():
    apply_custom_ui()
    st.markdown("""
        <div class="main-container">
            <h1>🎵 Voice Emotion Analyzer</h1>
            <div class="subheader">Discover emotions through speech</div>
        </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📤 Upload Audio", "🎤 Live Recording"])

    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
        st.session_state.audio_sr = None
        st.session_state.audio_name = None
        st.session_state.emotion_data = None

    with tab1:
        audio_files = st.file_uploader(
            "Upload audio files",
            type=['wav', 'mp3'],
            accept_multiple_files=True,
            help="Supports WAV and MP3 formats"
        )
        if audio_files:
            emotions_detected = []
            for audio_file in audio_files:
                try:
                    with BytesIO(audio_file.read()) as f:
                        st.session_state.audio_data, st.session_state.audio_sr = librosa.load(f, sr=None)
                    st.session_state.audio_name = audio_file.name
                    st.markdown(f'<div class="audio-container"><strong>File: {audio_file.name}</strong>', unsafe_allow_html=True)
                    st.audio(audio_file, format=f'audio/{audio_file.name.split(".")[-1]}')
                    st.markdown('</div>', unsafe_allow_html=True)
                    spinner_placeholder = st.empty()
                    spinner_placeholder.markdown('<div class="custom-spinner"></div>', unsafe_allow_html=True)
                    wave_plot(st.session_state.audio_data, st.session_state.audio_sr)
                    emotion, confidence = prediction(st.session_state.audio_data, st.session_state.audio_sr,
                                                    st.session_state.audio_name, is_real_time=False)
                    spinner_placeholder.empty()
                    if emotion and confidence:
                        emotions_detected.append(emotion)
                except Exception as e:
                    st.error(f"Error loading audio file {audio_file.name}: {str(e)}")
            if len(emotions_detected) > 1:
                emotion_counts = pd.Series(emotions_detected).value_counts()
                st.markdown("### Emotion Distribution")
                plot_emotion_pie_chart(emotion_counts)

    with tab2:
        duration_options = {
            "10 seconds": 10,
            "30 seconds": 30,
            "1 minute": 60,
            "1 minute 30 seconds": 90,
            "2 minutes": 120
        }
        selected_duration = st.selectbox(
            "Recording duration",
            list(duration_options.keys()),
            index=0
        )
        duration = duration_options[selected_duration]
        if st.button("Record Audio"):
            audio_data, audio_sr = record_audio(duration=duration)
            if audio_data is not None:
                st.session_state.audio_data = audio_data
                st.session_state.audio_sr = audio_sr
                st.session_state.audio_name = "Live Recording"
                temp_audio = BytesIO()
                sf.write(temp_audio, audio_data, audio_sr, format='WAV')
                temp_audio.seek(0)
                st.markdown('<div class="audio-container">', unsafe_allow_html=True)
                st.audio(temp_audio, format='audio/wav')
                st.markdown('</div>', unsafe_allow_html=True)
                spinner_placeholder = st.empty()
                spinner_placeholder.markdown('<div class="custom-spinner"></div>', unsafe_allow_html=True)
                wave_plot(st.session_state.audio_data, st.session_state.audio_sr)
                prediction(st.session_state.audio_data, st.session_state.audio_sr,
                           st.session_state.audio_name, is_real_time=True)
                spinner_placeholder.empty()

if __name__ == '__main__':
    main()