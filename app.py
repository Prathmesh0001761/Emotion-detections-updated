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

# Best color combinations for dark mode:
primary_color = '#ff6f91'  # Vibrant pink
secondary_color = '#a78bfa'  # Soft purple
text_color = '#f5f7fa'  # Light gray
background_dark = '#1a253b'  # Dark blue-gray
card_bg = 'rgba(255, 255, 255, 0.05)'  # Translucent white
border_color = 'rgba(255, 255, 255, 0.1)' # Subtle white

# Best color combinations for light mode:
light_primary_color = '#dc3545' # Strong red
light_secondary_color = '#007bff' # Classic blue
light_text_color = '#212529' # Dark gray
background_light = '#f8f9fa' # Light gray
light_card_bg = 'rgba(0, 0, 0, 0.03)' # Translucent black
light_border_color = 'rgba(0, 0, 0, 0.1)' # Subtle black

# Custom CSS for a refined UI
def apply_custom_ui():
    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@500;700&family=Roboto:wght@300;400;500&display=swap');

            /* Root styles */
            :root {{
                --primary-color: {primary_color};
                --secondary-color: {secondary_color};
                --text-color: {text_color};
                --background-dark: {background_dark};
                --background-light: {background_light};
                --card-bg: {card_bg};
                --border-color: {border_color};
            }}

            .light-mode {{
                --primary-color: {light_primary_color};
                --secondary-color: {light_secondary_color};
                --text-color: {light_text_color};
                --background-dark: {background_light};
                --card-bg: {light_card_bg};
                --border-color: {light_border_color};
            }}

            /* Main app styles */
            .stApp {{
                background: var(--background-dark);
                font-family: 'Roboto', sans-serif;
                color: var(--text-color);
                overflow-x: hidden;
            }}

            .light-mode .stApp {{
                background: var(--background-light);
            }}

            /* Main container */
            .content-container {{
                background: var(--card-bg);
                border-radius: 12px;
                padding: 40px; /* Increased padding */
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                border: 1px solid var(--border-color);
                animation: fadeIn 1s ease-out;
                margin: 30px auto; /* Increased margin */
                max-width: 1200px; /* Increased max-width */
            }}

            .light-mode .content-container {{
                background: var(--light-card-bg);
                border-color: var(--light-border-color);
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            }}

            /* Heading styles */
            h1 {{
                color: var(--primary-color);
                text-align: center;
                font-weight: 700;
                margin-bottom: 15px; /* Increased margin-bottom */
                font-size: 3.5rem; /* Increased font-size */
                font-family: 'Montserrat', sans-serif;
                letter-spacing: 1.2px;
                transition: color 0.3s ease;
            }}

            .light-mode h1 {{
                color: var(--primary-color);
            }}

            h1:hover {{
                color: {secondary_color};
            }}

            .light-mode h1:hover {{
                color: {light_secondary_color};
            }}

            /* Subheader styles */
            .subheader {{
                color: rgba(245, 247, 250, 0.8);
                text-align: center;
                font-size: 1.8rem; /* Increased font-size */
                margin-bottom: 50px; /* Increased margin-bottom */
                font-weight: 400;
                transition: color 0.3s ease;
            }}

            .light-mode .subheader {{
                color: #4b5563;
            }}

            .subheader:hover {{
                color: var(--text-color);
            }}

            .light-mode .subheader:hover {{
                color: var(--text-color);
            }}

            /* Tab container */
            .stTabs {{
                background: transparent;
                padding: 15px; /* Increased padding */
                display: flex;
                justify-content: center;
            }}

            [data-baseweb="tab-list"] {{
                gap: 20px; /* Increased gap */
                justify-content: center;
                background: transparent;
                border-bottom: 2px solid rgba(255, 255, 255, 0.1);
            }}

            .light-mode [data-baseweb="tab-list"] {{
                border-bottom: 2px solid rgba(0, 0, 0, 0.1);
            }}

            [data-baseweb="tab"] {{
                background: transparent !important;
                border-radius: 8px !important;
                padding: 15px 30px !important; /* Increased padding */
                transition: all 0.3s ease !important;
                border: none !important;
                font-size: 1.3rem; /* Increased font-size */
                color: rgba(245, 247, 250, 0.7) !important;
                font-family: 'Roboto', sans-serif;
                position: relative;
            }}

            .light-mode [data-baseweb="tab"] {{
                color: #4b5563 !important;
            }}

            [data-baseweb="tab"]:hover {{
                color: var(--text-color) !important;
                background: rgba(255, 111, 145, 0.1) !important;
            }}

            .light-mode [data-baseweb="tab"]:hover {{
                color: var(--text-color) !important;
                background: rgba(0, 0, 0, 0.05) !important;
            }}

            [aria-selected="true"] {{
                color: var(--text-color) !important;
                font-weight: 500;
                background: linear-gradient(45deg, var(--primary-color), var(--secondary-color)) !important;
                border-radius: 8px !important;
            }}

            .light-mode [aria-selected="true"] {{
                color: var(--light-text-color) !important;
                background: linear-gradient(45deg, var(--light-primary-color), var(--light-secondary-color)) !important;
            }}

            /* File uploader */
            .stFileUploader {{
                margin: 30px 0; /* Increased margin */
                border: 2px dashed rgba(255, 111, 145, 0.3) !important;
                border-radius: 12px !important;
                padding: 50px !important; /* Increased padding */
                transition: all 0.3s ease !important;
                background: var(--card-bg) !important;
                font-size: 1.3rem; /* Increased font-size */
                color: rgba(245, 247, 250, 0.8);
            }}

            .light-mode .stFileUploader {{
                border-color: rgba(0, 0, 0, 0.1) !important;
                background: var(--light-card-bg) !important;
                color: #4b5563;
            }}

            .stFileUploader:hover {{
                border-color: var(--primary-color) !important;
                background: rgba(255, 111, 145, 0.1) !important;
            }}

            .light-mode .stFileUploader:hover {{
                border-color: var(--light-primary-color) !important;
                background: rgba(0, 0, 0, 0.05) !important;
            }}

            /* Button styles */
            .stButton>button {{
                background: linear-gradient(45deg, var(--primary-color), var(--secondary-color)) !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 10px !important;
                padding: 15px 40px !important; /* Increased padding */
                font-weight: 500 !important;
                font-family: 'Roboto', sans-serif;
                transition: all 0.3s ease !important;
                box-shadow: 0 3px 10px rgba(255, 111, 145, 0.2) !important;
                width: 100%;
                font-size: 1.3rem; /* Increased font-size */
                animation: pulse 2s infinite;
            }}

            .light-mode .stButton>button {{
                background: linear-gradient(45deg, var(--light-primary-color), var(--light-secondary-color)) !important;
                color: #ffffff !important;
                box-shadow: 0 2px 8px rgba(0, 123, 255, 0.2) !important;
            }}

            .stButton>button:hover {{
                background: linear-gradient(45deg, {secondary_color}, #b69efc) !important;
                box-shadow: 0 5px 15px rgba(255, 111, 145, 0.3) !important;
                transform: translateY(-2px) !important;
            }}

            .light-mode .stButton>button:hover {{
                background: linear-gradient(45deg, {light_secondary_color}, #66a3ff) !important;
                box-shadow: 0 3px 10px rgba(0, 123, 255, 0.3) !important;
            }}

            /* Audio container */
            .audio-container {{
                margin: 30px 0; /* Increased margin */
                border-radius: 12px;
                overflow: hidden;
                background: var(--card-bg);
                padding: 20px; /* Increased padding */
                transition: background 0.3s ease;
                box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
            }}

            .light-mode .audio-container {{
                background: var(--light-card-bg);
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            }}

            .audio-container:hover {{
                background: rgba(255, 111, 145, 0.1);
            }}

            .light-mode .audio-container:hover {{
                background: rgba(0, 0, 0, 0.05);
            }}

            /* Result box */
            .result-box {{
                background: var(--card-bg);
                border-radius: 12px;
                padding: 35px; /* Increased padding */
                margin: 30px 0; /* Increased margin */
                border: 1px solid var(--border-color);
                animation: fadeIn 1s ease-out;
                box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
            }}

            .light-mode .result-box {{
                background: var(--light-card-bg);
                border-color: var(--light-border-color);
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            }}

            .emotion-display {{
                font-size: 3.5rem; /* Increased font-size */
                margin: 25px 0; /* Increased margin */
                text-align: center;
                color: var(--primary-color);
                font-weight: 500;
                font-family: 'Montserrat', sans-serif;
            }}

            .light-mode .emotion-display {{
                color: var(--primary-color);
            }}

            .confidence {{
                color: rgba(245, 247, 250, 0.8);
                text-align: center;
                font-size: 1.5rem; /* Increased font-size */
                font-weight: 400;
            }}

            .light-mode .confidence {{
                color: #4b5563;
            }}

            /* Selectbox styles */
            .stSelectbox > div {{
                background: var(--card-bg);
                border-radius: 10px;
                border: 1px solid var(--border-color);
                transition: all 0.3s ease;
                color: var(--text-color);
                font-size: 1.3rem; /* Increased font-size */
            }}

            .light-mode .stSelectbox > div {{
                background: var(--light-card-bg);
                border-color: var(--light-border-color);
                color: #212529;
            }}

            .stSelectbox > div:hover {{
                border-color: var(--primary-color);
            }}

            .light-mode .stSelectbox > div:hover {{
                border-color: var(--light-primary-color);
            }}

            .stSelectbox [data-baseweb="select"] > div {{
                background: transparent;
                color: var(--text-color);
            }}

            .light-mode .stSelectbox [data-baseweb="select"] > div {{
                color: #212529;
            }}

            /* Countdown container */
            .countdown-container {{
                background: var(--card-bg);
                border-radius: 12px;
                padding: 25px; /* Increased padding */
                margin: 25px 0; /* Increased margin */
                text-align: center;
                border: 1px solid var(--border-color);
                box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
            }}

            .light-mode .countdown-container {{
                background: var(--light-card-bg);
                border-color: var(--light-border-color);
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            }}

            .countdown-text {{
                font-size: 3rem; /* Increased font-size */
                font-weight: 500;
                color: var(--primary-color);
                font-family: 'Montserrat', sans-serif;
            }}

            .light-mode .countdown-text {{
                color: var(--primary-color);
            }}

            /* Alert box */
            .alert-box {{
                background: rgba(255, 111, 145, 0.1);
                border-radius: 12px;
                padding: 25px; /* Increased padding */
                margin: 25px 0; /* Increased margin */
                border-left: 4px solid var(--primary-color);
                color: var(--text-color);
                font-size: 1.3rem; /* Increased font-size */
                animation: fadeIn 1s ease-out;
                box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
            }}

            .light-mode .alert-box {{
                background: rgba(220, 53, 69, 0.1);
                border-left-color: var(--light-primary-color);
                color: #212529;
                box-shadow: 0 2px 8px rgba(220,53, 69, 0.1);
            }}

            /* Plot container */
            .plot-container {{
                border: 1px solid var(--border-color);
                border-radius: 12px;
                padding: 20px; /* Increased padding */
                margin: 30px 0; /* Increased margin */
                background: var(--card-bg);
                box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
            }}

            .light-mode .plot-container {{
                background: var(--light-card-bg);
                border-color: var(--light-border-color);
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            }}

            /* File info */
            .file-info {{
                background: var(--card-bg);
                border-radius: 12px;
                padding: 25px; /* Increased padding */
                margin: 25px 0; /* Increased margin */
                font-size: 1.3rem; /* Increased font-size */
                color: rgba(245, 247, 250, 0.8);
                box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
                animation: fadeIn 1s ease-out;
            }}

            .light-mode .file-info {{
                background: var(--light-card-bg);
                color: #4b5563;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            }}

            /* Custom spinner */
            .custom-spinner {{
                width: 50px; /* Increased size */
                height: 50px; /* Increased size */
                border: 5px solid rgba(255, 111, 145, 0.3); /* Increased border width */
                border-top: 5px solid var(--primary-color); /* Increased border width */
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 30px auto; /* Increased margin */
            }}

            .light-mode .custom-spinner {{
                border-color: rgba(220, 53, 69, 0.3);
                border-top-color: var(--light-primary-color);
            }}

            /* Animations */
            @keyframes fadeIn {{
                0% {{ opacity: 0; transform: translateY(10px); }}
                100% {{ opacity: 1; transform: translateY(0); }}
            }}

            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}

            @keyframes pulse {{
                0% {{ transform: scale(1); box-shadow: 0 3px 10px rgba(255, 111, 145, 0.2); }}
                50% {{ transform: scale(1.02); box-shadow: 0 5px 15px rgba(255, 111, 145, 0.3); }}
                100% {{ transform: scale(1); box-shadow: 0 3px 10px rgba(255, 111, 145, 0.2); }}
            }}

            .light-mode @keyframes pulse {{
                0% {{ transform: scale(1); box-shadow: 0 2px 8px rgba(0, 123, 255, 0.2); }}
                50% {{ transform: scale(1.02); box-shadow: 0 3px 10px rgba(0, 123, 255, 0.3); }}
                100% {{ transform: scale(1); box-shadow: 0 2px 8px rgba(0, 123, 255, 0.2); }}
            }}

            /* Responsive adjustments */
            @media (max-width: 768px) {{
                h1 {{
                    font-size: 2.5rem; /* Increased font-size for smaller screens */
                }}
                .subheader {{
                    font-size: 1.5rem; /* Increased font-size for smaller screens */
                }}
                .content-container {{
                    padding: 25px; /* Adjusted padding for smaller screens */
                    margin: 15px; /* Adjusted margin for smaller screens */
                }}
                [data-baseweb="tab"] {{
                    padding: 12px 20px !important; /* Adjusted padding for smaller screens */
                    font-size: 1.1rem; /* Adjusted font-size for smaller screens */
                }}
                .stFileUploader {{
                    padding: 40px !important; /* Adjusted padding for smaller screens */
                }}
                .stButton>button {{
                    padding: 12px 30px !important; /* Adjusted padding for smaller screens */
                    font-size: 1.1rem; /* Adjusted font-size for smaller screens */
                }}
                .emotion-display {{
                    font-size: 2.8rem; /* Adjusted font-size for smaller screens */
                }}
                .countdown-text {{
                    font-size: 2rem; /* Adjusted font-size for smaller screens */
                }}
            }}
        </style>
    """, unsafe_allow_html=True)

# Voice Activity Detection: Trim silence from audio
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

# Waveform plot with a refined theme
def wave_plot(data, sampling_rate):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='none') # Increased figure size

    librosa.display.waveshow(y=data, sr=sampling_rate, color=primary_color, ax=ax)

    ax.spines['bottom'].set_color(text_color)
    ax.spines['left'].set_color(text_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', colors=text_color, labelsize=12) # Increased label size
    ax.tick_params(axis='y', colors=text_color, labelsize=12) # Increased label size
    ax.set_xlabel("Time (s)", color=text_color, fontsize=14) # Increased font size
    ax.set_ylabel("Amplitude", color=text_color, fontsize=14) # Increased font size
    ax.set_title("Audio Waveform", fontweight="bold", color=text_color, fontsize=16, pad=15) # Increased font size
    ax.grid(color='#444444', alpha=0.3)

    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close()

# Pie chart for emotion distribution
def plot_emotion_pie_chart(emotion_counts):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9, 9), facecolor='none') # Increased figure size

    emotions = emotion_counts.index
    counts = emotion_counts.values
    colors = plt.cm.Paired(np.linspace(0, 1, len(emotions)))
    text_kwargs = {'color': text_color, 'fontsize': 14} # Increased font size

    ax.pie(counts, labels=emotions, autopct='%1.1f%%', startangle=140, colors=colors,
            textprops=text_kwargs)
    ax.set_title("Emotion Distribution", fontweight="bold", color=text_color, fontsize=16, pad=15) # Increased font size

    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close()

# CNN model prediction with mental health alerts
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

        # Trim silence using VAD before prediction
        trimmed_data, sr = trim_silence(data, sampling_rate)
        if len(trimmed_data) == 0:
            st.warning("Audio is entirely silent after trimming. Skipping prediction.")
            return None, None

        # Extract MFCCs from trimmed audio
        mfccs = np.mean(librosa.feature.mfcc(y=trimmed_data, sr=sr, n_mfcc=40,
                                            n_fft=2048, hop_length=512).T, axis=0)
        X_test = np.expand_dims([mfccs], axis=2)

        # Custom spinner using a placeholder
        spinner_placeholder = st.empty()
        spinner_placeholder.markdown('<div class="custom-spinner"></div>', unsafe_allow_html=True)

        predict = model.predict(X_test, verbose=0)

        # Clear the spinner
        spinner_placeholder.empty()

        detected_emotion = emotion_dict[np.argmax(predict)]
        confidence = np.max(predict) * 100

        # Display file info
        source = "Real-time Audio" if is_real_time else file_name
        st.markdown(f"""
            <div class="file-info">
                <strong>Source:</strong> {source}<br>
                <strong>Duration (after trimming):</strong> {len(trimmed_data)/sr:.2f} seconds<br>
                <strong>Sample Rate:</strong> {sr} Hz
            </div>
        """, unsafe_allow_html=True)

        # Display prediction
        st.markdown(f"""
            <div class="result-box">
                <h3 style="color: var(--text-color); text-align: center; margin-bottom: 25px; font-size: 1.8rem;">Emotion Detection Result</h3>
                <div class="emotion-display">{detected_emotion}</div>
                <div class="confidence">Confidence: {confidence:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)

        # Alert for high-risk emotions
        high_risk_emotions = ["Sad", "Angry", "Fear"]
        emotion_name = detected_emotion.split(" ")[1]
        if emotion_name in high_risk_emotions and confidence > 80:
            st.markdown(f"""
                <div class="alert-box">
                    ⚠️ <strong>Alert:</strong> Detected high-confidence {emotion_name} emotion ({confidence:.2f}%). Consider reviewing this session.
                </div>
            """, unsafe_allow_html=True)

        return detected_emotion.split(" ")[1], confidence

    except Exception as e:
        st.error(f"Error in CNN prediction: {str(e)}")
        return None, None

# Real-time audio recording with countdown
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

# Main app function with refined UI
def main():
    apply_custom_ui()

    st.markdown("""
        <div class="content-container">
            <h1>🎤 Speech Emotion Classifier</h1>
            <div class="subheader">Unlock the Emotions in Every Voice</div>
        </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📂 Upload Audio", "🎙️ Real-time Detection"])

    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
        st.session_state.audio_sr = None
        st.session_state.audio_name = None

    with tab1:
        audio_files = st.file_uploader(
            "Upload your audio files",
            type=['wav', 'mp3'],
            accept_multiple_files=True,
            help="Supported formats: WAV, MP3 | Max size: 200MB"
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

                    # Custom spinner for processing
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
                st.markdown("### Emotion Distribution Across Files")
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
            "Select recording duration",
            list(duration_options.keys()),
            index=0
        )
        duration = duration_options[selected_duration]

        if st.button("Start Recording"):
            audio_data, audio_sr = record_audio(duration=duration)
            if audio_data is not None:
                st.session_state.audio_data = audio_data
                st.session_state.audio_sr = audio_sr
                st.session_state.audio_name = "Real-time Recording"

                temp_audio = BytesIO()
                sf.write(temp_audio, audio_data, audio_sr, format='WAV')
                temp_audio.seek(0)
                st.markdown('<div class="audio-container">', unsafe_allow_html=True)
                st.audio(temp_audio, format='audio/wav')
                st.markdown('</div>', unsafe_allow_html=True)

                # Custom spinner for processing
                spinner_placeholder = st.empty()
                spinner_placeholder.markdown('<div class="custom-spinner"></div>', unsafe_allow_html=True)

                wave_plot(st.session_state.audio_data, st.session_state.audio_sr)
                prediction(st.session_state.audio_data, st.session_state.audio_sr,
                                        st.session_state.audio_name, is_real_time=True)

                spinner_placeholder.empty()

if __name__ == '__main__':
    main()