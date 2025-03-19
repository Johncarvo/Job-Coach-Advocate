import streamlit as st
import openai
import assemblyai as aai
import numpy as np
from pathlib import Path
import os
from dotenv import load_dotenv
from streamlit_webrtc import webrtc_streamer
import av

load_dotenv()

st.title("AI Job Profile Generator")

# Input method selection
input_method = st.radio("Choose input method:", ["Text", "Audio"])

candidate_info = None

def generate_job_profile(candidate_info):
    if not candidate_info:
        st.error("Candidate information is missing!")
        return

    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI()
    system_prompt = """You are a professional job profile writer. Create a structured profile with the following sections:
    1. Professional Summary
    2. Key Skills
    3. Work Experience
    4. Education
    5. Certifications (if any)
    6. Technical Skills (if applicable)

    Format the output in markdown."""

    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create a structured job profile from this information: {candidate_info}"}
        ]
    )

    st.markdown(response.choices[0].message.content)

    # Add download button for the profile
    st.download_button(
        label="Download Profile",
        data=response.choices[0].message.content,
        file_name="job_profile.md",
        mime="text/markdown"
    )

if input_method == "Text":
    candidate_info = st.text_area("Enter candidate information:", height=200)
    if candidate_info and st.button("Generate Profile"):
        generate_job_profile(candidate_info)

else:
    # Initialize session state variables
    if 'recording' not in st.session_state:
        st.session_state.recording = False
        st.session_state.audio_buffer = []
        st.session_state.mic_access_granted = False
        st.session_state.default_audio_device = None

    # Store microphone permissions if granted
    def on_mic_access(status):
        st.session_state.mic_access_granted = status
        if status:
            st.rerun()

    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state.recording:
            if st.button("Start Recording"):
                st.session_state.recording = True
                st.session_state.audio_buffer = []
                st.rerun()
    with col2:
        if st.session_state.recording:
            if st.button("Stop Recording"):
                st.session_state.recording = False
                # Save any remaining audio data before stopping
                if processor.accumulated_data and len(processor.accumulated_data) > 0:
                    st.session_state.saved_audio = processor.accumulated_data
                st.rerun()

    # Recording status indicators
    status_col1, status_col2 = st.columns([3, 1])
    with status_col1:
        if st.session_state.recording:
            st.markdown("### 🔴 Recording in progress...")
            if hasattr(processor, 'accumulated_data'):
                # Show buffer fill status
                buffer_progress = len(processor.accumulated_data) / 32000  # Based on our chunk size
                st.progress(min(1.0, buffer_progress), "Buffer")
        else:
            st.markdown("### Click Start Recording to begin")
    
    with status_col2:
        if st.session_state.recording:
            st.metric("Buffer Size", f"{len(processor.accumulated_data)/1000:.1f}k")

    # Create placeholders for audio visualization and transcription
    audio_display = st.empty()
    transcription_container = st.container()
    transcription_text = transcription_container.empty()
    edit_text = transcription_container.empty()

    if 'current_transcription' not in st.session_state:
        st.session_state.current_transcription = ""

    class AudioProcessor:
        def __init__(self):
            self.text_buffer = ""
            self.transcriber = aai.Transcriber()
            self.audio_chunks = []
            self.accumulated_data = bytearray()

        def process(self, frame):
            try:
                audio_data = frame.to_ndarray()
                if audio_data.size > 0:
                    # Update audio visualization
                    amplitude = np.abs(audio_data).mean()
                    audio_display.progress(min(1.0, amplitude * 20))

                    # Accumulate audio data
                    self.accumulated_data += bytearray(audio_data.tobytes())

                    # Every few seconds, transcribe accumulated audio
                    if len(self.accumulated_data) >= 32000:  # Process chunks of ~2 seconds
                        temp_path = Path("temp.wav")
                        with open(temp_path, 'wb') as f:
                            f.write(self.accumulated_data)

                        try:
                            result = self.transcriber.transcribe(filename=str(temp_path))
                            if result.text and result.text != self.text_buffer:
                                self.text_buffer = result.text
                                st.session_state.current_transcription = result.text
                                transcription_text.markdown(f"**Current transcription:**\n{result.text}")
                                st.session_state.candidate_info = result.text
                                st.session_state.saved_audio = self.accumulated_data
                        except Exception as e:
                            st.error(f"Transcription error: {str(e)}")

                        temp_path.unlink()
                        self.accumulated_data = bytearray()
                return frame
            except Exception as e:
                st.error(f"Processing error: {str(e)}")
                return frame

    processor = AudioProcessor()

    def audio_frame_callback(frame):
        audio_data = frame.to_ndarray()
        if audio_data.size > 0:
            processor.accumulated_data += bytearray(audio_data.tobytes())  # Append to audio buffer
            
            # Process accumulated data when enough is collected
            if len(processor.accumulated_data) >= 32000:
                temp_path = Path("temp.wav")
                with open(temp_path, 'wb') as f:
                    f.write(processor.accumulated_data)

                try:
                    result = processor.transcriber.transcribe(filename=str(temp_path))
                    if result.text and result.text != processor.text_buffer:
                        processor.text_buffer = result.text
                        st.session_state.current_transcription = result.text
                        st.session_state.candidate_info = result.text
                        st.session_state.saved_audio = processor.accumulated_data
                except Exception as e:
                    st.error(f"Transcription error: {str(e)}")

                temp_path.unlink()
                processor.accumulated_data = bytearray()
            
            # Update audio visualization
            amplitude = np.abs(audio_data).mean()
            audio_display.progress(min(1.0, amplitude * 20))
            
        return frame

    # Add editable transcription
    if st.session_state.current_transcription:
        edited_text = edit_text.text_area(
            "Edit transcription if needed:",
            value=st.session_state.current_transcription,
            key="transcript_editor"
        )
        if edited_text != st.session_state.current_transcription:
            st.session_state.candidate_info = edited_text

    webrtc_ctx = webrtc_streamer(
        key="audio-recorder",
        mode="sendonly",
        audio_frame_callback=audio_frame_callback,
        media_stream_constraints={"video": False, "audio": True},
        async_processing=True,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )

    if 'candidate_info' in st.session_state:
        candidate_info = st.session_state.candidate_info

    # Add download options for audio and transcript
    if st.session_state.get('saved_audio'):
        try:
            # Ensure we have valid audio data
            audio_data = bytes(st.session_state.saved_audio) if isinstance(st.session_state.saved_audio, bytearray) else st.session_state.saved_audio
            
            if audio_data:
                # Create audio player
                st.write("### 🎵 Recorded Audio")
                st.audio(audio_data, format='audio/wav')

                # Save audio file for download
                audio_path = Path("recorded_audio.wav")
                audio_path.write_bytes(audio_data)  # More reliable way to write binary data

                # Create download button
                with open(audio_path, 'rb') as f:
                    audio_bytes = f.read()
                    st.download_button(
                        "💾 Download Audio",
                        audio_bytes,
                        file_name="recorded_audio.wav",
                        mime="audio/wav",
                        help="Click to download the recorded audio"
                    )
        except Exception as e:
            st.error(f"Error saving audio: {str(e)}")

    if st.session_state.get('final_transcript'):
        st.download_button(
            "Download Transcript",
            st.session_state.final_transcript,
            file_name="transcript.txt",
            mime="text/plain"
        )

    # Add transcript summary using OpenAI GPT
    if st.session_state.get('current_transcription'):
        if st.button("Generate Summary"):
            summary_response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "Summarize this candidate's job profile."},
                    {"role": "user", "content": st.session_state.current_transcription}
                ]
            )
            st.write("### Summary")
            st.write(summary_response.choices[0].message.content)

    if candidate_info and st.button("Generate Profile"):
        generate_job_profile(candidate_info)