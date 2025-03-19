import streamlit as st
import openai
import assemblyai as aai
import numpy as np
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

st.title("AI Job Profile Generator")

# Input method selection
input_method = st.radio("Choose input method:", ["Text", "Audio"])

candidate_info = None

if input_method == "Text":
    candidate_info = st.text_area("Enter candidate information:", height=200)
    if candidate_info and st.button("Generate Profile"):
        client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
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
                st.rerun()

    if st.session_state.recording:
        st.write("🔴 Recording in progress...")
    else:
        st.write("Click Start Recording to begin")

    from streamlit_webrtc import webrtc_streamer
    import av

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
            config = aai.TranscriptionConfig(
                speaker_labels=True,
                speech_model=aai.SpeechModel.nano
            )
            self.transcriber = aai.Transcriber(config=config)
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
                    self.accumulated_data.extend(audio_data.tobytes())

                    # Every few seconds, transcribe accumulated audio
                    if len(self.accumulated_data) >= 32000:  # Process chunks of ~2 seconds
                        temp_path = Path("temp.wav")
                        with open(temp_path, 'wb') as f:
                            f.write(self.accumulated_data)

                        try:
                            result = self.transcriber.transcribe(str(temp_path))
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
        return processor.process(frame)

    # Add editable transcription
    if st.session_state.current_transcription:
        edited_text = edit_text.text_area(
            "Edit transcription if needed:",
            value=st.session_state.current_transcription,
            key="transcript_editor"
        )
        if edited_text != st.session_state.current_transcription:
            st.session_state.candidate_info = edited_text

    webrtc_streamer(
        key="audio-recorder",
        audio_frame_callback=audio_frame_callback,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]}
            ]
        },
        media_stream_constraints={"video": False, "audio": True},
        async_processing=True,
        sendback_audio=False
    )

    if 'candidate_info' in st.session_state:
        candidate_info = st.session_state.candidate_info

# Add download options for audio and transcript
    if st.session_state.get('saved_audio') is not None:
        st.audio(st.session_state.saved_audio, format='audio/wav')

        # Save audio file for download
        audio_path = Path("recorded_audio.wav")
        with open(audio_path, 'wb') as f:
            st.session_state.saved_audio.tofile(f)

        with open(audio_path, 'rb') as f:
            st.download_button(
                "Download Audio",
                f,
                file_name="recorded_audio.wav",
                mime="audio/wav"
            )

    if st.session_state.get('final_transcript'):
        st.download_button(
            "Download Transcript",
            st.session_state.final_transcript,
            file_name="transcript.txt",
            mime="text/plain"
        )

    # Add transcript summary using LLM
    if st.session_state.get('current_transcription'):
        if st.button("Generate Summary"):
            prompt = "Provide a brief professional summary of the candidate based on this transcript."
            result = processor.transcriber.lemur.task(
                prompt, 
                final_model=aai.LemurModel.claude3_5_sonnet
            )
            st.write("### Summary")
            st.write(result.response)

    if candidate_info and st.button("Generate Profile"):
            client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

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