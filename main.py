
import streamlit as st
import openai
import assemblyai as aai
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
else:
    st.write("Click start to begin recording")
    
    if 'audio_buffer' not in st.session_state:
        st.session_state.audio_buffer = []
    
    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
    
    start = st.button("Start Recording")
    stop = st.button("Stop Recording")
    
    from streamlit_webrtc import webrtc_streamer
    import av
    
    def audio_frame_callback(frame):
        if frame.to_ndarray().size > 0:
            temp_path = Path("temp.wav")
            frame.to_ndarray().tofile(str(temp_path))
            
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(str(temp_path))
            
            if transcript.text:
                st.write("Transcription:", transcript.text)
                st.session_state.candidate_info = transcript.text
                
            temp_path.unlink()
            
    webrtc_streamer(
        key="audio-recorder",
        audio_frame_callback=audio_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True}
    )
    
    if 'candidate_info' in st.session_state:
        candidate_info = st.session_state.candidate_info

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
