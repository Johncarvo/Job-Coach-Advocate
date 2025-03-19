
import streamlit as st
import openai
import assemblyai as aai
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

st.title("AI Job Profile Generator")

# File upload
audio_file = st.file_uploader("Upload audio file", type=['mp3', 'wav'])

if audio_file:
    # Save uploaded file temporarily
    temp_path = Path("temp.wav")
    with open(temp_path, "wb") as f:
        f.write(audio_file.getbuffer())
    
    # Transcribe with AssemblyAI
    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(str(temp_path))
    
    if transcript.text:
        st.write("Transcription:", transcript.text)
        
        if st.button("Generate Profile"):
            client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a professional job profile writer."},
                    {"role": "user", "content": f"Create a structured job profile from this experience: {transcript.text}"}
                ]
            )
            
            st.write("Generated Profile:")
            st.write(response.choices[0].message.content)

    # Cleanup
    temp_path.unlink()
