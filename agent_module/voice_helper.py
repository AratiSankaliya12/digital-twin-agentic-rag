import os
import re
from openai import OpenAI

def get_openai_client():
    """
    Initializes and returns the OpenAI client.
    Assumes config.py has loaded the environment variables.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback to check if we can import config to load it
        try:
            import config
            api_key = os.getenv("OPENAI_API_KEY")
        except Exception:
            pass
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment. Please set it in your .env file.")
    
    return OpenAI(api_key=api_key)

def transcribe_audio(audio_file) -> str:
    """
    Transcribes an audio file object (e.g. from st.audio_input) using OpenAI Whisper API.
    """
    client = get_openai_client()
    
    # Reset the file pointer to the beginning before reading
    audio_file.seek(0)
    
    # Prepare the file tuple for the API call: (filename, bytes, content_type)
    file_name = getattr(audio_file, "name", "audio.wav") or "audio.wav"
    file_type = getattr(audio_file, "type", "audio/wav") or "audio/wav"
    file_bytes = audio_file.read()
    
    file_tuple = (file_name, file_bytes, file_type)
    
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=file_tuple
    )
    
    return response.text.strip()

def generate_speech(text: str, voice: str = "nova", rate: str = "+0%", pitch: str = "+0Hz") -> bytes:
    """
    Generates speech audio bytes from text using edge-tts (premium Microsoft neural voices).
    Provides natural-sounding human voices for free with pitch and speed controls.
    """
    voice_map = {
        "nova": "en-US-AvaNeural",
        "shimmer": "en-US-EmmaNeural",
        "alloy": "en-US-JennyNeural",
        "echo": "en-US-BrianNeural"
    }
    
    edge_voice = voice_map.get(voice, voice)
    
    import asyncio
    import edge_tts
    
    async def _generate():
        communicate = edge_tts.Communicate(text, edge_voice, rate=rate, pitch=pitch)
        data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                data += chunk["data"]
        return data
        
    try:
        return asyncio.run(_generate())
    except Exception as e:
        print(f"Edge-TTS failed: {e}. Falling back to gTTS...")
        try:
            from gtts import gTTS
            import io
            tts = gTTS(text=text, lang="en")
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            return fp.read()
        except Exception as fallback_err:
            raise RuntimeError(f"Voice generation failed completely. Edge-TTS: {e}, gTTS: {fallback_err}")

def clean_text_for_speech(text: str) -> str:
    """
    Removes markdown elements, URLs, and code blocks from text to make it sound natural when spoken.
    """
    if not text:
        return ""
        
    # Remove code blocks
    text = re.sub(r"```.*?```", "[Code block omitted]", text, flags=re.DOTALL)
    
    # Remove inline code (backticks)
    text = re.sub(r"`.*?`", "", text)
    
    # Remove markdown bold/italic asterisks and underscores
    text = re.sub(r"\*\*|__|\*|_", "", text)
    
    # Remove markdown links, keeping only the text: [Link text](URL) -> Link text
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    
    # Remove headers: # Header -> Header
    text = re.sub(r"#+\s+", "", text)
    
    # Remove bullet markers at the beginning of lines
    text = re.sub(r"^\s*[\*\+\-\d\.]+\s+", "", text, flags=re.MULTILINE)
    
    # Clean up double spaces and newlines
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()
