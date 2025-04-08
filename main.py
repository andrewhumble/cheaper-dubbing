import os
import argparse
import logging
import subprocess
from io import BytesIO
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip, AudioFileClip

import whisper
import requests
from pydub import AudioSegment
from openai import OpenAI
from elevenlabs import ElevenLabs

load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
openai_client = OpenAI(api_key=OPENAI_API_KEY)
el_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
previous_request_ids = []
input_video_path = ""
base_name = ""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio, translate, and apply to video.")
    parser.add_argument(
        "input_video_path",
        type=str,
        nargs="?",
        default="trimmed_english.mp4",
        help="Path to the input video file",
    )
    parser.add_argument(
        "--target_language",
        type=str,
        default="Spanish",
        help="Target language for translation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save output files (defaults to same as input)",
    )
    parser.add_argument(
        "--use_voice_clone",
        action='store_true',
        help="Use voice cloning feature (default is False)",
    )
    return parser.parse_args()


def load_whisper_model() -> whisper.Whisper:
    model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
    logging.info(f"Loading Whisper model '{model_size}'...")
    return whisper.load_model(model_size)


def transcribe_audio(model: whisper.Whisper, audio_path: str) -> list:
    logging.info(f"Transcribing audio: {audio_path}")
    result = model.transcribe(audio_path, word_timestamps=True)
    segments = result.get("segments", [])
    logging.info(f"Found {len(segments)} segments.")
    return segments


def translate_text(text: str, target_language: str) -> str:
    logging.info(f"Translating text to {target_language}...")
    response = openai_client.responses.create(
        model="gpt-4o-mini",
        instructions=f"You are a helpful translation assistant. Translate everything to {target_language}.",
        input=text,
    )
    translated_text = response.output_text.strip()
    logging.info(f"Translated text: {translated_text}")
    return translated_text


def get_cloned_voice_id():
    """
    Get or create a cloned voice ID from Eleven Labs API
    
    Args:
        input_video_path: Path to the audio file for voice cloning
        base_name: Base name to use for the cloned voice
    
    Returns:
        str: The voice ID of the existing or newly created voice clone
    """
    logging.info(f"Getting cloned voice ID for {base_name}")
    voice_name = f"{base_name}-voice-clone"
    
    # First check if the voice already exists
    url = "https://api.elevenlabs.io/v2/voices"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    
    try:
        retrieve_voices_response = el_client.voices.search(
            search=voice_name
        )
        
        # Check if voice with this name already exists
        for voice in retrieve_voices_response.voices:
            if voice.get("name") == voice_name:
                voice_id = voice.get("voice_id")
                logging.info(f"Found existing voice clone with ID: {voice_id}")
                return voice_id
                
        # If no matching voice found, create a new one
        logging.info(f"No existing voice clone found. Creating new voice clone: {voice_name}")
        
        # Open the audio file for cloning
        with open(input_video_path, "rb") as file:            
            clone_response = el_client.voices.add(
                name=voice_name,
                files=[input_video_path]
            )
            
            # Extract the voice ID from the response
            voice_id = clone_response.json().get("voice_id")
            logging.info(f"Successfully created new voice clone with ID: {voice_id}")
            return voice_id
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Error communicating with Eleven Labs API: {str(e)}")
        if hasattr(e, 'response') and e.response:
            logging.error(f"Response content: {e.response.text}")
        raise

def synthesize_speech(text: str, voice_id: str) -> AudioSegment:
    logging.info("Synthesizing speech using Eleven Labs TTS API...")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY,
    }

    payload = {
        "text": text,
        "voice_settings": {"speed": 0.7},
        "previous_request_ids": previous_request_ids[-3:],
        "model_id": "eleven_multilingual_v2",
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    logging.info("Speech synthesis successful.")
    previous_request_ids.append(response.headers["request-id"])
    return AudioSegment.from_file(BytesIO(response.content), format="mp3")


def save_audio(audio: AudioSegment, output_path: str) -> None:
    logging.info(f"Saving audio to {output_path}")
    audio.export(output_path, format="mp3")
    logging.info("Audio saved successfully.")


def extract_audio_from_video(video_path: str, output_audio_path: str) -> None:
    """Extract audio from video using FFmpeg"""
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-q:a", "0",           # Best quality
        "-map", "a",           # Extract audio only
        "-y",                  # Overwrite output file
        output_audio_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        logging.info(f"Audio extracted from video to {output_audio_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error extracting audio: {e.stderr.decode() if e.stderr else str(e)}")
        raise


def remove_vocals_from_audio(input_path: str, output_path: str) -> None:
    """
    Remove voice audio from input file while keeping other audio like music and background sounds.
    Uses FFmpeg's center channel removal technique which works well for many professionally mixed videos.
    
    Args:
        input_path: Path to input audio/video file
        output_path: Path to save the processed audio (without vocals)
    """
    logging.info(f"Removing vocals from {input_path}")
    
    # Use FFmpeg's center channel vocal removal technique
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-af", "pan=stereo|c0=c0-c1|c1=c1-c0,highpass=f=200,lowpass=f=3000",  # Vocal removal filter
        "-b:a", "192k",
        "-y",
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        logging.info(f"Vocals removed successfully. Output saved to {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error removing vocals: {e.stderr.decode() if e.stderr else str(e)}")
        raise


def mix_audio_files(background_audio_path: str, translated_audio_path: str, output_path: str) -> None:
    """Mix background audio with translated audio using FFmpeg"""
    logging.info(f"Mixing background audio with translated speech")
    
    cmd = [
        "ffmpeg",
        "-i", background_audio_path,     # Background audio input
        "-i", translated_audio_path,     # Translated speech input
        "-filter_complex", "[0:a]volume=0.5[bg];[1:a]volume=1.0[fg];[bg][fg]amix=inputs=2:duration=longest",  # Mix with adjusted volumes
        "-b:a", "192k",
        "-y",
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        logging.info(f"Audio mixing completed successfully. Output saved to {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error mixing audio: {e.stderr.decode() if e.stderr else str(e)}")
        raise


def apply_audio_to_video(video_path: str, audio_path: str, output_path: str) -> None:
    """Replace the audio in a video file with a new audio file using MoviePy"""
    logging.info(f"Replacing audio in {video_path} with {audio_path}")
    
    try:
        # Load the video and audio
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        
        # If audio is longer than video, trim it
        if audio.duration > video.duration:
            audio = audio.subclip(0, video.duration)
        
        # Apply the new audio to the video
        video_with_new_audio = video.set_audio(audio)
        
        # Write the final video file
        video_with_new_audio.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True
        )
        
        # Close the clips to free resources
        video.close()
        audio.close()
        video_with_new_audio.close()
        
        logging.info(f"Video with replaced audio created successfully: {output_path}")
    except Exception as e:
        logging.error(f"Error applying audio to video: {str(e)}")
        raise


def main():
    args = parse_arguments()
    global input_video_path
    input_video_path = args.input_video_path
    
    # Determine directory to save files
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(input_video_path) or "."
    
    target_language = args.target_language

    # Create file paths
    filename = os.path.basename(input_video_path)
    global base_name
    base_name = os.path.splitext(filename)[0]
    temp_original_audio_path = os.path.join(output_dir, f"{base_name}_original_audio.wav")
    temp_no_vocals_path = os.path.join(output_dir, f"{base_name}_no_vocals.mp3")
    temp_translated_audio_path = os.path.join(output_dir, f"{base_name}_translated_speech.mp3")
    temp_mixed_audio_path = os.path.join(output_dir, f"{base_name}_mixed_audio.mp3")
    final_audio_path = os.path.join(output_dir, f"{base_name}_output.mp3")
    output_video_path = os.path.join(output_dir, f"{base_name}_{target_language}_dub.mp4")
    
    # Define voice_id
    voice_id = get_cloned_voice_id() if args.use_voice_clone else "29vD33N1CtxCmqQRPOHJ"

    # Extract audio from video for processing
    extract_audio_from_video(input_video_path, temp_original_audio_path)

    # Load Whisper model and transcribe audio
    model = load_whisper_model()
    segments = transcribe_audio(model, temp_original_audio_path)

    if not segments:
        logging.warning("No segments found in the transcription. Exiting.")
        return

    # Get audio duration for silent base
    audio_segment = AudioSegment.from_file(temp_original_audio_path)
    duration_ms = len(audio_segment)
    
    # Create silent base for translated audio
    final_audio = AudioSegment.silent(duration=duration_ms)
    
    # Generate translated speech for each segment
    for segment in segments:
        logging.info(f"Processing segment: {segment['text']}")
        translated_text = translate_text(segment["text"], target_language)
        tts_audio = synthesize_speech(translated_text, voice_id)
        start_ms = int(segment["start"] * 1000)  # Convert start time to milliseconds
        final_audio = final_audio.overlay(tts_audio, position=start_ms)

    # Save translated speech audio
    save_audio(final_audio, temp_translated_audio_path)
    
    # Remove vocals but keep background music
    remove_vocals_from_audio(temp_original_audio_path, temp_no_vocals_path)

    # Mix translated speech with background music
    mix_audio_files(temp_no_vocals_path, temp_translated_audio_path, final_audio_path)
    
    # Apply the final audio to the original video
    apply_audio_to_video(input_video_path, final_audio_path, output_video_path)
    
    logging.info(f"Process completed. Output video saved to {output_video_path}")
    
    # Clean up temporary files
    temp_files = [
        temp_original_audio_path,
        temp_no_vocals_path,
        temp_translated_audio_path,
        temp_mixed_audio_path
    ]
    
    for file_path in temp_files:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.debug(f"Removed temporary file: {file_path}")
            except OSError as e:
                logging.warning(f"Could not remove temporary file {file_path}: {e}")


if __name__ == "__main__":
    main()
