import os
import argparse
import logging
from io import BytesIO
from dotenv import load_dotenv

import whisper
import requests
from pydub import AudioSegment
from openai import OpenAI

load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
client = OpenAI(api_key=OPENAI_API_KEY)
previous_request_ids = []


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio and generate TTS.")
    parser.add_argument(
        "input_audio_path",
        type=str,
        nargs="?",
        default="trimmed_english.mp4",
        help="Path to the input audio file",
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
    response = client.responses.create(
        model="gpt-4o-mini",  # Use GPT-4 or another version if you have access
        instructions=f"You are a helpful translation assistant. Translate everything to {target_language}.",
        input=text,
    )
    translated_text = response.output_text.strip()
    logging.info(f"Translated text: {translated_text}")
    return translated_text


def synthesize_speech(text: str) -> AudioSegment:
    logging.info("Synthesizing speech using Eleven Labs TTS API...")
    voice_id = "EXAVITQu4vr4xnSDxMaL"  # TODO: Make this configurable
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY,
    }

    payload = {
        "text": text,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        "previous_request_ids": previous_request_ids[-3:],
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


def main():
    args = parse_arguments()
    input_audio_path = args.input_audio_path
    output_audio_path = f"{os.path.splitext(input_audio_path)[0]}_output.mp3"

    model = load_whisper_model()
    segments = transcribe_audio(model, input_audio_path)

    if not segments:
        logging.warning("No segments found in the transcription. Exiting.")
        return

    final_audio = AudioSegment.silent(duration=len(AudioSegment.from_file(input_audio_path, format="mp4")))
    target_language = "Spanish"

    # Generate audio for each segment
    for segment in segments:
        logging.info(f"Processing segment: {segment['text']}")
        translated_text = translate_text(segment["text"], target_language)
        tts_audio = synthesize_speech(translated_text)
        start_ms = int(
            segment["start"] * 1000
        )  # Convert start time from seconds to milliseconds
        final_audio = final_audio.overlay(tts_audio, position=start_ms)

    # Export final audio
    save_audio(final_audio, output_audio_path)

    logging.info(f"Process completed. Output saved to {output_audio_path}")


if __name__ == "__main__":
    main()
