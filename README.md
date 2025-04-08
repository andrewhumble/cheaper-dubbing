# Cheaper Dubbing with Whisper + ElevenLabs TTS

ElevenLabs continues to pioneer the world of text-to-speech, but it comes at a cost. For dubbing, this price quickly becomes a barrier for many use cases at a cost of 3000 credits/minute. For ElevenLab's "Starter" plan, this means users have only 10 minutes of dubbing capability.

This project aims to resolve this issue by performing transcription, translation, and alignment manually, followed by the usage of ElevenLabs' text-to-speech API (which is much cheaper than their dubbing feature) to create the final audio. With this approach, dubbing generations are achieved at a cost of 761 credits/minute, translating to **75% cheaper** credit consumption and **3.94x capacity** for "Starter" dubbing usage. 

## Requirements
- Required Python packages:
  - `whisper`
  - `requests`
  - `pydub`
  - `openai`

You can install the required packages using pip:
```bash
pip install whisper requests pydub openai
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ahumble/cheaper-dubbing.git
   cd cheaper-dubbing
   ```
2. Install the required packages as mentioned in the Requirements section.
3. Copy the `.env.example` file to create your own `.env` file:
   ```bash
   cp .env.example .env
   ```  

   Open the `.env` file and add your API keys as follows:
   ```env
   ELEVENLABS_API_KEY=your_elevenlabs_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```  

   Replace `your_elevenlabs_api_key` and `your_openai_api_key` with your actual API keys. You can create these keys from https://elevenlabs.io/app/settings/api-keys and https://platform.openai.com/api-keys.

## Usage
To use this script, run the following command in your terminal:
```bash
python main.py <path_to_audio_file>
```
- Replace `<path_to_audio_file>` with the path to your input audio file. If no path is provided, it defaults to `trimmed_english.mp4`.
- The output will be saved as `<input_audio_file_name>_output.mp3` in the same directory.

### Arguments
The script accepts the following optional arguments:

- `--target_language`: Specify the target language for translation (default is "Spanish").
- `--keep_background`: Use this flag to keep the background audio while removing vocals (default is False).
- `--output_dir`: Specify the directory to save output files (defaults to the same directory as the input).
- `--use_voice_clone`: Use this flag to enable the voice cloning feature (default is False).

### Example
```bash
python main.py my_audio.mp4
```

This will process `my_audio.mp4` and save the output as `my_audio_output.mp3`.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.