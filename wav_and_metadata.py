import os
import csv
from pydub import AudioSegment
import speech_recognition as sr

# Set paths
mp3_dir = ""       # Directory containing MP3 files
wav_dir = ""       # Directory to save converted WAV files (e.g. data/new_language/wavs)
metadata_file = "" # Path of the metadata.csv to write (e.g. data/new_language/metadata.csv)
file_limit = 50       # Set the maximum number of files to process

if not (mp3_dir and wav_dir and metadata_file):
    raise SystemExit("Set mp3_dir, wav_dir, and metadata_file at the top of this script first.")

# Ensure WAV output directory exists
os.makedirs(wav_dir, exist_ok=True)

# Paths in metadata.csv are relative to the folder that holds it (F5-TTS resolves them that way)
metadata_dir = os.path.dirname(os.path.abspath(metadata_file))

# Initialize the recognizer for transcription
recognizer = sr.Recognizer()

mp3_files = sorted(f for f in os.listdir(mp3_dir) if f.lower().endswith(".mp3"))[:file_limit]

# Open CSV file for writing
with open(metadata_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file, delimiter="|")
    writer.writerow(["audio_file", "text"])  # F5-TTS skips the first row as a header

    # Loop through MP3 files and process up to the file limit
    for i, mp3_filename in enumerate(mp3_files, start=1):
        # Define paths
        mp3_path = os.path.join(mp3_dir, mp3_filename)
        wav_filename = f"audio_{i:04d}.wav"
        wav_path = os.path.join(wav_dir, wav_filename)

        # Convert MP3 to WAV
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")

        # Transcribe the WAV file
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data, language="es-MX")
            except sr.UnknownValueError:
                print(f"Skipped {mp3_filename}: speech was unintelligible")
                os.remove(wav_path)
                continue
            except sr.RequestError as err:
                print(f"Skipped {mp3_filename}: API request error ({err})")
                os.remove(wav_path)
                continue

        # Write metadata entry
        writer.writerow([os.path.relpath(wav_path, metadata_dir), text])
        print(f"Processed {wav_filename}: {text}")
