# F5-TTS-Finetuning
This project fine-tunes the F5-TTS model to add a new language, using a pre-trained base to enhance multilingual capabilities.

This tutorial will guide you through the fine-tuning process. To access the official F5-TTS repository, visit: https://github.com/SWivid/F5-TTS.

> These steps were written for **F5-TTS v0.3.4** (commit `3e73553`). Newer releases renamed several training options, so check out that version as shown below.

# Installation
```
# create a python 3.10 env
conda create --name f5-tts python=3.10
conda activate f5-tts
```
## Install PyTorch with your CUDA version
```
pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```
## Install PyTorch (CPU Version) on Mac
```
pip install torch==2.3.0 torchaudio==2.3.0
```
## Clone the original repository and install the required dependencies
```
git clone https://github.com/SWivid/F5-TTS.git
cd F5-TTS
git checkout 3e73553   # v0.3.4
pip install -e .
```
## Install the dependencies for `wav_and_metadata.py`
```
pip install pydub SpeechRecognition
# pydub needs ffmpeg to read MP3 files
brew install ffmpeg        # macOS
# sudo apt install ffmpeg  # Ubuntu
```
## Setting up your dataset
1. Create a directory for your data within the F5-TTS/data folder. For example: `F5-TTS/data/new_language`.
2. Inside your new language directory, create a folder named `wavs` to store your audio files. For example: `F5-TTS/data/new_language/wavs`.

At this point, your file structure should look like this:
```
/F5-TTS
|-- data/
|   |-- new_language/
|   |   |-- wavs/
```
3. Use `wav_and_metadata.py` to convert your MP3 files to WAV format and generate a `metadata.csv` file.

* Set `mp3_dir` (your source MP3 files), `wav_dir` (the `wavs` directory you created), and `metadata_file` (where `metadata.csv` should be written) at the top of the script, then run:
```
python wav_and_metadata.py
```
* The script transcribes each clip with Google's speech recognition (`es-MX`, internet required) and skips clips it can't transcribe. Your `metadata.csv` file should look like this:
```
audio_file|text
wavs/audio_0001.wav|tenían prohibida por ley la presencia en teatros y aglomeraciones públicas
wavs/audio_0002.wav|Cómo operan el camarote claro que no nada de eso
```
The first line is a header that F5-TTS skips, and each path is relative to the folder that holds `metadata.csv`. Review the transcriptions before training; the model learns whatever the text says.

Once your `metadata.csv` file is created and your MP3 files are converted to WAV, your file structure should look like this:
```
/F5-TTS
|-- data/
|   |-- new_language/
|   |   |-- metadata.csv
|   |   |-- wavs/
|   |   |   |-- audio_0001.wav
|   |   |   |-- audio_0002.wav
|   |   |   |-- ...
```
4. Prepare your metadata and WAV files by running the `prepare_csv_wavs.py` script.

The script takes an input directory (where `metadata.csv` and `wavs/` are) and an output directory. F5-TTS looks for training data in `data/<dataset_name>_<tokenizer>`, so name the output folder `new_language_pinyin`. From the `F5-TTS` folder, run:
```
python src/f5_tts/train/datasets/prepare_csv_wavs.py data/new_language data/new_language_pinyin
```
Three new files should have been created in `data/new_language_pinyin`: `raw.arrow`, `duration.json`, and `vocab.txt` (for fine-tuning, `vocab.txt` is copied from the pre-trained model so the vocabularies match).
If the `duration.json` file is empty, it likely means the input path was not set correctly. At this point, the file structure should look like this:
```
/F5-TTS
|-- data/
|   |-- new_language/
|   |   |-- metadata.csv
|   |   |-- wavs/
|   |-- new_language_pinyin/
|   |   |-- raw.arrow
|   |   |-- duration.json
|   |   |-- vocab.txt
```
## Proceed to Training
1. Download the pre-trained base model.

Create a directory named `F5TTS_Base` inside your `ckpts` directory and download the base model into it:
```
mkdir -p ckpts/F5TTS_Base
curl -L -o ckpts/F5TTS_Base/model_1200000.pt "https://huggingface.co/SWivid/F5-TTS/resolve/main/F5TTS_Base/model_1200000.pt?download=true"
```
(If you skip this step and leave out `--pretrain`, the training script downloads the same checkpoint automatically.)

Make sure the directory structure matches the expected format before moving on.
```
/F5-TTS
|-- ckpts/
|   |-- F5TTS_Base/
|   |   |-- model_1200000.pt
|-- data/
|   |-- new_language/
|   |-- new_language_pinyin/
|   |   |-- raw.arrow
|   |   |-- duration.json
|   |   |-- vocab.txt
```
2. Start training

`--dataset_name` must match the data folder without the tokenizer suffix (`new_language` for `data/new_language_pinyin`). The `pinyin` tokenizer only converts Chinese characters, so Spanish text, including accented letters, is kept as-is and matches the pre-trained vocabulary. With `--batch_size_type frame`, `--batch_size_per_gpu` counts mel frames, not clips (3200 frames is about 34 seconds of audio). These settings are a small test run; raise `--epochs` for a real dataset.
```
accelerate launch src/f5_tts/train/finetune_cli.py \
    --exp_name F5TTS_Base \
    --dataset_name new_language \
    --tokenizer pinyin \
    --finetune True \
    --pretrain ckpts/F5TTS_Base/model_1200000.pt \
    --learning_rate 1e-05 \
    --batch_size_per_gpu 3200 \
    --batch_size_type frame \
    --max_samples 25 \
    --grad_accumulation_steps 1 \
    --max_grad_norm 1 \
    --epochs 5 \
    --num_warmup_updates 2 \
    --save_per_updates 20 \
    --last_per_steps 10 \
    --logger tensorboard
```
Checkpoints are saved to `ckpts/new_language/`.

If you get a division by zero error, no batches were created: with `--batch_size_type frame`, every clip longer than `--batch_size_per_gpu` frames is skipped. Raise the frame count, or use `--batch_size_type sample --batch_size_per_gpu 1`. If you run out of GPU memory, lower `--batch_size_per_gpu` or `--max_samples`.
