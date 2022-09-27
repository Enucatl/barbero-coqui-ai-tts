# barbero-coqui-ai-tts
Train a TTS neural network for Alessandro Barbero

```bash
ln -s $(realpath data/audio_metadata.csv) /hd4tb/archivio_barbero/splits/
python barbero_coqui_ai_tts/create_data.py data/original_files.csv data/pipeline_metadata.csv
python barbero_coqui_ai_tts/convert_to_flac.py data/pipeline_metadata.csv 
GOOGLE_APPLICATION_CREDENTIALS=secrets/barbero-translations-0ea9c0d3c73f.json python barbero_coqui_ai_tts/upload_flac_to_gcs.py data/pipeline_metadata.csv 
GOOGLE_APPLICATION_CREDENTIALS=secrets/barbero-translations-0ea9c0d3c73f.json python barbero_coqui_ai_tts/speech_to_text.py data/pipeline_metadata.csv 
GOOGLE_APPLICATION_CREDENTIALS=secrets/barbero-translations-0ea9c0d3c73f.json python barbero_coqui_ai_tts/download_transcripts.py data/pipeline_metadata.csv 
python barbero_coqui_ai_tts/split_sentences.py data/pipeline_metadata.csv  data/audio_metadata.csv
CUDA_VISIBLE_DEVICES=0 python barbero_coqui_ai_tts/train.py
```
