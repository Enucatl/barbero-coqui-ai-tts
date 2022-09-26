# barbero-coqui-ai-tts
Train a TTS neural network for Alessandro Barbero

```bash
python barbero_coqui_ai_tts/convert_to_flac.py data/original_files.csv 
GOOGLE_APPLICATION_CREDENTIALS=secrets/barbero-translations-0ea9c0d3c73f.json python barbero_coqui_ai_tts/upload_flac_to_gcs.py data/original_files.csv 
GOOGLE_APPLICATION_CREDENTIALS=secrets/barbero-translations-0ea9c0d3c73f.json python barbero_coqui_ai_tts/speech_to_text.py data/original_files.csv 
```
