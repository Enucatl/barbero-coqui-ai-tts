# barbero-coqui-ai-tts
Train a TTS neural network for Alessandro Barbero

```bash
CUDA_VISIBLE_DEVICES=0 \
    python barbero_coqui_ai_tts/train.py \
    --output_path /ssd500gb/archivio_barbero/split_wav/ \
    --restore_path /ssd500gb/archivio_barbero/tts_models--it--mai_male--vits/model_file.pth
```
