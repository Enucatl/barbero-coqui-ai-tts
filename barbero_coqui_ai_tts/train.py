import pathlib

import click

# Trainer: Where the ✨️ happens.
# TrainingArgs: Defines the set of arguments of the Trainer.
from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


@click.command()
@click.option("--output_path")
@click.option("--restore_path", default=None)
def main(output_path, restore_path):
    output_path = pathlib.Path(output_path)

    # DEFINE DATASET CONFIG
    # You can also use a simple Dict to define the dataset and pass it to your custom formatter.
    dataset_config = BaseDatasetConfig(
        formatter="coqui", meta_file_train="audio_metadata.csv", path=output_path
    )

    audio_config = VitsAudioConfig(
        sample_rate=16000, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
    )

    config = VitsConfig(
        audio=audio_config,
        batch_group_size=5,
        batch_size=16,
        compute_input_seq_cache=True,
        cudnn_benchmark=False,
        datasets=[dataset_config],
        epochs=1000,
        eval_batch_size=8,
        mixed_precision=True,
        num_eval_loader_workers=4,
        num_loader_workers=4,
        output_path=output_path,
        phoneme_cache_path=output_path / "phoneme_cache",
        phoneme_language="it-it",
        print_eval=True,
        print_step=25,
        run_eval=True,
        run_name="vits_coqui",
        test_delay_epochs=-1,
        text_cleaner="phoneme_cleaners",
        use_phonemes=True,
    )

    # INITIALIZE THE AUDIO PROCESSOR
    # Audio processor is used for feature extraction and audio I/O.
    # It mainly serves to the dataloader and the training loggers.
    ap = AudioProcessor.init_from_config(config)

    # INITIALIZE THE TOKENIZER
    # Tokenizer is used to convert text to sequences of token IDs.
    # config is updated with the default characters if not defined in the config.
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # LOAD DATA SAMPLES
    # Each sample is a list of ```[text, audio_file_path, speaker_name]```
    # You can define your custom sample loader returning the list of samples.
    # Or define your custom formatter and pass it to the `load_tts_samples`.
    # Check `TTS.tts.datasets.load_tts_samples` for more details.
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init model
    model = Vits(config, ap, tokenizer, speaker_manager=None)

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(restore_path=restore_path),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
