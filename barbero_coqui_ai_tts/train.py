import pathlib

import click
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# from TTS.tts.datasets.tokenizer import Tokenizer


@click.command()
@click.argument("output_path")
def main(output_path):
    # init configs
    dataset_config = BaseDatasetConfig(
        name="coqui",
        meta_file_train="audio_metadata.csv",
        path=output_path,
    )

    audio_config = BaseAudioConfig(
        sample_rate=16000,
        do_trim_silence=True,
        trim_db=60.0,
        signal_norm=False,
        mel_fmin=50,
        mel_fmax=8000,
        spec_gain=1.0,
        log_func="np.log",
        ref_level_db=20,
        preemphasis=0.0,
    )

    config = Tacotron2Config(
        audio=audio_config,
        batch_size=64,
        eval_batch_size=16,
        eval_split_size=0.05,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        r=6,
        gradual_training=[[0, 6, 6], [10000, 4, 5], [50000, 3, 5], [100000, 2, 5]],
        double_decoder_consistency=True,
        epochs=1000,
        text_cleaner="multilingual_cleaners",
        use_phonemes=True,
        phoneme_language="it-it",
        phoneme_cache_path=pathlib.Path(output_path) / "phoneme_cache",
        precompute_num_workers=8,
        print_step=25,
        print_eval=True,
        mixed_precision=False,
        output_path=output_path,
        datasets=[dataset_config],
    )

    # INITIALIZE THE AUDIO PROCESSOR
    # Audio processor is used for feature extraction and audio I/O.
    # It mainly serves to the dataloader and the training loggers.
    ap = AudioProcessor.init_from_config(config)

    # INITIALIZE THE TOKENIZER
    # Tokenizer is used to convert text to sequences of token IDs.
    # If characters are not defined in the config, default characters are
    # passed to the config
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

    # INITIALIZE THE MODEL
    # Models take a config object and a speaker manager as input
    # Config defines the details of the model like the number of layers, the
    # size of the embedding, etc.
    # Speaker manager is used by multi-speaker models.
    model = Tacotron2(config, ap, tokenizer)

    # INITIALIZE THE TRAINER
    # Trainer provides a generic API to train all the 🐸TTS models with all its
    # perks like mixed-precision training,
    # distributed training, etc.
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    # AND... 3,2,1... 🚀
    trainer.fit()


if __name__ == "__main__":
    main()
