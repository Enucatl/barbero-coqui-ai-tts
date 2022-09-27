import logging
import pathlib

import pandas as pd
import click


logger = logging.getLogger(__name__)


def calculate_paths(x, bucket_name):
    original_audio = pathlib.Path(x["original_audio"])
    folder = original_audio.parent
    name = original_audio.stem
    flac = folder / "flac" / f"{name}.flac"
    gcs_uri = pathlib.Path(bucket_name) / flac.name
    transcription_gcs_uri = (
        pathlib.Path(bucket_name) / "transcripts" / f"{name}.json"
    )
    transcription_local = folder / "transcripts" / f"{name}.json"
    return pd.Series(
        {
            "original_audio": original_audio,
            "flac": flac,
            "gcs_uri": f"gs://{gcs_uri}",
            "transcription_gcs_uri": f"gs://{transcription_gcs_uri}",
            "transcription_download_blob": f"transcripts/{name}.json",
            "transcription_local": transcription_local,
            "split_folder": folder / "splits" / name,
            "tts_working_directory": folder / "tts",
        }
    )


@click.command()
@click.argument("input_file", type=click.File("r"))
@click.argument("output_file", type=click.File("w"))
@click.option("--bucket_name", default="barbero-translations")
def main(input_file, output_file, bucket_name):
    df = pd.read_csv(input_file)
    logger.debug(df)
    df = df.apply(calculate_paths, args=(bucket_name,), axis=1)
    logger.debug(df)
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
