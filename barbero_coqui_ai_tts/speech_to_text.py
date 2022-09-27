import logging
import pathlib

import click
from tqdm import tqdm
import pandas as pd
from google.cloud import speech


logger = logging.getLogger(__name__)


def speech_to_text(row, client):
    audio = speech.RecognitionAudio(uri=row["gcs_uri"])
    config = speech.RecognitionConfig(
        language_code="it-IT",
        profanity_filter=False,
        enable_automatic_punctuation=True,
        model="latest_long",
    )
    output_config = speech.TranscriptOutputConfig(
        gcs_uri=row["transcription_gcs_uri"],
    )
    request = speech.LongRunningRecognizeRequest(config=config, audio=audio, output_config=output_config)
    operation = client.long_running_recognize(request=request)
    response = operation.result()
    return response


@click.command()
@click.argument("input_file", type=click.File("r"))
def main(input_file):
    client = speech.SpeechClient()
    lines = pd.read_csv(input_file).head(1)
    tqdm.pandas()
    lines.progress_apply(speech_to_text, args=(client,), axis=1)


if __name__ == "__main__":
    main()
