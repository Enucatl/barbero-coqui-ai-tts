import logging

import click
import pydub
from tqdm import tqdm
import pandas as pd


logger = logging.getLogger(__name__)


def convert(row):
    input_file = row["input_file"]
    output_file = row["output_file"]
    sound = pydub.AudioSegment.from_mp3(input_file)
    sound = sound.set_channels(1)  # save to mono
    sound.export(output_file, format="flac")
    logger.debug("converted %s to %s", input_file, output_file)


@click.command()
@click.argument("input_file", type=click.File("r"))
def main(input_file):
    lines = pd.read_csv(input_file)
    tqdm.pandas()
    lines.progress_apply(convert, axis=1)


if __name__ == "__main__":
    main()
