import logging
import pathlib

import pydub
import click
from tqdm import tqdm
import pandas as pd


logger = logging.getLogger(__name__)


def split_file(row, sound, folder):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    output_name = pathlib.Path(folder) / f"{row.name:04}.wav"
    output_sound = sound[row["begin_time"] : row["end_time"]]
    output_sound.export(output_name, format="wav")
    return output_name.relative_to(output_name.parent.parent)


def split_files(row):
    df = pd.read_json(row["transcription_local"])["results"].apply(
        lambda x: pd.Series(
            {
                "transcript": x["alternatives"][0]["transcript"],
                # pydub does things in milliseconds
                "end_time": int(1000 * float(x["resultEndTime"].replace("s", ""))),
            }
        ),
    )
    df["begin_time"] = df["end_time"].shift(periods=1, fill_value=0)
    sound = pydub.AudioSegment.from_file(row["flac"], format="flac").set_frame_rate(16000)
    tqdm.pandas()
    df["text"] = df["transcript"]
    df["audio_file"] = df.progress_apply(
        split_file, args=(sound, row["split_folder"]), axis=1
    )
    return df


@click.command()
@click.argument("input_file", type=click.File("r"))
@click.argument("output_file", type=click.File("w"))
def main(input_file, output_file):
    lines = pd.read_csv(input_file).head(1)
    tqdm.pandas()
    df = lines.progress_apply(split_files, axis=1)
    df = pd.concat(df.tolist())
    df[["audio_file", "text"]].to_csv(
        output_file, index=False, sep="|"
    )


if __name__ == "__main__":
    main()
