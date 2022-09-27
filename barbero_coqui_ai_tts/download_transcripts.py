import logging

import click
from tqdm import tqdm
import pandas as pd
from google.cloud import storage


logger = logging.getLogger(__name__)


def download(row, bucket):
    blob = bucket.blob(row["transcription_download_blob"])
    blob.download_to_filename(row["transcription_local"])
    logger.debug(
        "downloaded %s from bucket %s blob %s", row["transcription_local"], bucket, blob
    )


@click.command()
@click.argument("input_file", type=click.File("r"))
@click.option("--bucket_name", default="barbero-translations")
def main(input_file, bucket_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    lines = pd.read_csv(input_file)
    tqdm.pandas()
    lines.progress_apply(download, args=(bucket,), axis=1)


if __name__ == "__main__":
    main()
