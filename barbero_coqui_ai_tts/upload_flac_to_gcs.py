import logging
import pathlib

import click
from tqdm import tqdm
import pandas as pd
import google.cloud.storage


logger = logging.getLogger(__name__)


def upload(row, bucket):
    blob = bucket.blob(pathlib.Path(row["output_file"]).name)
    blob.upload_from_filename(row["output_file"])
    logger.debug("uploaded %s to bucket %s blob %s", row["output_file"], bucket, blob)


@click.command()
@click.argument("input_file", type=click.File("r"))
@click.option("--bucket_name", default="barbero-translations")
def main(input_file, bucket_name):
    storage_client = google.cloud.storage.Client()
    bucket = storage_client.bucket(bucket_name)
    lines = pd.read_csv(input_file)
    tqdm.pandas()
    lines.progress_apply(upload, args=(bucket,), axis=1)


if __name__ == "__main__":
    main()
