import pandas as pd
import boto3
import concurrent.futures


bucket = "barbero-tts2"
vocabulary_name = "barbero-guerre"


def speech_to_text(row, client):
    job_name = f'{row["path"].parent.name}-{row["path"].stem}'
    output_key = f"transcriptions/{job_name}.json"
    client.start_transcription_job(
        TranscriptionJobName=job_name,
        OutputBucketName=bucket,
        OutputKey=output_key,
        Media={'MediaFileUri': row["s3_uri"]},
        MediaFormat='wav',
        LanguageCode='it-IT',
        Settings={
            "VocabularyName": vocabulary_name,

        },
        JobExecutionSettings={
            "AllowDeferredExecution": True,
            "DataAccessRoleArn": "arn:aws:iam::180517682866:role/s3-role",
        },
    )

    # create a future to wait for the job to finish
    return job_name


df = pd.read_pickle("/hd4tb/archivio_barbero/splits/notebook/chunks.pkl.gz")
df.head(1).apply(speech_to_text, axis=1, client=boto3.client("transcribe"))
