#****************import python lib*****************************************#
import json, requests, pandas as pd
from google.cloud import bigquery
from google.cloud import storage
from io import StringIO 
import click
from typing import Any, Dict, Optional
import logging
import google.cloud.logging

from configs.utils import file_type_from_name

log = logging.getLogger(__name__)

scope_files = [
    "test.txt",
    "test1.txt"
]

#****************python function******************************************#

def main(event: Dict, context: Optional[Any]) -> None:
    """Entry-point for the Cloud Function """
    setup_logging(context.event_type)
    file_name = event["name"]
    print(f"Started to process an added file `{file_name}`")
    bucket = event["bucket"]
    
    config_generator = router(file_name)
    print (config_generator)

    #****************imort variable list**********************************#
    bucket_name = bucket
    file_name = file_name
    storage_client = storage.Client()

    # Get the GCS bucket and file
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Download the content of the file
    content = blob.download_as_text()

    # Convert the content to a Pandas DataFrame
    #df = pd.read_csv(pd.compat.StringIO(content), sep=',')
    df = pd.read_csv(StringIO(content), sep=',')
    if file_name == 'test1.txt':
        print (df)


    return "Success"

def setup_logging(event_type: str):
    """While running in production the logging is set up automatically - otherwise it is setup here"""
    if event_type != "local":
        client = google.cloud.logging.Client()
        client.setup_logging()
        log.info("Running in production - using inbuilt logging")
    else:
        log.info("Running locally")
        root = logging.getLogger()
        root.setLevel(logging.getLevelName(log_level))
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        root.addHandler(handler)


def router(file_name: str) -> callable:
    """Routes config generation to the appropriate function, based on the file name"""
    file_type = file_type_from_name(file_name)
    if file_type in scope_files:
        log.info(f"Simple file `{file_name}` detected - creating the relevant config")
        return 'present'#simple.get_simple_config

    else:
        error_message = f"No router found for file `{file_name}` of detected type `{file_type}`"
        raise NotImplementedError(error_message)


@click.command()
@click.option("--bucket", default="nbm-data-ingestion-input-e2e_pip_compile")
@click.option(
    "--gcs_path", default="e2e_pip_compile_data_ingestion_tables/RU/media_spend_weekly.xlsx",
)

def cli(bucket: str, gcs_path: str):
    DummyContext = namedtuple("DummyContext", "event_id event_type")
    dummy_event = {
        "name": gcs_path,
        "bucket": bucket,
    }
    main(dummy_event, DummyContext(event_id=0, event_type="local"))

if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter