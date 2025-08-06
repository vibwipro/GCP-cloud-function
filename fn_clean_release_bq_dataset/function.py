from google.cloud import storage
from google.cloud import bigquery
#from google.cloud.bigquery.table import Row
from typing import Any
import datetime, os, json
from google.cloud.bigquery.table import Row

dev_project_name = "xxxxxxxx-business-metrics-dev"
usage_analysis_limit_days = 6
bucket_name = 'cs_clean_release_bq_dataset'
datasets_tagged_for_deletion_path = 'datasets_tagged_for_deletion.json'

bigquery_client = bigquery.Client()
storage_client = storage.Client()

def bucketname(request):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket("cs_clean_release_bq_dataset")
    return (bucket)

def get_all_datasets()-> list[dict[str, Any]]:
    """Gets metadata for all datasets in region EU"""

    dataset_query_job = bigquery_client.query(
        """
        SELECT *
        FROM `xxxxxxxx-business-metrics-dev.region-eu`.INFORMATION_SCHEMA.SCHEMATA
        """,
        location="EU",
    )
    all_datasets = [dict(dataset) for dataset in dataset_query_job]
    return all_datasets

def get_referenced_datasets() -> set[str]:
    """Gets datasets from BigQuery logs that have been referenced in the last `usage_analysis_limit_days` days"""
    
    query = f"""
    SELECT
        protopayload_auditlog.servicedata_v1_bigquery.jobCompletedEvent.job.jobStatistics.referencedTables
    FROM
        `{dev_project_name}.bq_logs_eu.cloudaudit_googleapis_com_data_access`
    WHERE
        protopayload_auditlog.methodName = "jobservice.jobcompleted" AND
        DATE(timestamp) >= "{(datetime.datetime.now() - datetime.timedelta(usage_analysis_limit_days)).strftime('%Y-%m-%d')}"
    """

    query_job = bigquery_client.query(query)

    def get_dataset_ids_from_row(row: Row) -> list[str]:
        (referenced_datasets,) = row
        return [table["datasetId"] for table in referenced_datasets]

    return {dataset for row in query_job for dataset in get_dataset_ids_from_row(row)}

def get_datasets_tagged_for_deletion() -> dict[str, str]:
    """Gets information from Cloud Storage about datasets tagged for deletion"""

    BUCKET = storage_client.get_bucket(bucket_name)
    blob = BUCKET.get_blob(datasets_tagged_for_deletion_path)
    datasets_tagged_for_deletion = (blob.download_as_string())
    if not datasets_tagged_for_deletion or datasets_tagged_for_deletion.strip() == '':
        datasets_tagged_for_deletion = {}
    else:
        datasets_tagged_for_deletion = json.loads(datasets_tagged_for_deletion)  

    return datasets_tagged_for_deletion

def save_datasets_tagged_for_deletion(datasets_tagged_for_deletion: dict[str, str]):
    """Saves information in Cloud Storage about datasets tagged for deletion"""

    BUCKET = storage_client.get_bucket(bucket_name)
    blob = BUCKET.blob(datasets_tagged_for_deletion_path)
    
    blob.upload_from_string(
        data=json.dumps(datasets_tagged_for_deletion),
        content_type='application/json'
        )