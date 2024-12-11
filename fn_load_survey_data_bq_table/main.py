import pandas as pd
from google.cloud import storage, bigquery
from io import BytesIO

def surveydataloading(request):

    # Example usage
    bucket_name = 'try-test-dev'
    file_name = 'IKEA_Ireland.xlsx'
    sheet_name = 'A1'
    dataset_id = 'cost_model_vVib'  # Replace with your BigQuery dataset ID
    table_id = 'survey_data'

    try:
        # Create a storage client
        storage_client = storage.Client()

        # Get the GCS bucket and file
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(file_name)

        # Download the content of the file as binary
        content = blob.download_as_bytes()

        # Convert the content to a Pandas DataFrame
        df = pd.read_excel(BytesIO(content), sheet_name=str(sheet_name), header=0)

        # Filter columns that start with 'S'
        selected_columns = [col for col in df.columns if str(col).strip().lower().startswith('s')]
        print('Selected columns are:')
        print(selected_columns)

        # Create a new DataFrame with selected columns
        filtered_df = df[selected_columns]
        print (filtered_df)

        # Create BigQuery client
        client = bigquery.Client()

        # Create BigQuery table schema
        schema = [bigquery.SchemaField(col, 'STRING') for col in selected_columns]

        # Create BigQuery table
        table_ref = client.dataset(dataset_id).table(table_id)
        table = bigquery.Table(table_ref, schema=schema)
        client.create_table(table)

        # Write data to BigQuery table
        job_config = bigquery.LoadJobConfig()
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE  # Use WRITE_TRUNCATE or WRITE_APPEND as needed

        job = client.load_table_from_dataframe(filtered_df, table_ref, job_config=job_config)
        job.result()  # Wait for the job to complete

        return "Success"

    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error"



