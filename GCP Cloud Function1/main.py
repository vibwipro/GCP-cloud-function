#****************import python lib*****************************************#
import json, requests, pandas as pd
from google.cloud import bigquery
from google.cloud import storage
from io import StringIO 
import re, csv
from slack_sdk import WebhookClient

#slack_support_webhook_url =  "https://hooks.slack.com/services/TMEBJ4/B0556/90wdMzQCVul"
slack_support_webhook_url =  "https://hooks.slack.com/services/TMJ4/B0/igXoAaNbN"
client = bigquery.Client()



#****************python function******************************************#
def dataloading1(request):
    # Read the CSV file containing schema names to be excluded
    exclude_df = pd.read_csv('exclude_schema.csv', header=None)  # Assuming no header in the CSV file
    exclude_schema_names = exclude_df[0].tolist()

    #****************imort variable list**********************************#
    bucket_name = 'try-test-dev' #'bigquery-dataset-cleanup-cloud-function'
    folder = 'bq_dataset_audit'
    file_name = 'bq_dataset_audit.txt'
    dev_project_name = 'igg-bunns-mes-dev'
    '''storage_client = storage.Client()

    # Get the GCS bucket and file
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Download the content of the file
    content = blob.download_as_text()

    # Convert the content to a Pandas DataFrame
    #df = pd.read_csv(pd.compat.StringIO(content), sep=',')
    df = pd.read_csv(StringIO(content), sep=',')
    print (df)'''
    # Get the GCS bucket and file
    #storage_client = storage.Client()

    # Get the GCS bucket and file
    #bucket = storage_client.get_bucket(bucket_name)
    #blob = bucket.blob(folder + '/' + file_name)

    region_code = "region-eu"
    query = f"""
        SELECT schema_name, creation_time, last_modified_time,
        ROW_NUMBER() OVER (PARTITION BY REGEXP_EXTRACT(schema_name, r'^([^\_]+_[^\_]+)') ORDER BY creation_time DESC) AS row_num
        FROM `ingxxx-eev.region-eu`.INFORMATION_SCHEMA.SCHEMATA
        where TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), last_modified_time, DAY) / 30.4368 > 6 -- Convert difference in days to months and filter
    """
    query_job = client.query(query)
    datasets = [row for row in query_job.result()]


    #*******************
    print ('print datasets')
    print (datasets)
    print ('data finish')
    #*******************

    # Store the SQL output in a list
    dataset_names = [row for row in query_job.result()]

    result = []
    pattern_utsikt_data = r'utsikt_data_v\d{1,2}\b'
    pattern_cost_model = r'cost_model_v\d{3,4}\b'
    pattern_data_ingestion = r'data_ingestion_v\d{1,3}_\d{1,2}_\d{1,2}\b'

    for dataset in datasets:
        schema_name = dataset.schema_name
        if re.search(pattern_utsikt_data, schema_name) or re.search(pattern_cost_model, schema_name) or re.search(pattern_data_ingestion, schema_name):
            result.append({
                'schema_name': dataset.schema_name,
                'creation_time': dataset.creation_time,
                'last_modified_time': dataset.last_modified_time
            })

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(result)

    # Calculate row_num equivalent using pandas
    df['row_num'] = df.groupby(df['schema_name'].str.extract(r'^([^\_]+_[^\_]+)')[0])['creation_time'].rank(method='first', ascending=False)

    # Filter the results based on the condition row_num > 4
    filtered_result = df[df['row_num'] > 4]

    # Extract only schema_name column
    schema_names = filtered_result['schema_name']

    # Filter out schema names obtained from BigQuery query that are not in the exclude list
    schema_names = [schema_name for schema_name in schema_names if schema_name not in exclude_schema_names]

    print (schema_names)
    # Write data to CSV file
    '''try:
        # Convert DataFrame to CSV string
        csv_string = schema_names.to_csv(index=False)

        # Upload the CSV string to GCS
        blob.upload_from_string(csv_string, content_type='text/csv')

        print('DataFrame uploaded to GCS successfully.')
    except Exception as e:
        print(f"An error occurred while uploading data: {e}")'''

    #********************************************************************

    # Download the CSV file from GCS
    #local_file_path = '/tmp/' + file_name
    #blob.download_to_filename(local_file_path)

    # Read the CSV file into a DataFrame
    #df = pd.read_csv(local_file_path)

    # Convert DataFrame to Markdown table format
    #table_md = df.to_markdown(index=False)

    #output = f"**Test Message, Please ignore it** \n Hi All👋, \n Following datasets are absolete and will be considered for deletion by next Monday. \n If any datasets needs to be restores then please contact Cost Model team"

    #output += f"\n\n Dataset names are given below: \n\n" + table_md 
    #output += f"\n\nYou can download the CSV file from this link: {blob.public_url}"

    #print(output)

    #WebhookClient(url=slack_support_webhook_url).send(text=output)


    #****************************************************************************

    # Download the file's content as a string
    #content = blob.download_as_string().decode("utf-8")

    # Parse the CSV content
    #csv_reader = csv.reader(content.splitlines())

    # Skip the first line (header)
    '''next(csv_reader)

    # Print each row line by line
    for row in csv_reader:
        print(row)
        dataset_name = row[0]
        #client.delete_dataset(
        #    dataset_name, delete_contents=True, not_found_ok=True
        #)
        print('dataset deleted: {row}')'''
    
    return "Success"
