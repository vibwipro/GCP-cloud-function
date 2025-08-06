#*************************************************************************#
#  Script : main.py
#  Developed by : Vibhor
#  Developed Date : 2022-08-08
#  Update Date : 2022-08-08
#  Version No : 1.00.01
#*************************************************************************#
# Version No : 1.00.01 : Initial version.
#*************************************************************************#

#****************import python lib****************************************#
import os, re, json, requests, datetime
from google.cloud import bigquery
#****************import python functions**********************************#
from function import bucketname, get_all_datasets, get_referenced_datasets, get_datasets_tagged_for_deletion, save_datasets_tagged_for_deletion

tagged_for_deletion_limit_days = 1
usage_analysis_limit_days = 6
dataset_expiration_limit_days = 10
bigquery_client = bigquery.Client()

def clean_bq_dataset(event):
    print("Dataset removal function started")
    
    all_datasets = get_all_datasets()
    print (all_datasets)
    #versioned_datasets = all_datasets
    versioned_datasets = [
        dataset
        for dataset in all_datasets
        if re.match(r"cost_model_v[a-f0-9]{6}", dataset["schema_name"])
    ]
    print ("Print Versioned DS")
    print (versioned_datasets)

    # Datasets over 21 days old (excluding the latest version)
    expired_datasets = [
        dataset
        for dataset in sorted(versioned_datasets, key=lambda x: x["schema_name"])[
            :-1
        ]
        if datetime.datetime.now(tz=datetime.timezone.utc) - dataset["creation_time"]
        > datetime.timedelta(dataset_expiration_limit_days)
    ]
    print ('printing *****expired_datasets******** ')
    print (expired_datasets)

    referenced_datasets = get_referenced_datasets()

    print (referenced_datasets)
    print ('******Start of get dataset for deletion***********')

    datasets_tagged_for_deletion = get_datasets_tagged_for_deletion()

    print (datasets_tagged_for_deletion)

    for dataset in expired_datasets:
        if dataset["schema_name"] in referenced_datasets:
            print(
                f"Dataset '{dataset['schema_name']}' was referenced during last {usage_analysis_limit_days} days"
            )
            datasets_tagged_for_deletion.pop(dataset['schema_name'], None)
            print (dataset['schema_name'])
            #send_notification_to_support_channel(f"Dataset '{dataset['schema_name']}' was untagged for deletion due to being referenced in the last {tagged_for_deletion_limit_days} days")

        else:
            print(
                f"Dataset '{dataset['schema_name']}' was not referenced during the last {usage_analysis_limit_days} days."
            )
            if dataset["schema_name"] in datasets_tagged_for_deletion:
                time_since_tagged =  datetime.datetime.now() - datetime.datetime.strptime(datasets_tagged_for_deletion[dataset["schema_name"]]['tagged_date'], "%Y-%m-%dT%H:%M:%S")
                
                if time_since_tagged > datetime.timedelta(tagged_for_deletion_limit_days):
                    print(f"Dataset '{dataset['schema_name']}' is now being deleted")
                    #send_notification_to_support_channel(f"Dataset '{dataset['schema_name']}' is now being deleted")
                    bigquery_client.delete_dataset(
                        dataset['schema_name'], delete_contents=True, not_found_ok=True
                    )  # Make an API request.
                    datasets_tagged_for_deletion.pop(dataset['schema_name'], None)
            else:
                print(f"Dataset '{dataset['schema_name']}' is now tagged for deletion")
                datasets_tagged_for_deletion[dataset['schema_name']] = {'tagged_date': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}
                #send_notification_to_support_channel(f"Dataset '{dataset['schema_name']}' is now tagged for deletion and will be deleted if not referenced in the next {tagged_for_deletion_limit_days} days")

    save_datasets_tagged_for_deletion(datasets_tagged_for_deletion)
    #send_notification_to_support_channel("Inspect logs <https://console.cloud.google.com/logs/query;query=resource.typecloud_function%22%0Aresource.labels.function_name%20%3D%20%22bigquery-dataset-cleanup%22%region%20%3D%20%22europe-west1?project=xxxxxxx|here>")

    return f'Dataset removal function completed!'