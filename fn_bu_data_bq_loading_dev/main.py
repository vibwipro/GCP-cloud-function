#*************************************************************************#
#  Script : main.py
#  Developed by : Vibhor
#  Developed Date : 2022-08-08
#  Update Date : 2022-08-08
#  Version No : 1.00.01
#*************************************************************************#
# Version No : 1.00.01 : Initial version.
#*************************************************************************#

#****************import python lib*****************************************#
import json, requests, pandas as pd
from google.cloud import bigquery

#****************imort python function************************************#
from funct import fetchSecret


#****************python function******************************************#
def bu_dataloading(request):
    #****************imort variable list**********************************#
    with open('variable.json', 'r') as variable_json:
        variable_value = variable_json.read()
        variable_json_value = json.loads(variable_value)

    url = variable_json_value['bu-api-url']    
    headers_str='''{"key": "''' + fetchSecret() + '''"}'''
    headers= json.loads(headers_str)
    req = requests.get(url, headers=headers)
    data = req.json()
    df = pd.DataFrame(data)
    

   #*************** Capture dataset name*******************#
    dataset = request.args.get('dataset')
    print (dataset)
    table_id = dataset + '.' + variable_json_value['table-name']

   #*************** Create BiqQuery table if not present*******************#
    client = bigquery.Client()
    sql = 'CREATE table if not exists ' + dataset + '.' + variable_json_value['table-name'] + variable_json_value['sql'] 
    query_job = client.query(sql)
    query_job.result()
  
   #*************** Get BiqQuery Set up************************************#
    table = client.get_table(table_id)
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", create_disposition="CREATE_IF_NEEDED")
    #errors = client.insert_rows_from_dataframe(table, df)  # Make an API request.
    errors = client.load_table_from_dataframe(df, table, job_config=job_config).result()

    return "Success"


