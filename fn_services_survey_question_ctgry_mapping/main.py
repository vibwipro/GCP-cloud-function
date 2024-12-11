#****************import python lib*****************************************#
import json, pandas as pd
from google.cloud import storage
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor



def read_country_data(country, content_conjoint):
    country_df = pd.read_excel(content_conjoint, sheet_name=country)
    return country, country_df

def process_countries(country_codes, content_conjoint, bucket_name, item_category_df):
    storage_client = storage.Client()
    question_category_df_list = []
    for country in country_codes:
        country_df = pd.read_excel(content_conjoint, sheet_name=country)
        # Process country data...
        s14_cols = list(country_df.columns[country_df.columns.str.startswith('S14')])
        question_category_single_df = pd.DataFrame()
        question_category_single_df['Question'] = s14_cols
        question_category_single_df['Country'] = country
        question_category_single_df['Question_number'] = 'S14'
        question_category_single_df['Answer'] = question_category_single_df['Question'].str.split('?', expand=True)[1]
        question_category_single_df['Answer'] = question_category_single_df['Answer'].str.lstrip()
        question_category_single_df = question_category_single_df.merge(item_category_df, how='left', on=['Answer'])
        question_category_df_list.append(question_category_single_df)

    return question_category_df_list

def split_country_codes(country_codes, num_chunks):
    chunk_size = (len(country_codes) + num_chunks - 1) // num_chunks
    return [country_codes[i:i + chunk_size] for i in range(0, len(country_codes), chunk_size)]

def process_survey_question_category_mapping(request):
    # Import variable list
    with open('variable.json', 'r') as variable_json:
        variable_value = variable_json.read()
        variable_json_value = json.loads(variable_value)

    #country = variable_json_value['country']
    #file_path = variable_json_value['process_file']
    #country_code = next((item.get(country) for item in variable_json_value['country_code'] if country in item), None)
    #strategy = variable_json_value['strategy']
    bucket_name = variable_json_value['bucket_name']
    dev_project_name = variable_json_value['dev_project_name']

    storage_client = storage.Client()
    #file_name = variable_json_value['process_file']

    # Get the GCS bucket
    bucket = storage_client.get_bucket(bucket_name)

    # Get the GCS data file
    blob = bucket.blob(variable_json_value['conjoint_analysis_intermediate_file'])
    content_conjoint = blob.download_as_string()

    # Get hfb mapping to product category
    item_category_df = pd.read_excel(content_conjoint, sheet_name='Item-category mapping')
    item_category_df.loc[:, 'Answer'] = item_category_df['Answer'].str.strip()

    country_codes_combine = {
        'Canada': 'CA',
        'Sweden': 'SE',
        'Spain': 'ES',
        'Germany_2024': 'DE',
        'Germany_2023': 'DE',
        'Italy_old': 'IT',
        'Italy_new': 'IT',
        'France': 'FR',
        'Russia': 'RU',
        'US': 'US',
        'Japan': 'JP',
        'Portugal': 'PT',
        'Netherlands_2022': 'NL',
        'Netherlands_2024': 'NL',
        'Australia': 'AU',
        'Switzerland': 'CH',
        'Austria': 'AT',
        'Norway': 'NO',
        'Hungary': 'HU',
        'Denmark': 'DK',
        'Czech': 'CZ',
        'Slovakia': 'SK',
        'South Korea': 'KR',
        'Finland': 'FI',
        'Serbia': 'RS',
        'Croatia': 'HR',
        'Romania': 'RO',
        'Belgium': 'BE',
        'Poland': 'PL',
        'Slovenia': 'SI',
        'UK': 'UK',
        'Ireland': 'IE',
        'India': 'IN',
    }

    # Split the country codes into chunks
    country_code_chunks = split_country_codes(list(country_codes_combine.keys()), 10)  # Adjust the number of chunks as needed

    # Process each chunk in parallel
    question_category_df_list = []
    with ProcessPoolExecutor(max_workers=len(country_code_chunks)) as executor:
        futures = []
        for chunk in country_code_chunks:
            future = executor.submit(process_countries, chunk, content_conjoint, bucket_name, item_category_df)
            futures.append(future)

    for future in futures:
        question_category_df_list.extend(future.result())

    question_category_mapping = pd.concat(question_category_df_list)

    excel_question_category = 'uk_data/question_category_mapping.xlsx'

    # Convert DataFrame to Excel format in memory
    excel_data = BytesIO()
    question_category_mapping.to_excel(excel_data, index=False, sheet_name='Question-category mapping')
    excel_data.seek(0)  # Reset the position to the beginning of the BytesIO object

    # Upload Excel data to GCS bucket
    blob = bucket.blob(excel_question_category)
    blob.upload_from_file(excel_data, content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    return "Success"