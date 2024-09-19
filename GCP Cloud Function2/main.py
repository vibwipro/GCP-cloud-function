#****************import python lib*****************************************#
import json, requests, pandas as pd
from google.cloud import bigquery
from google.cloud import storage
from io import StringIO 
import re, csv, numpy as np, io
from datetime import datetime
from io import BytesIO

current_time = datetime.now()
f_time = current_time.strftime('%H%M%S')


def process_survey_data(request):

    # imort variable list
    with open('variable.json', 'r') as variable_json:
        variable_value = variable_json.read()
        variable_json_value = json.loads(variable_value)

    country = variable_json_value['country']
    file_path = variable_json_value['process_file']
    country_code = next((item.get(country) for item in variable_json_value['country_code'] if country in item), None)
    strategy = variable_json_value['strategy']
    bucket_name = variable_json_value['bucket_name'] #'bigquery-dataset-cleanup-cloud-function'
    dev_project_name = variable_json_value['dev_project_name']



    storage_client = storage.Client()
    file_name = variable_json_value['process_file']
    print (file_name)

    # Get the GCS bucket
    bucket = storage_client.get_bucket(bucket_name)

    #****************my code ***********************************************#

    # Get the GCS data file
    blob = bucket.blob(variable_json_value['question_mapping_file'])
    # Download the content of the file
    content_question_mapping = blob.download_as_string()

    # read question mapping
    question_mapping_full = pd.read_excel(content_question_mapping, sheet_name='question_mapping')
    question_mapping = question_mapping_full[question_mapping_full['Country code'] == country]

    sheet_df_list = []

    # Get the GCS data file
    blob = bucket.blob(variable_json_value['process_file'])
    # Download the content of the file
    content_data = blob.download_as_string()

    for sheet in variable_json_value['sheets_to_extract']:
        # Load Excel sheet into a DataFrame
        df = pd.read_excel(content_data, sheet_name=sheet)
        df = df.dropna(how='all', axis=1)
        print(f'sheet "{sheet}" loaded')
        sheet_df_list.append(df)


    sheets_dict = dict(zip(variable_json_value['sheets_to_extract'], sheet_df_list))

    # Cleaning and transforming the datamap table
    datamap = sheets_dict['Datamap']
    datamap.columns = ['Question stub', 'Code', 'Answer']
    datamap = datamap[
            ~(datamap['Question stub'].str.contains('|'.join(['Open text response', 'Values:']), na=False))]
    datamap = datamap.dropna(how='all')
    datamap['Question stub'] = datamap['Question stub'].fillna(method='ffill')
    datamap = datamap[(datamap['Question stub'].str.contains('|'.join(
            ['S[0-9]+', 'G[0-9]+', 'J[0-9]+', 'EQ[0-9]+', 'AG', 'h_region_india']
    ), na=False))]

    # Removing sqaure brackets from question stub
    cond0 = datamap['Question stub'].str.contains('\[', na=False)
    #datamap.loc[cond0, 'Question stub'] = datamap.loc[cond0, 'Question stub'].str.replace('\[', '')
    #datamap.loc[cond0, 'Question stub'] = datamap.loc[cond0, 'Question stub'].str.replace('\]', '')
    datamap.loc[cond0, 'Question stub'] = datamap.loc[cond0, 'Question stub'].str.replace(r'\[', '', regex=True)
    datamap.loc[cond0, 'Question stub'] = datamap.loc[cond0, 'Question stub'].str.replace(r'\]', '', regex=True)

    datamap['Question stub'] = datamap['Question stub'].str.replace(': ', ' - ')

    # Removing square brackets from question code
    datamap['Question code'] = None
    cond = datamap['Code'].str.contains('\[', na=False)
    datamap.loc[cond, 'Question code'] = datamap.loc[cond, 'Code'].str.replace('\[', '')
    datamap.loc[cond, 'Question code'] = datamap.loc[cond, 'Question code'].str.replace('\]', '')

    datamap['Question number'] = datamap['Question stub'].str.split(' - ', expand=True)[0]

    # Creating the question column
    datamap['Question'] = None
    datamap.loc[datamap['Question code'].isnull(), 'Question code'] = ''
    cond3 = datamap['Question code'] != ''
    datamap.loc[cond3, 'Question'] = datamap.loc[cond3, 'Question stub'].astype(str) + ' ' + datamap.loc[
            cond3, 'Answer'].astype(str)
    datamap.loc[~cond3, 'Question'] = datamap.loc[~cond3, 'Question stub'].astype(str)

    ### Answer / value mapping - Final
    value_maps = datamap[datamap['Question code'] == ""][['Question', 'Code', 'Answer']]
    value_maps = value_maps.dropna(how='any')
    value_maps['Code'].apply(pd.to_numeric)

    # Final Question mapping - Final
    datamap.loc[datamap['Question code'] == "", 'Question code'] = datamap.loc[
            datamap['Question code'] == "", 'Question number']
    mapping = datamap[['Question code', 'Question']].drop_duplicates()

    # Filtering for completed survey entries only - this is indiciated by the 'status' field being set to 3
    conjoint_results = sheets_dict['A1']
    conjoint_results = conjoint_results[conjoint_results['status'] == 3]

    # Applying the question mapping
    conjoint_results = conjoint_results.rename(columns=dict(zip(question_mapping['Question code'], question_mapping['Question'])))
    #conjoint_results = conjoint_results.rename(columns=dict(zip(mapping['Question code'], mapping['Question'])))
    conjoint_results = conjoint_results.rename(columns={'record': 'Internal Respondent Number'})

    # Applying the answer mapping
    for question in value_maps['Question'].unique():
            df_map = value_maps[(value_maps['Question'] == question)]
            if any(df_map['Answer'].isin(['Checked', 'Unchecked'])):  # or ('G21' in question) or ('G22' in question)
                print('passed')
                pass
            else:
                try:
                    conjoint_results[question] = conjoint_results[question].replace(
                        dict(zip(df_map['Code'], df_map['Answer'])))
                except KeyError:
                    print('2nd attempt:', question)
                    for col in conjoint_results.columns[conjoint_results.columns.str.contains(question)]:
                        #                     conjoint_results[f'{col}_copy'] = conjoint_results[col].copy()
                        conjoint_results[col] = conjoint_results[col].replace(
                            dict(zip(df_map['Code'], df_map['Answer'])))

    # Creating the Household size question from the no. of children and no. of adults columns (for the newer surveys for which this is applicable)
    household_split_cond = conjoint_results.columns.str.contains('|'.join(['S11A', 'S11B']))
    if any(household_split_cond):
            household_cols = list(
                conjoint_results.columns[conjoint_results.columns.str.contains('|'.join(['S11A', 'S11B']))])
            s11_col = 'S11 - How many people are there in your household (including yourself)?'
            conjoint_results[s11_col] = conjoint_results[household_cols].replace("10+", "10").astype(int).sum(axis=1)
            conjoint_results.loc[conjoint_results[s11_col] >= 10, s11_col] = '10+'

    # Creating G2, G3, G4 and G5 for countries missing question
    missing_g234_questions = [
            'US', 'Japan', 'Portugal', 'Netherlands', 'Australia', 'Switzerland', 'Austria', 'Norway', 'Hungary',
            'Denmark',
            'Czech', 'Slovakia', 'South Korea', 'Finland', 'Serbia', 'Croatia', 'Romania', 'France', 'Belgium',
            'Poland',
            'Italy', 'Slovenia', 'UK', 'India', 'Ireland'
        ]


    if country in missing_g234_questions:

            # Total purchases including 'I brought it home with me directly'
            total_purchases_hfa = 0
            total_purchases_f = 0

            # Delivery purchases
            total_delivery_hfa = 0
            total_delivery_f = 0

            for a in ['G3', 'G4']:
                for b in ['HFA', 'F']:

                    all_purchase_cols = conjoint_results.columns[
                        conjoint_results.columns.str.contains('|'.join([f"{a}B_{b}"]))]

                    delivery_purchase_cols = [col for col in all_purchase_cols if
                                              'I brought it home with me directly' not in col]

                    purchase_cols_df_new = pd.DataFrame()
                    purchase_cols_df_new['Names'] = all_purchase_cols
                    purchase_cols_df_new['Names'].str.split('?', expand=True)

                    # G5
                    total_purchases = conjoint_results[all_purchase_cols].sum(axis=1)
                    delivery_purchases = conjoint_results[delivery_purchase_cols].sum(axis=1)
                    if b == 'HFA':
                        total_purchases_hfa += total_purchases
                        total_delivery_hfa += delivery_purchases
                    elif b == 'F':
                        total_purchases_f += total_purchases
                        total_delivery_f += delivery_purchases

                    if a == 'G3':
                        purchase_stub = 'Which of the following delivery methods have you used for in-store purchases of furniture and home furnishings in the last 12 months?'

                        # G2
                        g2_col = f'G2_{b} - Have you used any delivery service for purchases of furniture or home furnishing accessories in a physical store in the last 12 months?'
                        conjoint_results[g2_col] = np.where(delivery_purchases > 0, 1.0, 2.0)


                    elif a == 'G4':
                        purchase_stub = 'Think about your purchases of furniture online, which of the following delivery methods have you used the most recent 12 months?'

                    col_name = a + '_' + b + ' - ' + purchase_stub

                    purchase_cols_df_new['New col'] = col_name + \
                                                      purchase_cols_df_new['Names'].str.split('?', expand=True)[1]
                    purchase_new_cols = conjoint_results[delivery_purchase_cols] > 0
                    purchase_new_cols = purchase_new_cols.rename(
                        columns=dict(zip(purchase_cols_df_new['Names'], purchase_cols_df_new['New col'])))

                    for col in purchase_new_cols.columns:
                        purchase_new_cols[col] = np.where(purchase_new_cols[col], 1.0, 0.0)

                    conjoint_results = conjoint_results.merge(purchase_new_cols, how='left', left_index=True,
                                                              right_index=True)

            # Add G5
            # NL, CH and AU already have G5
            missing_g5_question = ['US', 'Japan', 'Portugal', 'UK', 'Ireland']
            if country in missing_g5_question:
                conjoint_results['total_purchases_hfa'] = total_purchases_hfa
                conjoint_results['total_purchases_f'] = total_purchases_f

                conjoint_results['total_delivery_hfa'] = total_delivery_hfa
                conjoint_results['total_delivery_f'] = total_delivery_f

                purchase_freq_map = {'Once': 1,
                                     'Twice': 2,
                                     '3 times': 3,
                                     '4-6 times': 4,
                                     '7-9 times': 7,
                                     '10+ times': 10,
                                     '20+ times': 20
                                     }

                for b1 in ['HFA', 'F']:
                    if b1 == 'HFA':
                        total_del_col = total_delivery_hfa
                        total_pur_col = total_purchases_hfa
                    elif b1 == 'F':
                        total_del_col = total_delivery_f
                        total_pur_col = total_purchases_f

                    conjoint_results[f'total_delivery_{b1}'] = total_del_col
                    g5_delivery_name = f'G5A_{b1} - Out of your home furniture or accessories purchases, for how many did you use delivery services?'
                    conjoint_results[g5_delivery_name] = None

                    conjoint_results[f'total_purchases_{b1}'] = total_pur_col
                    g5_purchases_name = f'G5_{b1} - How many times did you purchase home furnishing accessories during the past 12 months?'
                    conjoint_results[g5_purchases_name] = None

                    for key, value in purchase_freq_map.items():
                        conjoint_results.loc[conjoint_results[f'total_purchases_{b1}'] > value, g5_purchases_name] = key
                        conjoint_results.loc[conjoint_results[f'total_delivery_{b1}'] > value, g5_delivery_name] = key


    # Filtering out unneeded columns
    conjoint_results = conjoint_results[conjoint_results.columns[conjoint_results.columns.str.contains('|'.join(
            ['S[0-9]+', 'G[0-9]+', 'J[0-9]+', 'EQ[0-9]+', 'AG', 'CJ_TYPE', 'Internal Respondent Number',
             'h_region_india']
    ))]]

    # Exporting to file
    print('Exporting to file')
    #conjoint_results.to_excel(intermediate_folder + sep + f'delivery_conjoint_results_{country_code}.xlsx', sheet_name=country)

    '''file_name_final = 'final_output.csv'

    #Convert filtered data to CSV format
    csv_final_data = conjoint_results.to_csv(index=False)

    # Create a blob object
    blob1 = bucket.blob(file_name_final)

    # Upload CSV data to the blob
    blob1.upload_from_string(csv_final_data, content_type='text/csv')'''

    print('Done')   

    excel_name = f'uk_data/delivery_conjoint_results_{country_code}.xlsx'

    # Convert DataFrame to Excel format in memory
    excel_data = BytesIO()
    conjoint_results.to_excel(excel_data, index=False, sheet_name=country)
    excel_data.seek(0)  # Reset the position to the beginning of the BytesIO object

    # Upload Excel data to GCS bucket
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(excel_name)
    blob.upload_from_file(excel_data, content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    print(f'excel file {excel_name} written')

    #***************Notebook 2*************************
    # List of countries with postcode --> location mapping
    # for these counties we use a mapping from postcodes to locations to replace the city entered by the respondents in a free text column
    postcode_mapping_countries = [
        'Canada', 'UK', 'Germany', 'Spain', 'US', 'Russia', 'Portugal', 'Switzerland', 'Netherlands',
        'Austria', 'Norway', 'Denmark', 'South Korea', 'Belgium', 'Ireland']

    # Some countries have a secondary definition for locations
    secondary_location_column = {'Spain': 'drive_time', 'US': 'market', 'Portugal': 'Commercial Zone'}

    # Get the GCS data file
    blob = bucket.blob(variable_json_value['conjoint_analysis_intermediate_file'])
    # Download the content of the file
    content_conjoint = blob.download_as_string()

    # get hfb mapping to product category
    item_category_df = pd.read_excel(content_conjoint,
                                     sheet_name='Item-category mapping')
    item_category_df.loc[:, 'Answer'] = item_category_df['Answer'].str.strip()
    purchase_category_unique = item_category_df['Purchase category'].unique()

    item_copy = item_category_df.copy()

    # 1. Set truncation strategy for willingness to pay
    # same price as highest price in conjoint matrix (High for ROC kitchen)
    truncation_dict = {
        'Russia': 2000,
        'Italy': 165,
        'UK': 100,
        'Canada': 160,  # 80-100
        'Sweden': 1600,
        'Spain': 100,
        'Germany': 150,
        'US': 150,
        'Portugal': 165,
        'Japan': 20000,
        'Netherlands': 175,
        'Australia': 300,
        'Switzerland': 180,
        'Austria': 180,
        'Norway': 4000,
        'Hungary': 40000,
        'Denmark': 1500,
        'Czech': 2400,
        'Slovakia': 100,
        'South Korea': 160000,
        'Finland': 220,
        'Serbia': 7000,
        'Croatia': 449,
        'Romania': 299,
        'France': 125,
        'Belgium': 260,
        'Poland': 499,
        'Slovenia': 150,
        'Ireland': 100,
        'India': 3000,
    }

    # 2. Preprocess country-specific survey results
    question_category_df_list = []

    conjoint_filename_dict = {f'{country}': f'delivery_conjoint_results_{country_code}.xlsx'}

    print(conjoint_filename_dict.items())

    for country, file in conjoint_filename_dict.items():
        print(country)

        if country in ['Russia']:  # [ITALY','Russia']:
            sheets_to_extract = ['LABELS', 'VALUES']
        else:
            sheets_to_extract = [country]

        sheet_df_list = []
        for sheet in sheets_to_extract:
            #df_temp = pd.read_excel(file, sheet_name=sheet)
            df_temp = conjoint_results
            print(f'sheet "{sheet}" loaded')
            sheet_df_list.append(df_temp)

        sheets_dict = dict(zip(sheets_to_extract, sheet_df_list))


        if country not in ['Russia']:
            df = sheets_dict[country]
        else:
            questions_to_replace = [
                'S2', 'S3', 'S4', 'S6', 'S7', 'S7A', 'S8', 'S9', 'S12',
            ]

            col_list = []
            for question_number in questions_to_replace:
                col_list += [col for col in sheets_dict['VALUES'].columns if question_number + ' - ' in col]

            df = sheets_dict['VALUES']
            df[col_list] = sheets_dict['LABELS'][col_list]

        ########################################################################################################################
        # Creating the question-category mapping for purchase items (S14)
        s14_cols = list(df.columns[df.columns.str.startswith('S14')])
        question_category_single_df = pd.DataFrame()
        question_category_single_df['Question'] = s14_cols
        question_category_single_df['Country'] = country
        question_category_single_df['Question_number'] = 'S14'
        question_category_single_df['Answer'] = question_category_single_df['Question'].str.split('?', expand=True)[1]
        question_category_single_df['Answer'] = question_category_single_df['Answer'].str.lstrip()
        question_category_single_df = question_category_single_df.merge(item_category_df, how='left', on=['Answer'])
        question_category_df_list.append(question_category_single_df)
        print(question_category_single_df.head(3))

        ########################################################################################################################

        # Calculate purchase category
        for purchase_category in purchase_category_unique:
            cond = (question_category_single_df['Purchase category'] == purchase_category)
            question_select = question_category_single_df[cond]['Question']
            match_cols = list(set(question_select).intersection(set(df.columns)))
            df[purchase_category] = df[match_cols].sum(axis=1) > 0

        df['Purchases-category'] = 'None'
        df.loc[df['Accessories'], 'Purchases-category'] = 'Accessories only'
        df.loc[(~df['Accessories']) & (
                    df[purchase_category_unique].sum(axis=1) >= 1), 'Purchases-category'] = 'Multi non-accessories'
        df.loc[(df['Accessories']) & (df[purchase_category_unique].sum(axis=1) > 1), 'Purchases-category'] = 'Multi'
        #df.loc[df['None'], 'Purchases-category'] = 'None'
        missing_values = df['Purchases-category'].isna() 
        df.loc[missing_values, 'Purchases-category'] = 'None'


        # Trim whitespaces from column names
        df.columns = df.columns.str.strip()

        df['Country'] = country
        df['Flag'] = df['Country'] + '-' + df['Internal Respondent Number'].apply(str)
        print(df['Flag'])

        # IKEA customer indicator
        a = df[df.columns[(df.columns.str.contains('|'.join(['IKEA', 'Ikea']))) & (df.columns.str.contains('S15'))]]
        df['ikea-bool'] = a.max(axis=1, skipna=True)
        df['IKEA customer'] = np.where(df['ikea-bool'] == 1.0, 'Yes', 'No')

        # Fill NaN values with an empty string ('')
        df.columns = df.columns.fillna('')

        # Willingness to pay - truncate values
        country_cols_to_extract = df.columns[df.columns.str.contains('|'.join(['G3[A-Z]', 'G4[A-Z]', 'G6B']))]

        if country in truncation_dict.keys():
            for price_col in country_cols_to_extract:
                replace_vals = (df.loc[df[price_col] >= truncation_dict[
                    country], price_col] / 100.0) if strategy == 'divide_by_100' else np.nan
                df.loc[df[price_col] >= truncation_dict[country], price_col] = replace_vals

        if country in postcode_mapping_countries:

            # Get the GCS data file
            blob = bucket.blob(variable_json_value['location_file'])
            # Download the content of the file
            content_location = blob.download_as_string()

            # Get the postcode to location mapping
            location_df = pd.read_excel(content_location, sheet_name=country_code, skiprows=0)
            location_df = location_df.drop_duplicates()
            # ensure postcodes are capitalised
            location_df['postcode'] = location_df['postcode'].apply(lambda x: str(x).upper())

            location_columns = ['postcode', 'location']

            if country in secondary_location_column.keys():
                location_columns.append(secondary_location_column[country])

            if country == 'Russia':
                postcode_column = 'S5 - What is your postal code?'
            elif country in ['US', 'Portugal', 'Switzerland', 'Netherlands', 'Austria', 'Denmark', 'Norway',
                             'South Korea', 'Belgium', 'UK', 'Ireland']:
                postcode_column = 'S4 - What is your zip code?'
            else:
                postcode_column = 'S4 - What is your postal code?'

            print("Records before location mapping:")
            print(df.shape[0])

            # ensure postcodes are capitalised in the data before the join
            df[postcode_column] = df[postcode_column].apply(lambda x: str(x).upper())
            location_df = location_df[location_columns].drop_duplicates()

            # Use left 4 numbers
            if country in ['Netherlands', 'Portugal']:
                df[postcode_column] = df[postcode_column].str[:4]

            df = df.merge(location_df, how='left', left_on=[postcode_column], right_on=['postcode'])
            df.loc[df['location'].isnull(), 'location'] = 'Other'
            df.rename(columns={location_columns[1]: "Location1"}, inplace=True)

            if country in secondary_location_column.keys():
                df.rename(columns={location_columns[2]: "Location2"}, inplace=True)

            print("Records after location mapping:")
            print(df.shape[0])

        # Exporting sheet
        print('exporting', sheet)
        #df.to_excel('countryTab.xlsx', sheet_name=country, index_label=False, index=False)

        excel_countryTab = 'uk_data/countryTab.xlsx'

        # Convert DataFrame to Excel format in memory
        excel_data = BytesIO()
        df.to_excel(excel_data, index=False, sheet_name=country)
        excel_data.seek(0)  # Reset the position to the beginning of the BytesIO object

        # Upload Excel data to GCS bucket
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(excel_countryTab)
        blob.upload_from_file(excel_data, content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        print('excel file {excel_countryTab} written')

    # check question G6A
    delivery_preferences_cols_future = df.columns[df.columns.str.contains('G6A')].to_list()
    delivery_preferences_cols_future.insert(0, 'Internal Respondent Number')



    # 3. Export the question-category mapping for all countries

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

    question_category_df_list = []
    for country, code in country_codes_combine.items():
        print('reading in', country)
        country_df = pd.read_excel(content_conjoint, sheet_name=country)
        print('done')
        # Creating the question-category mapping for purchase items (S14)
        s14_cols = list(country_df.columns[country_df.columns.str.startswith('S14')])
        question_category_single_df = pd.DataFrame()
        question_category_single_df['Question'] = s14_cols
        question_category_single_df['Country'] = country
        question_category_single_df['Question_number'] = 'S14'
        question_category_single_df['Answer'] = question_category_single_df['Question'].str.split('?', expand=True)[1]
        question_category_single_df['Answer'] = question_category_single_df['Answer'].str.lstrip()
        question_category_single_df = question_category_single_df.merge(item_category_df, how='left', on=['Answer'])
        question_category_df_list.append(question_category_single_df)

    question_category_mapping = pd.concat(question_category_df_list)

    print('exporting', sheet) 
    #question_category_mapping.to_excel('question_category_mapping.xlsx', sheet_name='Question-category mapping'+f_time, index_label=False, index=False)

    excel_question_category = 'uk_data/question_category_mapping.xlsx'

    # Convert DataFrame to Excel format in memory
    excel_data = BytesIO()
    df.to_excel(excel_data, index=False, sheet_name='Question-category mapping')
    excel_data.seek(0)  # Reset the position to the beginning of the BytesIO object

    # Upload Excel data to GCS bucket
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(excel_question_category)
    blob.upload_from_file(excel_data, content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    print('excel file {excel_question_category} written')


    return "Success"

