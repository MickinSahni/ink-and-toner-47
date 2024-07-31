from utilities import loop_through_datasets_on_data_gov, extract_metadata_from_dataset_website, extract_data_dictionary_from_download_metadata_url, get_relevant_datasets, get_relevant_datasets2, get_dataset_data, get_data_dictionary, strip_recent_views, ADDITIONAL_RULES, construct_question, generate_sql_query
import pandas as pd
from openai import OpenAI
import requests
import re
import pandasql as ps

datasets_info = loop_through_datasets_on_data_gov(1, 2)

api_key = 'INSERT API KEY HERE'
data_gov_api_key = 'INSERT API KEY HERE'
# Open Terminal: You can find it in the Applications folder or search for it using Spotlight (Command + Space).
# Edit Bash Profile: Use the command nano ~/.bash_profile or nano ~/.zshrc (for newer MacOS versions) to open the profile file in a text editor.
# Add Environment Variable: In the editor, add the line below, replacing your-api-key-here with your actual API key:
# export OPENAI_API_KEY='your-api-key-here'
# Save and Exit: Press Ctrl+O to write the changes, followed by Ctrl+X to close the editor.
# Load Your Profile: Use the command source ~/.bash_profile or source ~/.zshrc to load the updated profile.
# Verification: Verify the setup by typing echo $OPENAI_API_KEY in the terminal. It should display your API key.
max_characters = 10
pd.set_option("display.max_colwidth", max_characters)
print(datasets_info.iloc[0].T)


if __name__ == "__main__":
    # for practice, let's use JUST the datasets that have data dictionaries
    dfs_with_data_dictionaries = datasets_info[datasets_info['data_dictionary'].notnull()]
    # prompt the user to ask a question of the data
    user_question = input("Please enter your question that can possibly be answered using data.gov datasets:")
    # find relevant datasets
    client = OpenAI(api_key = api_key)
    relevant_datasets = get_relevant_datasets2(user_question, dfs_with_data_dictionaries)
    print(f"for your question {user_question}, here are the datasets we'll use: {relevant_datasets}")
    dataset_name = relevant_datasets[0]
    # get data
    df = get_dataset_data(dataset_name, data_gov_api_key)
    df.columns = df.columns.str.lower() #for consistency later with query generation
    # uploading using .csv because I cannot find the data.gov api
    electric_vehicle_population_data = pd.read_csv("Electric_Vehicle_Population_Data.csv")
    electric_vehicle_population_data.columns = electric_vehicle_population_data.columns.str.lower()
    # get schema
    dfs_with_data_dictionaries['dataset_name'] = dfs_with_data_dictionaries['dataset_name'].str.replace(r'\s*recent views$', '', regex=True)
    schema_link = get_data_dictionary(dfs_with_data_dictionaries, dataset_name)
    schema = requests.get(schema_link)
    schema = schema.json()
    # generate query
    dataset_name_with_underscores = dataset_name.replace(' ', '_')
    question = construct_question(schema=schema, user_question=user_question, df=dataset_name_with_underscores)
    query = generate_sql_query(question)
    sql_query = re.sub(r'^```sql\s*|\s*```$', '', query).strip()
    # Remove quotes and newline characters
    cleaned_query = sql_query.replace('"', '').replace('\n', ' ')
    # apply query to table
    result = ps.sqldf(cleaned_query, locals())
    print(result)
