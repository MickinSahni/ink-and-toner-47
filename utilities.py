import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from openai import OpenAI
import re

# get dataset metadata
def loop_through_datasets_on_data_gov(start_page_number, end_page_number, search_word=""):
    '''
    This function loops through pages of data.gov and extracts all datasets and associated metadata
    :param start_page_number: The starting page number for the search results
    :param end_page_number: The ending page number for the search results
    :param search_word: The search term to filter datasets, default is an empty string
    :return: A list of dictionaries containing dataset names and URLs
    '''
    base_url = "https://catalog.data.gov/dataset/"
    datasets = []
    # do loop
    for page_number in range(start_page_number, end_page_number + 1):
        if search_word:
            url = f"{base_url}?q={search_word}&page={page_number}"
        else:
            url = f"{base_url}?page={page_number}"
        # fetch response
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        # Find all dataset items
        dataset_items = soup.find_all("div", class_="dataset-content")
        # get items
        for item in dataset_items:
            dataset_name = item.find("h3").get_text(strip=True)
            dataset_url = item.find("a", href=True)['href']
            # Extract additional metadata
            metadata = extract_metadata_from_dataset_website(dataset_url)
            # append to the list
            datasets.append({
                "dataset_name": dataset_name,
                "dataset_url": f"https://catalog.data.gov{dataset_url}",
                "description": metadata["description"],
                "metadata_update_date": metadata["metadata_update_date"],
                "download_metadata_url": metadata["download_metadata_url"],
                "data_dictionary": metadata["data_dictionary"]
            })
    # Convert the list of dictionaries to df
    datasets_df = pd.DataFrame(datasets)
    return datasets_df

def extract_metadata_from_dataset_website(dataset_url):
    '''
    This function extracts pertinent metadata from the dataset's website 
    :param dataset_url: The URL of the dataset on data.gov
    :return: A dictionary containing the description and metadata update date
    '''
    full_url = f"https://catalog.data.gov{dataset_url}"
    response = requests.get(full_url)
    soup = BeautifulSoup(response.content, "html.parser")
    # Extract description
    description_div = soup.find("div", itemprop="description")
    description = description_div.get_text(strip=True) if description_div else None
    # Extract metadata update date
    metadata_update_span = soup.find("span", property="dct:modified")
    metadata_update_date = metadata_update_span.get_text(strip=True) if metadata_update_span else None
    # Extract download metadata URL
    download_metadata_link = soup.find("a", href=True, text="Download Metadata")
    download_metadata_url = f"https://catalog.data.gov{download_metadata_link['href']}" if download_metadata_link else "No download URL available"
    # Extract the data dictionary from download_metadata_url (if possible)
    data_dictionary = extract_data_dictionary_from_download_metadata_url(download_metadata_url)
    return {
        "description": description,
        "metadata_update_date": metadata_update_date,
        "download_metadata_url": download_metadata_url,
        "data_dictionary": data_dictionary
    }

def extract_data_dictionary_from_download_metadata_url(download_metadata_url):
    '''
    This function searches within the download_metadata_url to identify if there is a data dictionary we can use
    CURRENTLY, we are only looking for .json data dictionaries referenced by XXXX.json
    In the future, we hope to develop more robust search functionality to find more types of data dictionaries (.csv, .rdf, html, pdf, etc.)
    '''
    try:
        # Request the JSON data from the URL
        response = requests.get(download_metadata_url)
        response.raise_for_status()  # Raise an error for bad responses
        metadata = response.json()
        # Extract the data dictionary URL
        data_dictionary_url = None
        distributions = metadata.get('distribution', [])
        for distribution in distributions:
            described_by = distribution.get('describedBy', '')
            if described_by.endswith('.json'):
                data_dictionary_url = described_by
                break
        return data_dictionary_url
    except requests.exceptions.RequestException as e:
        print(f"Error fetching metadata: {e}")
        return None
    except json.JSONDecodeError:
        print("Error decoding JSON from metadata.")
        return None

# example
datasets_info = loop_through_datasets_on_data_gov(1, 2)
ev_url = "/dataset/electric-vehicle-population-data"
ev_metadata = extract_metadata_from_dataset_website(ev_url)
max_characters = 10
pd.set_option("display.max_colwidth", max_characters)
print(datasets_info.iloc[0].T)
ev_metadata_url = "https://catalog.data.gov/harvest/object/53dd245f-25d9-4758-8fb3-c6e2383e081b"
ev_data_dictionary = extract_data_dictionary_from_download_metadata_url(ev_metadata_url)




# Open AI
api_key = 'INSERT API KEY'
client = OpenAI(api_key=api_key)
def get_relevant_datasets(user_question, datasets_info):
    """
    Given the user's question, identify the most relevant datasets from the provided DataFrame.

    Parameters:
    user_question (str): The question provided by the user.
    datasets_info (pd.DataFrame): DataFrame containing datasets with data dictionaries.

    Returns:
    list: List of relevant datasets (rows from the DataFrame).
    """
    try:
        # Validate the datasets_info DataFrame
        if not isinstance(datasets_info, pd.DataFrame):
            raise ValueError("datasets_info must be a pandas DataFrame")
        if 'dataset_name' not in datasets_info.columns or 'description' not in datasets_info.columns:
            raise ValueError("DataFrame must contain 'dataset_name' and 'description' columns")
        # Prepare the context for GPT by combining dataset names and descriptions
        context = ""
        for _, row in datasets_info.iterrows():
            context += f"Dataset Name: {row['dataset_name']}\nDescription: {row['description']}\n\n"
        # Create a prompt for GPT
        prompt = f"The user asked: '{user_question}'\n\nBased on the question, which of the following datasets seem most relevant?\n\n{context}"
        # Call the GPT model to get the response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant, skilled in identifying relevant datasets based on user queries."},
                {"role": "user", "content": prompt}
            ]
        )
        # Print the raw response for debugging
        print("Raw response:", response)
        
        # Ensure response contains expected data
        if not isinstance(response, dict):
            raise ValueError("Response is not a dictionary")
        choices = response.get('choices', [])
        if not choices:
            raise ValueError("No choices found in response")
        first_choice = choices[0]
        if 'message' not in first_choice:
            raise ValueError("First choice does not contain 'message'")
        message = first_choice['message']
        if 'content' not in message:
            raise ValueError("Message does not contain 'content'")
        relevant_datasets_text = message['content']
        # Extract relevant datasets from the response
        relevant_datasets = []
        for _, row in datasets_info.iterrows():
            if row['dataset_name'] in relevant_datasets_text:
                relevant_datasets.append(row)
        return relevant_datasets
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def get_relevant_datasets2(user_question, datasets_info):
    """
    Given the user's question, identify the most relevant datasets from the first 3 rows of the DataFrame.
    
    Parameters:
    user_question (str): The question provided by the user.
    datasets_info (pd.DataFrame): DataFrame containing datasets with data dictionaries.
    
    Returns:
    list: List of relevant datasets (rows from the DataFrame).
    """
    try:
        # get relevant datasets
        context = ""
        for i, (_, row) in enumerate(datasets_info.iterrows()):
            if i >= 3:  # Limit to 3 iterations
                break
            context += f"Dataset Name: {row['dataset_name']}\nDescription: {row['description']}\n\n"
        prompt = f"The user asked: '{user_question}'\n\nBased on the question, which of the following datasets seem most relevant?\n\n{context}"
        client = OpenAI(api_key = api_key)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                    {"role": "system", "content": "You are a knowledgeable assistant, skilled in identifying relevant datasets based on user queries."},
                    {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        print(completion.choices[0].message)
        response_message = completion.choices[0].message.content
        # Use regular expressions to extract dataset names
        dataset_names = re.findall(r'\*\*Dataset Name: ([^\*]+)\*\*', response_message)
        # Clean up dataset names (remove leading/trailing whitespace)
        dataset_names = [name.strip() for name in dataset_names]
        return dataset_names
    except Exception as e:
        print(f"An error occurred: {e}")


# Get data and schema
def get_dataset_data(api_key, limit=1):
    """
    Retrieve data from the NREL Alt Fuel Stations API.
    
    Parameters:
    api_key (str): API key for NREL.
    limit (int): Number of records to retrieve.
    
    Returns:
    dict: The data from the API or an error message if the request fails.
    """
    try:
        # Construct the URL for the NREL Alt Fuel Stations API
        url = f"https://developer.nrel.gov/api/alt-fuel-stations/v1.json"
        params = {
            'limit': limit,
            'api_key': api_key
        }
        # Make the GET request to the API
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        # Parse and return the JSON data
        data = response.json()
        return data
    except requests.RequestException as e:
        return {"error": f"An error occurred: {e}"}
    
def get_data_dictionary(df, dataset_name):
    """
    Retrieve the value from the 'data_dictionary' column for the row with the given dataset_name.

    Parameters:
    datasets_info (pd.DataFrame): DataFrame containing dataset information with 'dataset_name' and 'data_dictionary' columns.
    dataset_name (str): The name of the dataset to search for.

    Returns:
    dict or None: The value from 'data_dictionary' if the dataset_name is found, otherwise None.
    """
    # Filter the DataFrame to find the row with the matching dataset_name
    row = df[df['dataset_name'] == dataset_name]
    if row.empty:
        return None  # Return None if the dataset_name is not found
    # Extract and return the 'data_dictionary' value
    return row.iloc[0]['data_dictionary']


def strip_recent_views(df, column_name):
    """
    Strip the words 'recent views' from the end of each string in the specified column.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the column to modify.
    column_name (str): The name of the column from which to strip 'recent views'.

    Returns:
    pd.DataFrame: The DataFrame with updated column values.
    """
    # Regular expression to match 'recent views' at the end of the string
    df[column_name] = df[column_name].str.replace(r'\s*recent views$', '', regex=True)
    return df


# get query for the data
ADDITIONAL_RULES = """
If more than one quesion was asked, only answer the first question.
If you are unsure of the answer, reply with 'I am not sure' then give a more detailed explanation
Do now include any additional text before or after the SQL statement
Only display the SQL statement
The SQL statement will only return a maximum of 100 rows
Use the table name provided (it comes after 'for the table ' in the prompt instead of 'FROM your_table_name'. Do not guess the table name
all column names in the query should be lower case
"""
def construct_question(schema, user_question, df):
    '''
    Generate question that we'll submit to chatgpt to create the query
    '''
    prompt = f"for the table '{df}', please generate a query to answer the {user_question}. The schema appears below: {schema} ADDITIONAL RULES {ADDITIONAL_RULES}"
    return prompt


def generate_sql_query(query):
    """
    given the construct_question, generate a SQL query to use
    """
    try:
        prompt = query
        client = OpenAI(api_key = api_key)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                    {"role": "system", "content": "You are a SQL expert"},
                    {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        print(completion.choices[0].message)
        response_message = completion.choices[0].message.content
        return response_message
    except Exception as e:
        print(f"An error occurred: {e}")
