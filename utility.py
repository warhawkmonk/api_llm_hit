from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage  # For input format
import numpy as np
import wikipedia
import wikipediaapi
# import regex as re
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import requests

# Initialize the QA model for question-answering tasks
# qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")


@tool
def consume_llm_api(prompt):
    """
    Sends a prompt to a local LLM API endpoint and retrieves the generated response text for most of the generic results.

    Args:
        prompt (str): The input text prompt to be processed by the LLM.

    Returns:
        str: The generated text response extracted from the API's JSON output.

    Raises:
        requests.exceptions.RequestException: If the API request fails.
        KeyError: If the expected 'text' key is missing from the response.
    """
    url = "http://127.0.0.1:6000/api/llm-response"
    headers = {"Content-Type": "application/json"}
    payload = {"prompt": prompt,"extension":True}

    import requests
    print("Sending prompt to the LLM API...")
    response_ = requests.post(url, json=payload, verify=False)
    response_data = response_.json()
    return response_data['text']

@tool
def relevent_value(long_query, count=3):
    """
    Retrieves summary text and HTML content for the top Wikipedia search results. Use only when information is mandatory to confirm.

    Args:
        long_query (str): The search query to look up on Wikipedia.
        count (int, optional): The number of search results to retrieve. Defaults to 3.

    Returns:
        tuple: A tuple containing two dictionaries:
            - values (dict): A dictionary mapping page titles to their plain text summaries.
            - html_values (dict): A dictionary mapping page titles to their full HTML content.

    Note:
        This function requires the 'wikipedia' and 'wikipediaapi' libraries to be installed.
    """
    import wikipedia
    import wikipediaapi
    
    results = wikipedia.search(long_query, results=count)
    
    wiki_wiki = wikipediaapi.Wikipedia(user_agent='MyProjectName (merlin@example.com)', language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)
    wiki_wiki_html = wikipediaapi.Wikipedia(user_agent='MyProjectName (merlin@example.com)', language='en', extract_format=wikipediaapi.ExtractFormat.HTML)
    
    values = {}
    html_values = {}
    
    for result in results:
        page_py = wiki_wiki.page(result)
        page_html = wiki_wiki_html.page(result)
        html_values[result] = page_html.text
        values[result] = page_py.text
        
    return values, html_values
        
@tool
def query_generator(schema):
    """
    Generates 10 relevant search queries based on a provided schema.

    Args:
        schema (dict): The data schema to generate queries for.

    Returns:
        str: A Python list string containing 10 search queries.
    """
    query_search = "Generate 10 relevant small queries based on the schema provided.\nYour task is specifically to generate queries to search in google.\nGenerate in a python list only list no other text"
    long_query = f"{schema} \n {query_search}"
    query_list = consume_llm_api(long_query)
    return query_list


@tool
def query_template(query):
    """
    Refines a user query by making it more specific and relevant for data generation.

    Args:
        query (str): The original user query.

    Returns:
        str: A refined query with better specificity and clarity.

    Note:
        Removes format instructions and makes the query more relevant to data generation tasks.
    """
    template = f"user query: {query}.\n Above mentionded is user query and below are the instruction to update the query\n"
    template+= "1. Make the query more specific and relevant to the data generation task.\n"
    template+= "2. Remove the instruction in what format the data should be generated.\n"
    result = consume_llm_api(template)
    return result

def image_generation(prompt):
    """Generate an image prompt response from a local LLM endpoint.

    Args:
        prompt (str): The input text prompt to be processed by the local LLM.

    Returns:
        dict: Parsed JSON response from the local LLM endpoint.

    Raises:
        requests.exceptions.RequestException: If the API request fails.
        ValueError: If the response body cannot be parsed as JSON.
    """
    url = "http://127.0.0.1:6000/api/llm-response"
    headers = {"Content-Type": "application/json"}
    payload = {"only_prompt": prompt}
    init_image = [[1,1,1]*1000]*1000
    mask_image = [[True]*1000]*1000
    initial_image_base64 = numpy_to_list(np.array(init_image))
    mask_image_base64 = numpy_to_list(np.array(mask_image))
    payload = {
        "prompt": prompt,  # Replace with your desired prompt
        "initial_img": initial_image_base64,
        "masked_img": mask_image_base64,
        "negative_prompt": "go according to prompt" # Replace with your negative prompt
    }
    import requests
    print("Sending prompt to the LLM API...")
    response_ = requests.post(url, json=payload, verify=False)
    response_data = response_.json()
    return response_data


def numpy_to_list(array):

    current=[]
    for value in array:
        if isinstance(value,type(np.array([]))):
            result=numpy_to_list(value)
            current.append(result)
        else:
            
            current.append(int(value))
    return current