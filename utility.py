from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage  # For input format

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
    Sends a prompt to a local LLM API endpoint and retrieves the generated response text.

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
def construction_edit(textual_value, schema):
    """
    Constructs a structured prompt to guide an LLM in formatting raw text into a 
    specific dictionary-based JSON schema.

    Args:
        textual_value (str): The raw text data extracted (e.g., from Wikipedia) 
            to be parsed and formatted.
        schema (dict/str): The required dictionary structure or schema definition 
            that the output must adhere to.

    Returns:
        str: The structured text returned by the LLM, representing the 
            populated dictionary.

    Note:
        This function relies on the `consume_llm_api` function to process 
        the constructed prompt. It enforces strict output constraints, 
        such as equal list lengths for dictionary values and the omission 
        of conversational filler text.
    """
    construction_prompt = textual_value + "\n"
    construction_prompt += "Above is the generated text from wikipedia and below is the rule that has to be filled in the data. \n"
    construction_prompt += "The data should be in the form of a dictionary and it must follow the following schema: \n"
    construction_prompt += str(schema) + "\n"
    construction_prompt += "The length of each list of each key must be same in the generated data(mandatory)." + "\n"
    construction_prompt += "No helper text like 'Here is the filled-in JSON schema based on the provided text' or 'Note: I've filled in the keys with relevant data' . " + "\n"
    construction_prompt += "The output must be a dictionary" + "\n"
    
    # Assuming consume_llm_api is defined in the caller's scope
    constructed_text = consume_llm_api(construction_prompt)
    return constructed_text


@tool
def dictionary_check(construction_edit):
    """
    Validates that all values in a dictionary are non-empty lists or non-empty values.
    
    Args:
        construction_edit (dict): A dictionary containing lists or other values to validate.
        
    Returns:
        bool: True if all values are non-empty, False if any value is empty.
        
    Example:
        >>> dictionary_check({'key1': [], 'key2': [1,2]})
        False
        >>> dictionary_check({'key1': [1], 'key2': [2,3]})
        True
    """
    for keys in construction_edit:
        if len(construction_edit[keys]) == 0:
            return False
    return True


@tool
def actual_value(textual_value, schema):
    """
    Processes textual values and formats them according to a specified schema.

    Args:
        textual_value (dict): A dictionary containing text data to be processed.
        schema (dict): The schema that defines the expected structure and keys.

    Yields:
        dict: Formatted dictionary with values matching the schema structure.

    Note:
        Removes duplicates from list values and filters out 'Na' entries.
        Handles errors gracefully by yielding None if processing fails.
    """
    for j in textual_value:
        formatted_result = str(textual_value[j])+ "\n"
        formatted_result += "Please fill the following schema with the relevant data from the text above."+ "\n"
        formatted_result += "Here is the schema"+"\n"
        formatted_result += str(schema)
        formatted_result += "Please generate data according to schema and fill this template with your answers.\n"
        formatted_result += "You have to fill each key with the relevant data from the text above."+ "\n"
        formatted_result += "Please return the exact key value pair as the schema above. "+ "\n"
        formatted_result += "No helper text like 'Here is the filled-in JSON schema based on the provided text' or 'Note: I've filled in the keys with relevant data' ."+ "\n"
        formatted_result += "Only fill the keys that are in the schema."+ "\n"
        formatted_result += "If you are not sure about the data, you can add 'Na'."+ "\n"
        formatted_result += "It's an order you can not add any other text(e.g Here is the filled-in JSON schema) or note ."+ "\n"
        formatted_result += "The length of each list of each key must be same in the generated data(mandatory)."+"\n"
        raw_output = consume_llm_api(formatted_result)
        try:
            data = construction_edit(raw_output, schema)
            json_object_match = re.search(r'\{(?:[^{}]|(?R))*\}', data)
            access_value = eval(json_object_match.group())
            for schema_key in schema:
                if schema_key not in access_value:
                    access_value[schema_key] = list(set())
            for schema_key in access_value:
                access_value[schema_key] = list(set(access_value[schema_key]))
                access_value[schema_key] = list(set(access_value[schema_key]) - set(["Na"]))
            yield access_value
        except Exception as e:
            print(f"Error processing output: {e}")
            access_value = None
        


@tool
def context_data_relevancy(value, context):
    """
    Checks if missing values can be filled from the provided context.

    Args:
        value (dict): A dictionary with potentially missing values.
        context (str): The context text to search for filling missing values.

    Returns:
        str: Returns '@yahoo@' if data can be filled (90% confidence), '@NO@' otherwise.

    Note:
        Uses an LLM to assess relevancy of context data with strictness applied.
    """
    researcher =  "You are a professional reasearcher from data ."+ "\n"
    researcher += "You have to check can we fill some of the missing values in the "+str(value) + ". \n"  
    researcher += "The possible part which available in the context has to be relevent with already present data"+ ". \n"
    researcher += "from the context given below"+ ". \n"
    researcher += context+ "\n"
    researcher += "Be strict while thing of filling data"+ ". \n"
    researcher += "Just return @yahoo@ if 90% possible else @NO@"+ ". \n"


    result = consume_llm_api(researcher)
    return result


@tool
def agent_work_result(query, value):
    """
    Processes agent commands based on query classification and performs relevant actions.

    Args:
        query (str): The user query/command to be processed.
        value (dict): The current data dictionary being filled.

    Returns:
        str: Result message indicating the action taken or status.

    Note:
        Classifies queries into three categories:
        1. Extract data - searches Wikipedia for relevant information
        2. Append data - adds new values to specified keys
        3. Check data - validates data completeness and consistency
    """
    agent_understanding = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    query_embedding = agent_understanding.encode(query)
    score1 = util.cos_sim(query_embedding,agent_understanding.encode("extract data for"))
    score2 = util.cos_sim(query_embedding,agent_understanding.encode("append data in "))
    score3 = util.cos_sim(query_embedding,agent_understanding.encode("check data"))

    if score1 > score2 and score1 > score3:
        # print("Extracting query:", query)
        question = "search word ?"
        result = qa_model(question=question, context=query)
        result = result['answer']
        print("Extracting query:", result)
        wikisearch = relevent_value(result,3)
        html_pages = wikisearch[1]
        wikisearch = wikisearch[0]

        for searches in wikisearch:
            if "@yahoo@" in context_data_relevancy(value,wikisearch[searches]):
                return wikisearch[searches]
        return "No data found"
    elif score2 > score1 and score2 > score3:
        try:
            print("Appending command:", query)
            question1 = "which value we are adding to key ?"
            result1 = qa_model(question=question1, context=query)
            question2 = "In which key we are appending ?"
            result2 = qa_model(question=question2, context=query)
            result1 = result1['answer']
            result2 = result2['answer']

            if  len(value[result2])==0:
                value[result2].append(result1)
                return "Now you can fill the remaining columns"
            else:
                return "You are putting value in the same key column again not accepted."
        except Exception as e:
            return str(e)
    else:
        min_=0
        max_=0
        for keys in value:
            
            if len(value[keys])<min_:
                min_=len(value[keys])
            if len(value[keys])>max_:
                max_=len(value[keys])
        if min_==max_:
            return "You dia a great job"
        else:
            return "Please append the data correctly so that the length of each key is same and data is also relevant"
    
@tool
def full_alignment(value):
    """
    Checks if all keys in a dictionary have non-empty values.

    Args:
        value (dict): A dictionary with list or other values to check.

    Returns:
        bool: True if all values are non-empty, False if any value is empty.
    """
    for values in value:
        if len(value[values])==0:
            return False
    return True


@tool
def query_formatting(result):
    """
    Formats query results by splitting on newlines and removing empty first lines.

    Args:
        result (str): A newline-separated string to be formatted.

    Returns:
        list: A list of non-empty formatted query strings.
    """
    values=result.split("\n")
    if len(values)!=0:
        values.pop(0)
    return values

@tool
def missing_value_completion(store, value):
    """
    Attempts to fill missing values in a dictionary using data from a store.

    Args:
        store (dict): A dictionary containing search results or reference data.
        value (dict): The target dictionary with missing values to be filled.

    Returns:
        dict: Updated dictionary with missing values filled where possible.

    Note:
        Returns early once all values are aligned/non-empty.
        Uses LLM to extract relevant data from store text.
    """
    filler_prompt = "Below is mentioned ajson data\n"
    filler_prompt += str(value)+"\n"
    filler_prompt += "you only need to find missing data from the mentioned context section."
    filler_prompt += "You will return the results in below mentioned format.\n"
    filler_prompt += "The output will be in json format."
    filler_prompt += "context:\n"
    
    for search_key in store:
        try:
            fill_text = store[search_key]
            response = consume_llm_api(filler_prompt+fill_text)
        
            json_object_match = re.search(r'\{(?:[^{}]|(?R))*\}', response)
            access_value=eval(json_object_match.group())
            for keys in value:
                if len(value[keys])==0 and keys in access_value:
                    value[keys].append(access_value[keys].pop(0))
            print(value)
            if full_alignment(value):
                return value
        except:
            pass
        


@tool
def verification(value):
    """
    Searches for and fills missing values by generating search queries.

    Args:
        value (dict): A dictionary with potentially missing values.

    Returns:
        dict or list: The updated dictionary with filled values, or search query results.

    Note:
        Generates search queries for missing values and uses Wikipedia to find relevant data.
        Continues searching until all values are aligned or no more results are found.
    """
    validation_prompt = "Can you prepare a list of text(many as possible) that can be searched on google for filling(relevent data) the missing data below.\n"
    validation_prompt += str(value)+"\n"
    validation_prompt += "You need to prepare it by the following manner"
    validation_prompt += "1. Mention it line by line.\n"
    validation_prompt += "2. Please seperate it line by line.\n"
    validation_prompt += "3. Headers are not required\n"
    validation_prompt += "4. Please do not add any helper text example: Here is the required search queries , Here are the search queries .\n"
    validation_prompt += "5. Please do not add any notes"
    print("Searching for missing values")
    result=query_formatting(consume_llm_api(validation_prompt))

    for search_queries in result:
        if len(search_queries)!=0:
            print(search_queries)
            store=relevent_value(search_queries)
            html_pages = store[1]
            store = store[0]
            missing_value_completion(store,value)
        if full_alignment(value):
            return value

            



    return result



@tool
def agent_data_prep(value, query):
    """
    Prepares and fills data dictionary using an agentic workflow with LLM decision-making.

    Args:
        value (dict): The initial data dictionary to be filled.
        query (str): The user query describing the data preparation task.

    Returns:
        dict: The completed and fully aligned data dictionary.

    Note:
        Uses OpenAI LLM to generate intelligent commands for data extraction and appending.
        Implements a reward system (virtual $1000 per correct append) and penalty system.
        Continues iterating until all dictionary keys have values of equal length.
    """
    end_result = ""
    angent_earlier_income ="0"
    pre_money_saving = "0"
    mission = "First to fill most importent column \n"
    while end_result!="You dia a great job":
        
        if full_alignment(value):
            return value
            

        agent_instruction = mission
        agent_instruction += "your previous income"+pre_money_saving+"\n"
        agent_instruction += "your current income"+angent_earlier_income+"\n"
        pre_money_saving = angent_earlier_income
        if end_result=="You are putting value in the same key column again not accepted.":
            
            mission = "Why you are always filling the"+[i for i in value][-1]+"only.\n"
            mission += "We are removing $1000 from you account \n"
            angent_earlier_income = str(int(angent_earlier_income)-1000)
        agent_instruction += end_result + "\n" +"Above is the result of your previous command. Please give the next command to the agent."
        agent_instruction += query  + "\n"
        agent_instruction += "Below is the data gathered upto now" + "\n"
        agent_instruction += str(value) + "\n"
        agent_instruction += "Please utilize the tool where you can command the agent to do any of the following tasks(one instruction at a time )"+ "\n"
        agent_instruction += "You only have to fill one value for each key if its not present. \n"
        agent_instruction += "From now onwards your each statement is understand as command which is categoried in any of the commands in mentioned below examples. \n"
        agent_instruction += "1. Ask agent to extract data from the web about anything like search for lamp production ,smartphone parts etc .\n"
        agent_instruction += "2. Give any specific value to append in current generated data . Please also mention the key in which the agent has to append the data .\n"
        agent_instruction += "3. Ask the agent to put the generated data on check weather each column fills correctly or not .\n"
        agent_instruction += "Here is the instruction to give commands to the agent. \n"
        agent_instruction += "You can give commands to the agent ,few examples are mentioned below. \n"
        
        agent_instruction += "1. Extract data about iron man suit  or iron man suit mark1 \n"
        agent_instruction += "(while thinking about extract data look into the data \n"
        agent_instruction += "where data can be append and then search relevent query \n"
        agent_instruction += "like green arrow from DC only if DC and green arraow is in different column key values )\n\n"

        agent_instruction += "2. Append value 'bmw 4' to Car Model key \n"
        agent_instruction += "(While appending the value you must have read the data from extract data command and remember, if you found anything relevent don't forget to append.\n"
        agent_instruction += "The appending value has to be different not already present.) \n\n"
            
        agent_instruction += "Any different grammatical version of the above commands. \n"
        agent_instruction += "Command has to be given only for 'data filling' purpose. \n"

        agent_instruction += "While command like search for or extract information about something it has to be relevent query search. \n"
        agent_instruction += "The relevent the query the more accurate the data will be. \n"
        agent_instruction += "Be cautious while filling the data It has to be correct. \n"
        agent_instruction += "For each correct append you will get $1000. \n"

        agent_instruction += "Give your command only no text . \n"

        agent_instruction += "There will an audit after filling all the columns on data for its validity. \n"
        agent_instruction += "Some mistakes are okay but But if we find you guilty there are some repercussion."

        # instructionto give commands to the agent

        llm = ChatOpenAI(
            model="google/gemma-4-e4b",
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            temperature=0.7
        )
        response = llm.invoke(agent_instruction)
        command = response.content
        
        end_result = agent_work_result(command,value)
        if "Now you can fill the remaining columns" in end_result:
            angent_earlier_income = str(int(angent_earlier_income)+1000)
        print("--------------------")
        print(value)
        print("--------------------")
    return value


@tool
def dictionary_formatting(value):
    """
    Formats a dictionary by stripping whitespace from keys and reorganizing structure.

    Args:
        value (dict): The input dictionary to be formatted.

    Returns:
        dict: A new dictionary with cleaned keys and same values.
    """
    new_dict={}
    for data_keys in [i for i in value]:
        key_values = data_keys.strip()
        if key_values in value:
            if key_values not in new_dict:
                new_dict[key_values] =[]
            new_dict[key_values] = value.pop(key_values)
        else:
            new_dict[key_values] = value.pop(data_keys)
    return new_dict
            

@tool 
def schema_formatter(output):
    """
    Converts a comma-separated string into a schema dictionary with empty lists as values.

    Args:
        output (str): A comma-separated string of schema keys.

    Returns:
        dict: A dictionary with keys from the output string and empty lists as values.
    """
    schema = {i.strip():[] for i in output.split(",")}
    return schema

@tool
def schema_generator(query):
    """
    Generates a CSV-based schema from a user query using LLM.

    Args:
        query (str): A user query describing the data structure needed.

    Returns:
        dict: A schema dictionary with generated column names as keys and empty lists as values.

    Note:
        Creates up to 5 columns by default unless otherwise specified.
        Uses LLM to generate descriptive, simple column names.
    """
    formatting = "The above statement is given by the user. Please create a single .csv-based schema by following the points below:\n"

    formatting += "1. Only create the schema, no additional text or statement.\n"

    formatting += "2. Keep the schema simple, avoid complex column names.\n"

    formatting+=  "3. please only generate 5 schema if not mentioned.\n"

    formatting += "4. For example, if the user provides a statement like: 'Generate data for students getting placements from IIT Bombay,' the response should be:\n"

    formatting += "Student Name, Student Roll Number, Student Branch, Student Year, Student Placement Status, Student Company Name, Student Package, Student Location, Student Role\n"

    formatting += "Follow the above example but remember above is not actual schema you have to provide the schema depending on the user prompt.\n"

    formatting+=  "5. please only generate schema no notes or anything.\n"

    output=consume_llm_api(query+"\n"+formatting)

    return schema_formatter(output)

@tool
def sorting(data_dict):
    """
    Sorts a list of dictionaries by the number of non-empty values in descending order.

    Args:
        data_dict (list): A list of dictionaries to be sorted.

    Returns:
        dict: A dictionary mapping dictionary identifiers to their non-empty value counts, sorted in descending order.
    """
    new_dict={str(i):0 for i in data_dict}

    for i in data_dict:
        for j in i:
            if len(i[j])!=0:
                new_dict[str(i)] +=1
    new_dict=[(new_dict[i],i) for i in new_dict]
    new_dict.sort(reverse=True)
    new_dict={i[-1]:i[0] for i in new_dict}
    return new_dict

@tool
def process_data(query):
    """
    Processes a user query to generate and validate data according to a generated schema.

    Args:
        query (str): The user query describing the data to be generated.

    Yields:
        dict: Verified and formatted data dictionaries matching the schema.

    Note:
        Generates schema from query, retrieves Wikipedia data, creates actual values,
        sorts by completeness, and verifies results.
    """
    formatting = "The above statement is given by the user. Please create a single .csv-based schema by following the points below:\n"
    formatting += "1. Only create the schema, no additional text or statement.\n"
    formatting += "2. Keep the schema simple, avoid complex column names.\n"
    formatting+=  "3. please only generate 5 schema if not mentioned.\n"
    formatting += "4. For example, if the user provides a statement like: 'Generate data for students getting placements from IIT Bombay,' the response should be:\n"
    formatting += "Student Name, Student Roll Number, Student Branch, Student Year, Student Placement Status, Student Company Name, Student Package, Student Location, Student Role\n"
    formatting += "Follow the above example but remember above is not actual schema you have to provide the schema depending on the user prompt.\n"
    formatting+=  "5. please only generate schema no notes or anything.\n"
    print("Query:",query)
    output=consume_llm_api(query+"\n"+formatting)

    schema = {i:[] for i in output.split(",")}
    textual_value=relevent_value(str(schema).lower(),3)
    html_pages = textual_value[1]
    textual_value = textual_value[0]
    data_dict =[j for j in actual_value(textual_value,schema)]
    for j in sorting(data_dict):
        try:
            # Convert string to dictionary
            dummy_value = eval(j)
            
            # Process dictionary values
            for key in dummy_value:
                while len(dummy_value[key]) >= 2:
                    dummy_value[key].pop(0)
            
            # Format dictionary
            formatted = dictionary_formatting(dummy_value)
            print(formatted)
            # Verify and store result
            verification_result = verification(formatted) if formatted else None
            
            yield verification_result
            
        except Exception as e:
            print(f"Error processing dictionary {j}: {e}")

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
def user_query_understading(user_query, schema, data):
    """
    Evaluates if generated data is relevant to the user query and schema.

    Args:
        user_query (str): The original user query.
        schema (dict): The schema used for data generation.
        data (dict): The generated data to be evaluated.

    Returns:
        str: 'True' if data is relevant to query and schema, 'False' otherwise.
    """
    understanding_query = f"schema : {schema} \n data: {data} \n user_query:{user_query} \n as you see return True if the data is relevant to the user query and schema provided else return False (remember to return true/False only no other text)"
    keywords = consume_llm_api(understanding_query)
    return keywords

@tool
def fast_data_generation(query):
    """
    Quickly generates data based on a query/schema without additional steps. Only to be used when user is sure about the data and just want to generate it without any validation or checking.

    Args:
        query (str): The query or schema to generate data for.

    Returns:
        str: Generated data in the form of a list of dictionaries.
    """
    fast_gen_data = consume_llm_api(f"Generate data based on the following schema and query. \n schema: {query} \n query: {query} \n Please generate data in the form of a list of dictionaries where each dictionary follows the schema provided. \n Only generate data without any additional text or explanation.")

    return fast_gen_data


@tool
def fast_gen_error_remover(input_data):
    """
    Removes parsing errors from LLM-generated data by incrementally fixing it.

    Args:
        input_data (str): The potentially malformed data string.

    Returns:
        dict or list: The corrected and evaluated data structure.

    Note:
        Iteratively removes characters from the end until valid Python syntax is achieved.
    """
    Error=True
    while Error:
        try:
            input_data=eval(input_data)
            Error=False
        except:
            input_data = input_data[:-1]+"]"
    return input_data

@tool
def rows_limit(prompt):
    """
    Checks if a prompt requests more than 100 rows of data generation.

    Args:
        prompt (str): The user prompt to be analyzed.

    Returns:
        str: 'True' if data generation is within 100 rows limit, 'False' if exceeding.
    """
    instruction = f"user prompt:{prompt} \n If the above mentioned prompt is asking to generate data more than 100 rows then must return False else True. Please return only True or False no other text."
    result = consume_llm_api(instruction)   
    return result


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


