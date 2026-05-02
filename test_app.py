import torch
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage  # For input format
import wikipedia
import wikipediaapi
from utility import (
    consume_llm_api, relevent_value, construction_edit,
    dictionary_check, actual_value, context_data_relevancy, agent_work_result,
    full_alignment, query_formatting, missing_value_completion, verification,
    agent_data_prep, dictionary_formatting, schema_formatter, schema_generator,
    sorting, process_data, query_generator, user_query_understading,
    fast_data_generation, fast_gen_error_remover, rows_limit, query_template
)
torch.cuda.empty_cache()
from diffusers import AutoPipelineForInpainting
from flask import Flask, jsonify, request,Response
from PIL import Image
import numpy as np
from transformers import pipeline
app = Flask(__name__)

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

# Load model at startup
def load_model():
    # return AutoPipelineForInpainting.from_pretrained(
    #     "kandinsky-community/kandinsky-2-2-decoder-inpaint",
    #     torch_dtype=torch.float16,token="ADD your Hugging Face token here"
    # ).to("cuda")
    return {}

pipeline_ = load_model()
# pipeline_.enable_model_cpu_offload()

# Initialize tools list with actual tool functions from utility
tools = [
    consume_llm_api, relevent_value, construction_edit,
    dictionary_check, actual_value, context_data_relevancy, agent_work_result,
    full_alignment, query_formatting, missing_value_completion, verification,
    agent_data_prep, dictionary_formatting, schema_formatter, schema_generator,
    sorting, process_data, query_generator, user_query_understading,
    fast_data_generation, fast_gen_error_remover, rows_limit, query_template
]

# Initialize agent with OpenAI
llm = ChatOpenAI(
    model="google/gemma-4-e4b",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    temperature=0.7
)
agent = create_react_agent(llm, tools)


def numpy_to_list(array):

    current=[]
    for value in array:
        if isinstance(value,type(np.array([]))):
            result=numpy_to_list(value)
            current.append(result)
        else:
            
            current.append(int(value))
    return current

def model_out_put(init_image, mask_image, prompt, negative_prompt):
    image = pipeline_(
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        image=init_image, 
        mask_image=mask_image
    ).images[0]
    return image

@app.route('/api/llm-response', methods=['POST'])
def normal_response():
    try:
        
        # Parse request JSON
        data = request.get_json()
        if "initial_img" in data:
            
            data = request.get_json()
            prompt = data.get("prompt", "")
            initial_img_base64 = data.get("initial_img", "")
            masked_img_base64 = data.get("masked_img", "")
            negative_prompt = data.get("negative_prompt", "")
            
            init_image = Image.fromarray(np.array(initial_img_base64,dtype=np.uint8))

            mask_image = np.array(masked_img_base64,dtype=np.uint8)
            output_image = model_out_put(init_image=init_image, mask_image=mask_image, prompt=prompt, negative_prompt=negative_prompt)

            output_image = numpy_to_list(np.array(output_image,dtype=np.uint8))
            return jsonify({"img": output_image})
        
        
        elif "only_prompt" in data:
            output_image = pipeline_(data["only_prompt"]).images[0]
            output_image = numpy_to_list(np.array(output_image,dtype=np.uint8))
            return jsonify({"img": output_image})
        


        elif "extension" in data:
            prompt = data.get("prompt", "")
   
            if not prompt:
                return jsonify({"error": "Prompt is required"}), 400
            # Use agent instead of text generator
            response = llm.invoke(prompt)
            agent_response = response.content
            return jsonify({"text": agent_response})
         
        else:
            prompt = data.get("prompt", "")
            if not prompt:
                return jsonify({"error": "Prompt is required"}), 400
    
            # Use agent to process the prompt
            response = agent.invoke({"messages": [HumanMessage(content=prompt)]})
            agent_response = response["messages"][-1].content
            
            # Return agent response
            return jsonify({"text": agent_response})            
    except ValueError as ve:
        print("ValueError:", ve)
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    import os
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.getenv('PORT', 6000))
    app.run(debug=debug, port=port, host='0.0.0.0')