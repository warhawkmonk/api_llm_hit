import torch
from diffusers import AutoPipelineForInpainting
from flask import Flask, jsonify, request,Response
from PIL import Image
import base64
import io
from PIL import Image
import numpy as np

from langchain_ollama import OllamaLLM
app = Flask(__name__)

# Load model at startup
def load_model():
    return AutoPipelineForInpainting.from_pretrained(
        "gubli/",
        torch_dtype=torch.float16
    ).to("cuda")

pipeline_ = load_model()

def llm_text_response():
    llm = OllamaLLM(model="steamdj/llama3.1-cpu-only",num_thread=14)
    return llm.stream
def llm_text_response_invoke():
    llm = OllamaLLM(model="llama3:latest",num_thread=14)
    return llm.invoke
    
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
    # Run the inpainting pipeline
    # pipeline_ = load_model()
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
            # mask_image = Image.fromarray(np.array(masked_img_base64,dtype=np.uint8))
            mask_image = np.array(masked_img_base64,dtype=np.uint8)
            # init_image = base64_to_image(initial_img_base64)
            # mask_image = base64_to_image(masked_img_base64)
    
            # Generate inpainted image
            output_image = model_out_put(init_image=init_image, mask_image=mask_image, prompt=prompt, negative_prompt=negative_prompt)
            # prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image
            output_image = numpy_to_list(np.array(output_image,dtype=np.uint8))
            # Convert output image to base64 for response
                
            
            return jsonify({"img": output_image})
        elif "extension" in data:
            prompt = data.get("prompt", "")
            if not prompt:
                return jsonify({"error": "Prompt is required"}), 400
            llm_stream = llm_text_response_invoke()(prompt)
            return jsonify({"text": llm_stream})
            
        else:
            prompt = data.get("prompt", "")
            if not prompt:
                return jsonify({"error": "Prompt is required"}), 400
    
            # Call the LLM response generator with the prompt
            llm_stream = llm_text_response()(prompt)
    
            # Create a generator to stream the data
            def generate():
                for chunk in llm_stream:
                    
                    yield chunk
    
            return Response(generate(), content_type='text/event-stream')            
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True,port=5000)
