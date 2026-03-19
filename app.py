import torch
torch.cuda.empty_cache()
from diffusers import AutoPipelineForInpainting
from flask import Flask, jsonify, request,Response
from PIL import Image
import base64
import io
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from transformers import pipeline
from langchain_ollama import OllamaLLM
model = SentenceTransformer("all-mpnet-base-v2")
app = Flask(__name__)

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

# Load model at startup
def load_model():
    # import gc
    # gc.collect()
    # torch.cuda.empty_cache()
    return AutoPipelineForInpainting.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder-inpaint",
        torch_dtype=torch.float16
    ).to("cuda")

pipeline_ = load_model()
pipeline_.enable_model_cpu_offload()



# Use a small Qwen model (CPU‑friendly)
HF_MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"  # or "Qwen/Qwen3.5-0.8B"

_TEXT_GENERATOR = pipeline(
    "text-generation",
    model=HF_MODEL_NAME,
    device_map="cuda",
    return_full_text=False,
    padding=True,
    truncation=True,
)


def llm_text_response():
    """Return a generator that mimics .stream()."""
    def stream_fn(prompt: str, **kwargs):
        gen = _TEXT_GENERATOR(
            prompt,
            max_new_tokens=kwargs.pop("max_tokens", 256),
            **kwargs
        )
        yield gen[0]["generated_text"]

    return stream_fn


def llm_text_response_invoke():
    """Return an invoke‑style callable."""
    def invoke_fn(prompt: str, **kwargs):
        gen = _TEXT_GENERATOR(
            prompt,
            max_new_tokens=kwargs.pop("max_tokens", 256),
            **kwargs
        )
        return gen[0]["generated_text"]

    return invoke_fn
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
        elif "only_prompt" in data:
            print("hi",data["only_prompt"])
            output_image = pipeline_(data["only_prompt"]).images[0]
            print("working")
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
        elif "api_key" in data:
            api_key = data.get("api_key", "")
            pc = Pinecone(api_key=api_key)
            index_name = "quickstart"
            index = pc.Index(index_name)

            value = index.query (
                id="lorum",
                top_k=1,
                include_metadata=True
                )
            
            value = value['matches'][0]['metadata']['string']
            output = llm_text_response_invoke()(value)
            index.upsert(
                vectors=[
                    {
                        "id": value[:512], 
                        "values": [float(i) for i in list(model.encode(value))],  
                        "metadata": {"string":str(output),"prompt":value}
                        
                    }
                ]
            )
            return {"status":True}            
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
    import os
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.getenv('PORT', 6000))
    app.run(debug=debug, port=port, host='0.0.0.0')