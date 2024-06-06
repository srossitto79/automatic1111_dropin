#build a flask api that mimic automatic111 api for stable diffusion on 7860 port, use huggingface pipelines to accomplish the generation tasks, assure to load and unload the model for every generation queue freeing up the memory when finished
import os
import json
import time
import PIL
import torch
from flask import Flask, request, jsonify
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoModelForTokenClassification
from transformers import AutoModelForQuestionAnswering
from transformers import AutoModelForMaskedLM
from transformers import AutoModelForNamedEntityRecognition
from transformers import AutoModelForTextClassification
from transformers import AutoModelForImageClassification
from transformers import AutoModelForImageSegmentation
from transformers import AutoModelForImageFeatureExtraction
from transformers import AutoModelForImageGeneration
from transformers import AutoModelForVideoClassification
from transformers import AutoModelForVideoFeatureExtraction
from transformers import AutoModelForVideoSegmentation
from transformers import AutoModelForVideoGeneration
from transformers import AutoModelForAudioClassification
from transformers import AutoModelForAudioFeatureExtraction
from transformers import AutoModelForAudioSegmentation
from transformers import AutoModelForAudioGeneration
import requests

app = Flask(__name__)
app.config["DEBUG"] = True

DIFFUSION_MODELS_DIR = 'models'
if not os.path.exists(DIFFUSION_MODELS_DIR):
    os.makedirs(DIFFUSION_MODELS_DIR)

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

def download_model(safetensors_file_url):    
    response = requests.get(safetensors_file_url)
    filename = safetensors_file_url.split("/")[-1]
    with open(f"{DIFFUSION_MODELS_DIR}/{filename}", "wb") as f:
        f.write(response.content)

def sd_img_pipeline(prompt : str, width : int=512, height : int=512, denoise_strength : float=0.75, 
                        guidance_from_last_attachment = True, guidance_scale: float=7.5, steps: int=30, 
                        model_id : str= 'dreamshaper_8.safetensors', negative_prompt="", init_image : PIL.Image = None):
    
    if not os.path.exists("./generated_images"):
        os.makedirs("./generated_images", exist_ok=True)

    images = []         
    text2image_gen_pipe = None
    image2image_gen_pipe = None
    if init_image is None:
        if text2image_gen_pipe is None:
            if torch.cuda.is_available():
                print(f"Loading Stable model {model_id} into GPU")
                text2image_gen_pipe = StableDiffusionPipeline.from_single_file(f"{DIFFUSION_MODELS_DIR}/" + model_id, torch_dtype=torch.float16, verbose=True, use_safetensors=True)
                text2image_gen_pipe = text2image_gen_pipe.to("cuda")                   
            else:
                print(f"Loading Stable model {model_id} into CPU")
                text2image_gen_pipe = StableDiffusionPipeline.from_single_file(f"{DIFFUSION_MODELS_DIR}/" + model_id, torch_dtype=torch.float32, verbose=True, use_safetensors=True)
                text2image_gen_pipe = text2image_gen_pipe.to("cpu")                   
        print("generating image from prompt...")
        images = text2image_gen_pipe(prompt, width=width, height=height, num_inference_steps=steps, negative_prompt=negative_prompt).images
    else:
        if image2image_gen_pipe is None:
            if torch.cuda.is_available():
                print(f"Loading Stable model {model_id} into GPU")
                image2image_gen_pipe = StableDiffusionImg2ImgPipeline.from_single_file(f"{DIFFUSION_MODELS_DIR}/" + model_id, torch_dtype=torch.float16, verbose=True, use_safetensors=True)
                image2image_gen_pipe = image2image_gen_pipe.to("cuda")   
            else:
                print(f"Loading Stable model {model_id} into CPU")
                image2image_gen_pipe = StableDiffusionImg2ImgPipeline.from_single_file(f"{DIFFUSION_MODELS_DIR}/" + model_id, torch_dtype=torch.float32, verbose=True, use_safetensors=True)
                image2image_gen_pipe = image2image_gen_pipe.to("cpu")                   
        print("generating image from prompt+image...")#
        init_image = init_image.convert("RGB")
        images = image2image_gen_pipe(prompt, image=init_image, width=width, height=height, 
                                            strength=denoise_strength, guidance_scale=guidance_scale, 
                                            num_inference_steps=steps, negative_prompt=negative_prompt).images
        
    paths = []
    for image in (images if images is not None else []):
        # Create a filename based on the current date and time
        filename = f'image_{datetime.now().strftime("%Y%m%d%H%M%S")}{(len(paths)+1)}.jpg'
        # Save the image to the specified path
        file_path = f"./generated_images/{filename}"
        image.save(file_path)
        paths.append(file_path)
    return f"Generated images from prompt \"{prompt}\" saved to files: {', '.join(paths)}"        

@app.route('/sdapi/v1/txt2img', methods=['POST'])
def txt2img():
    #TODO add all parameters supported by automatic111 api
    payload = request.json
    prompt = payload['prompt']
    steps = payload['steps'] if 'steps' in payload else 25
    model_name = payload['model_name'] if 'model_name' in payload else 'dreamshaper_8.safetensors'
    
    #{ "images": ["base_64_enc_image", "..."] }
    result = sd_img_pipeline(prompt, steps=steps, model_id=model_name)
    return jsonify(result)

@app.route('/sdapi/v1/img2img', methods=['POST'])
def img2img():
    #TODO add all parameters supported by automatic111 api
    payload = request.json
    prompt = payload['prompt']
    steps = payload['steps'] if 'steps' in payload else 25
    model_name = payload['model_name'] if 'model_name' in payload else 'dreamshaper_8.safetensors'
    image = PIL.Image.open(request.files['image'])

    #{ "images": ["base_64_enc_image", "..."] }
    result = sd_img_pipeline(prompt, steps=steps, model_id=model_name, init_image=image)
    
    return jsonify(result)

#TODO add all endpoint methods supported by automatic111 api

if __name__ == '__main__':
    host = os.getenv('AUTOMATIC111_HOST', '0.0.0.0')
    port = os.getenv('AUTOMATIC111_PORT', 7860)
    app.run(host=host, port=port)