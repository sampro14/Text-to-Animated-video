from flask import Flask, request, jsonify, Response, stream_with_context
from conditional_ldm import TextToImage
from io import BytesIO
import base64
import torch
import time
import json
from uuid import uuid4
from multiprocessing import Process
import os
import shutil
from PIL import Image
import numpy as np

device = torch.device("mps")
diffusion_model_path = 'models/cldm_models/unet/39600' # Your path here

default_inf_steps = 500
app = Flask(__name__)

def load_model(model_path):
    model = TextToImage.load(model_path)
    model.to(device)
    return model


def convert_base64_to_img(base64_str):
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_bytes))
    return image

def base64_to_arr(base64_str):
    decoded_bytes = base64.b64decode(base64_str)
    reconstructed_array = np.frombuffer(decoded_bytes, dtype=np.float32) 
    return reconstructed_array

def image_to_base64(image):
    # Convert PIL Image to base64-encoded string
    buffered = BytesIO()
    image.save(buffered, format='PNG')
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return image_base64

def numpy_to_base64(arr):
    arr_bytes = arr.tobytes()
    base64_string = base64.b64encode(arr_bytes).decode('utf-8')
    return base64_string

def generate_images(job_id, prompt, inf_steps, n_images, stream_freq=50, latent_input_base64=None, start_timestep=0, model_path=diffusion_model_path, save_dir=None):
    model = load_model(model_path)
    def create_response(img, details):
        image_base64 = [image_to_base64(i) for i in img]
        latents = details["latent"]
        latents_base64 = [numpy_to_base64(i) for i in latents]
        response = [
            {
                "t": details["t"] + 1,
                "total_t": details["total_t"],
                "img": i,
                "latent": l
            }
            for i, l in zip(image_base64, latents_base64)
        ]
        return response
    
    def save_images(images, details):
        if save_dir is None:
            print("Save Dir is None??")
            return
        t = details['t'] + 1
        for i, img in enumerate(images):
            img.save(f"{save_dir}/{i:02d}_{t:04d}.png")
    def do_collage(images, num_rows, num_cols):
        image_width, image_height = images[0].size
        # Create a new image with a size big enough to hold all images
        collage_width = image_width * num_cols
        collage_height = image_height * num_rows
        collage_image = Image.new('RGB', (collage_width, collage_height))

        # Paste each image into the collage
        for index, image in enumerate(images):
            x_offset = (index % num_cols) * image_width
            y_offset = (index // num_cols) * image_height
            collage_image.paste(image, (x_offset, y_offset))
        return collage_image

    def collage_images():
        image_files = [f for f in os.listdir(save_dir) if f.endswith('.png')]

        # Sort the images (you can customize this sorting if needed)
        image_files.sort()

        # Determine the number of images and the layout of the collage
        num_images = len(image_files)
        num_cols = (num_images) // n_images

        num_rows = (num_images + num_cols - 1) // num_cols

        # Open all images and get their sizes
        images = [Image.open(os.path.join(save_dir, file)) for file in image_files]
        collage_image = do_collage(images, num_rows, num_cols)
        # Save the collage image
        collage_image.save(f'{save_dir}/_{prompt}_consolidated.png')

    def collage_final_images():
        image_files = [f for f in os.listdir(save_dir) if f.endswith('0500.png')]

        # Sort the images (you can customize this sorting if needed)
        image_files.sort()

        # Determine the number of images and the layout of the collage
        num_images = len(image_files)
        num_cols = 4

        num_rows = (num_images + num_cols - 1) // num_cols

        # Open all images and get their sizes
        images = [Image.open(os.path.join(save_dir, file)) for file in image_files]
        collage_image = do_collage(images, num_rows, num_cols)

        # Save the collage image
        collage_image.save(f'{save_dir}/_{prompt}_final.png')
    input_latent = None
    if (latent_input_base64 is not None) and (latent_input_base64 != ""):
        input_latent = base64_to_arr(latent_input_base64)
        input_latent = np.array(input_latent)
        input_latent = np.reshape(input_latent, model.latent_shape)

    prompts = [prompt for _ in range(n_images)]
    json.dump({"image": None, "details": None, "finished": False},
            open(f"{save_dir}/out.json", "w"))
    
    for (img, details) in model.run_pipeline(prompts, num_inference_steps=inf_steps, stream_freq=stream_freq, image=input_latent, start_timestep_factor=start_timestep):
        response = create_response(img, details)
        save_images(img, details)
        json.dump({"response": response, "finished": False}, open(f"{save_dir}/out.json", "w"))
        t = details["t"] + 1
        json.dump({"response": response, "finished": False}, open(f"{save_dir}/out_{t}.json", "w"))
            
    
    response = create_response(img, details)
    save_images(img, details)
    collage_images()
    collage_final_images()
    json.dump({"response": response, "finished": True}, open(f"{save_dir}/out.json", "w"))

@app.route('/jobs/<job_id>/', methods=["GET"])
def get_images(job_id):
    filename = f"jobs/{job_id}/out.json"
    if os.path.exists(filename):
        try:
            return jsonify(json.load(open(filename)))
        except:
            pass
    return jsonify({"finished": False})


@app.route('/text_to_image', methods=["POST"])
def start_job():
    req = request.get_json()
    model_path = req.get("model_path", diffusion_model_path)
    prompt = req.get("prompt", "")
    n_images = req.get("n_images", 4)
    inf_steps = req.get("n_inf_steps", 500)
    stream_freq = req.get("stream_freq", 50)
    latent = req.get("latent", None) 
    start_timestep = req.get("start_timestep", 0)  
    print(prompt, inf_steps, n_images, stream_freq, latent, start_timestep)
    # model_id =  model_path.split("/")[1].split("/")[0]
    job_id = prompt.replace(" ", "_") + "_" + uuid4().hex[-4:]

    save_dir = f"jobs/{job_id}"
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)

    task_cb = Process(target=generate_images, args=(job_id, prompt, inf_steps, n_images, stream_freq, latent, start_timestep, model_path, save_dir))
    task_cb.start()

    return jsonify({"job_id": job_id})


if __name__ == '__main__':
    app.run(debug=True)