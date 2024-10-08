import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# Initialize the model
@st.cache_resource
def load_models():
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.to("cuda")
    return pipe

pipe = load_models()

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, use_column_width=True)
    
    # Convert the uploaded PIL image to a NumPy array
    np_image = np.array(input_image)
    # Apply Canny edge detection
    canny_image = cv2.Canny(np_image, 100, 200)
    # Increase the dimensions to convert the single-channel image to a three-dimensional array
    canny_image = canny_image[:, :, None]
    # Repeat the process three times to create an RGB image
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    # Convert the NumPy array back to a PIL image
    canny_pil_image = Image.fromarray(canny_image)

    user_prompt = st.text_input("Enter the prompt")

    # Generate a new image using the edge-detected image
    if st.button("Generate image"):
        with st.spinner("Generating..."):
            generator = torch.manual_seed(0)
            generated_image = pipe(
                prompt=user_prompt, 
                image=canny_pil_image, 
                num_inference_steps=20, 
                generator=generator
            ).images[0]
            
            st.image(generated_image, use_column_width=True)