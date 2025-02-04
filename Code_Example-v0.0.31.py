# This file contains code that is derived from Stability AI's software products, 
# which are licensed under the Stability AI Non-Commercial Research Community License Agreement.
# Copyright (c) Stability AI Ltd. All Rights Reserved.
#
# The original work is provided by Stability AI and is available under the terms of the 
# Stability AI Non-Commercial Research Community License Agreement, dated November 28, 2023.
# For more information, see https://stability.ai/use-policy.
import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
import os
import uuid  # Import the uuid library

# Set device and data type
device = "cuda"
dtype = torch.bfloat16

# Load models
prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=dtype).to(device)
decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", torch_dtype=dtype).to(device)

# Use a relative path for the output directory
output_directory = "./Output"
# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Main loop for generating images
continue_generating = "yes"
while continue_generating.lower() in ["yes", "y"]:
    # User inputs for customization
    prompt = input("Enter your prompt: ")
    height = int(input("Enter the image height (e.g., 1024): "))
    width = int(input("Enter the image width (e.g., 1024): "))
    negative_prompt = input("Enter your negative prompt, if any (or press enter to skip): ")
    guidance_scale = float(input("Enter the guidance scale (e.g., 4.0): "))
    num_images_per_prompt = int(input("Enter the number of images per prompt (e.g., 2): "))
    
    with torch.cuda.amp.autocast(dtype=dtype):
        prior_output = prior(
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
        )
        decoder_output = decoder(
            image_embeddings=prior_output.image_embeddings,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=0.0,
            output_type="pil",
        ).images

    # Display or save images
    for i, image in enumerate(decoder_output):
        # Optional: Display the image
        # image.show()
        
        # Generate a unique filename using a UUID
        unique_filename = f"generated_image_{uuid.uuid4()}.png"
        save_path = os.path.join(output_directory, unique_filename)
        image.save(save_path)
        print(f"Saved: {save_path}")

    # Ask user if they want to generate more images
    continue_generating = input("Do you want to generate more images? (yes/no): ")

print("Thank you for using the image generator!")
