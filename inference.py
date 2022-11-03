from diffusers import StableDiffusionPipeline
import torch

model_id = "./output"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

for i in range(50):
    prompt = "A photo of sks dog in a bucket"
    image = pipe(prompt, num_inference_steps=200, guidance_scale=7.5).images[0]

    image.save("./dogs/dog-bucket-"+str(i)+".png")
