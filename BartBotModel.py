from ai_models import AIModel
from typing import List, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
from PIL import Image
import io
import os

class BartBotModel(AIModel):
    def __init__(self, model_path: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        hf_token = os.getenv("HF_TOKEN")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=hf_token,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token,
            trust_remote_code=True
        )

        self.image_model = None
        self.vision_model = None

    def generate_response(self, messages: List[Dict], system_prompt: str, file_data: Optional[Dict] = None) -> str:
        prompt = self._format_messages(messages, system_prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Bartholemew:")[-1].strip()
        if not response:
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def _format_messages(self, messages: List[Dict], system_prompt: str) -> str:
        prompt_parts = []

        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}\n")

        for msg in messages:
            role = "User" if msg["role"] == "user" else "Bartholemew"
            content = msg["content"]
            if not content.startswith("IMAGE_DATA:"):
                prompt_parts.append(f"{role}: {content}\n")

        prompt_parts.append("Bartholemew:")
        return "\n".join(prompt_parts)
    
    def generate_image(self, prompt: str) -> bytes:
        if self.image_model is None:
            hf_token = os.getenv("HF_TOKEN")
            self.image_model = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16,
                token=hf_token
            ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        image = self.image_model(
            prompt,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()