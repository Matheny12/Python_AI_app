from ai_models import AIModel
from typing import List, Dict, Optional, Generator
from gpt4all import GPT4All
from diffusers import StableDiffusionPipeline
import io
import os
import streamlit as st

class BartBotModel(AIModel):
    @st.cache_resource
    def _get_llm(_self):
        model_name = "llama-3.2-1b-instruct-q4_k_m.gguf"
        return GPT4All(model_name=model_name, allow_download=True)
    
    def __init__(self):
        self.llm = self._get_llm()
        self.image_model = None

    def generate_response(self, messages: List[Dict], system_prompt: str, file_data: Optional[Dict] = None) -> Generator:        
        with self.llm.chat_session(system_prompt):
            user_input = messages[-1]["content"]   
            response_generator = self.llm.generate(
                user_input, 
                max_tokens=1024, 
                streaming=True
            )
            for token in response_generator:
                yield token

    def generate_image(self, prompt: str) -> bytes:
        if self.image_model is None:
            self.image_model = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                use_safetensors=True
            ).to("cpu")
        
        image = self.image_model(prompt, num_inference_steps=15).images[0]
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()