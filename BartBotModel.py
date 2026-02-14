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
        model_name = "Llama-3.2-1B-Instruct-Q4_0.gguf"
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
        import os
        from huggingface_hub import InferenceClient
        import io
        import time

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            raise Exception("HF_TOKEN not found. Please add your Hugging Face token to Streamlit secrets or environment variables for image generation.")

        models_to_try = [
            ("stabilityai/stable-diffusion-2-1", {"num_inference_steps": 20}),
            ("black-forest-labs/FLUX.1-schnell", {"num_inference_steps": 4}),
            ("ByteDance/SDXL-Lightning", {"num_inference_steps": 4}),
        ]


        client = InferenceClient(api_key=hf_token)
        last_error = None

        for model_name, params in models_to_try:
            try:
                image = client.text_to_image(
                    prompt,
                    model=model_name,
                    guidance_scale=7.5,
                    width=512,
                    height=512,
                    **params
                )
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                return img_byte_arr.getvalue()
            except Exception as e:
                last_error = str(e)
                if "timeout" in str(e).lower() or "cold state" in str(e).lower():
                    continue
                continue
        raise Exception(f"Failed to generate image after trying multiple models. Last error: {last_error}")



    def generate_video(self, prompt: str) -> bytes:
        import os
        from huggingface_hub import InferenceClient
        
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            raise Exception("HF_TOKEN not found. Please add your Hugging Face token to Streamlit secrets or environment variables for video generation.")
        
        try:
            client = InferenceClient(api_key=hf_token)
            video_bytes = client.text_to_video(
                prompt,
                model="Lightricks/LTX-Video-0.9.8-13B-distilled",
                num_frames=121,
                guidance_scale=3.0,
                num_inference_steps=30
            )
            
            return video_bytes
            
        except Exception as e:
            raise Exception(f"Failed to generate video: {str(e)}")
