from ai_models import AIModel
from typing import List, Dict, Optional, Generator
from gpt4all import GPT4All
import io
import os
import streamlit as st
import requests
import time
from PIL import Image as PILImage
import base64

try:
    from AnimateDiff import AnimateDiffGenerator, SimpleAnimateDiff
    ANIMATEDIFF_AVAILABLE = True
except ImportError as e:
    ANIMATEDIFF_AVAILABLE = False
    print(f"[WARNING] AnimateDiff not available: {e}")

class BartBotModel(AIModel):
    @st.cache_resource
    def _get_llm(_self):
        model_name = "Llama-3.2-1B-Instruct-Q4_0.gguf"
        return GPT4All(model_name=model_name, allow_download=True)
    
    def __init__(self):
        self.llm = self._get_llm()
        self.api_key = st.secrets.get("GEMINI_KEY") or os.getenv("GEMINI_KEY")
        from google import genai
        self.client = genai.Client(api_key=self.api_key)
        self.animatediff = None

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
        from huggingface_hub import InferenceClient
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            raise Exception("HF_TOKEN missing for image generation.")
        
        client = InferenceClient(api_key=hf_token)
        image = client.text_to_image(prompt, model="black-forest-labs/FLUX.1-schnell")
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()
    
    def generate_video(self, prompt: str, image_data: bytes = None) -> bytes:
        if not image_data:
            raise ValueError("Please upload an image first.")

        if not ANIMATEDIFF_AVAILABLE:
            raise Exception(
                "AnimateDiff is not available. Video generation requires AnimateDiff.py and dependencies."
            )

        try:
            if self.animatediff is None:
                print("[BartBotModel] Initializing AnimateDiff...")
                use_simple = os.getenv("USE_SIMPLE_ANIMATEDIFF", "true").lower() == "true"
                
                if use_simple:
                    self.animatediff = SimpleAnimateDiff()
                    print("[BartBotModel] Using SimpleAnimateDiff (faster)")
                else:
                    self.animatediff = AnimateDiffGenerator()
                    print("[BartBotModel] Using AnimateDiffGenerator (better quality)")
            
            print(f"[BartBotModel] Starting local video generation")
            print(f"[BartBotModel] Prompt: '{prompt}'")
            
            num_frames = int(os.getenv("ANIMATEDIFF_FRAMES", "16"))
            fps = int(os.getenv("ANIMATEDIFF_FPS", "8"))
            
            video_data = self.animatediff.generate_from_image(
                image_data=image_data,
                prompt=prompt if prompt else "smooth subtle motion, high quality",
                num_frames=num_frames,
                fps=fps
            )
            
            print(f"[BartBotModel] Video generated: {len(video_data)} bytes")
            
            if len(video_data) < 1000:
                raise Exception(f"Video file is too small ({len(video_data)} bytes), likely corrupted")
            
            return video_data
            
        except Exception as e:
            error_msg = f"Video generation failed: {e}"
            print(f"[ERROR] {error_msg}")
            raise Exception(error_msg)