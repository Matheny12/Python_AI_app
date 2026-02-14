from ai_models import AIModel
from typing import List, Dict, Optional, Generator
from gpt4all import GPT4All
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
    
    def generate_video(self, prompt: str, image_data: bytes = None) -> bytes:
        import time
        import io
        import tempfile
        import os
        import requests
        from PIL import Image as PILImage
        from google import genai
        from google.genai import types
        
        if not image_data:
            raise NotImplementedError(
                "Text-to-video is not supported. Please upload an image first, then use /video to animate it."
            )
        
        gemini_key = os.getenv("GEMINI_KEY") or st.secrets.get("GEMINI_KEY")
        if not gemini_key:
            raise Exception("GEMINI_KEY not found. Please add your Gemini API key to Streamlit secrets.")
        
        try:
            client = genai.Client(api_key=gemini_key)
            image = PILImage.open(io.BytesIO(image_data))
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                image.save(tmp_file.name, format='PNG')
                tmp_path = tmp_file.name
            
            try:
                uploaded_image = client.files.upload(file=tmp_path)
                
                operation = client.models.generate_videos(
                    model="veo-3.1-fast-generate-preview",
                    prompt=prompt if prompt else "animate this image naturally with smooth motion",
                    image=uploaded_image,
                    config=types.GenerateVideosConfig(
                        aspect_ratio="16:9",
                        duration_seconds=8,
                        resolution="720p"
                    )
                )
                
                max_wait = 180
                elapsed = 0
                while not operation.done and elapsed < max_wait:
                    time.sleep(5)
                    elapsed += 5
                    operation = client.operations.get(operation)
                
                if not operation.done:
                    raise Exception("Video generation timed out after 3 minutes")
                
                if operation.result and operation.result.generated_videos:
                    video_uri = operation.result.generated_videos[0].video.uri
                    
                    if video_uri.startswith("gs://"):
                        raise Exception(
                            "Video stored in Google Cloud Storage. "
                            "Please set up a GCS bucket and configure output_gcs_uri in the config."
                        )
                    
                    video_response = requests.get(video_uri)
                    return video_response.content
                
                raise Exception("No video generated")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
        except Exception as e:
            raise Exception(f"Failed to generate video: {str(e)}")