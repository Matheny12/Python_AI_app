from ai_models import AIModel
from typing import List, Dict, Optional, Generator
from gpt4all import GPT4All
import io
import os
import streamlit as st
import requests
import time
from PIL import Image as PILImage

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
        from google.genai import types
        if not image_data:
            raise ValueError("Please upload an image first.")

        try:
            try:
                img = PILImage.open(io.BytesIO(image_data))
                img_format = img.format.lower() if img.format else 'png'
                
                print(f"[DEBUG] Image format detected: {img_format}, Size: {img.size}")
                
                mime_type_map = {
                    'jpeg': 'image/jpeg',
                    'jpg': 'image/jpeg',
                    'png': 'image/png',
                    'gif': 'image/gif',
                    'webp': 'image/webp'
                }
                mime_type = mime_type_map.get(img_format, 'image/png')
                
                if img.mode not in ('RGB', 'RGBA'):
                    print(f"[DEBUG] Converting image from {img.mode} to RGB")
                    img = img.convert('RGB')
                    buffered = io.BytesIO()
                    img.save(buffered, format='JPEG')
                    image_data = buffered.getvalue()
                    mime_type = 'image/jpeg'
                
                print(f"[DEBUG] Final MIME type: {mime_type}, Data size: {len(image_data)} bytes")
                
            except Exception as e:
                print(f"[ERROR] Image processing failed: {str(e)}")
                mime_type = "image/png"
            
            print(f"[DEBUG] Starting video generation with prompt: '{prompt}'")
            operation = self.client.models.generate_videos(
                model="veo-3.1-fast-generate-preview",
                prompt=prompt or "animate this image naturally with subtle movement",
                image=types.Image(image_bytes=image_data, mime_type=mime_type),
                config=types.GenerateVideosConfig(
                    resolution="720p",
                    duration_seconds=8
                )
            )

            print(f"[DEBUG] Operation started: {operation.name}")
            
            max_wait_time = 300
            start_time = time.time()
            poll_count = 0
            
            while not operation.done:
                if time.time() - start_time > max_wait_time:
                    raise Exception(f"Video generation timed out after {max_wait_time} seconds")
                
                time.sleep(5)
                operation = self.client.operations.get(operation)
                poll_count += 1
                elapsed = int(time.time() - start_time)
                print(f"[DEBUG] Poll #{poll_count}, Elapsed: {elapsed}s, Done: {operation.done}")
            
            print(f"[DEBUG] Operation completed after {poll_count} polls")
            
            if not operation.result:
                raise Exception("Operation completed but no result was returned")
            
            if not operation.result.generated_videos:
                raise Exception("Operation completed but no videos were generated")
            
            uri = operation.result.generated_videos[0].video.uri
            print(f"[DEBUG] Video URI: {uri}")
            
            headers = {
                'x-goog-api-key': self.api_key
            }
            video_response = requests.get(uri, headers=headers, timeout=60)
            video_response.raise_for_status()
            video_data = video_response.content
            
            print(f"[DEBUG] Video downloaded: {len(video_data)} bytes")
            
            if len(video_data) < 1000:
                raise Exception(f"Video file is too small ({len(video_data)} bytes), likely corrupted")
            
            return video_data
            
        except Exception as e:
            error_msg = f"Video generation failed: {e}"
            print(f"[ERROR] {error_msg}")
            raise Exception(error_msg)