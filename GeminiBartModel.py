from ai_models import AIModel
from google import genai
from google.genai import types
from typing import List, Dict, Optional
import streamlit as st
import time
import requests
import os
from io import BytesIO
from PIL import Image as PILImage
import base64

try:
    from AnimateDiff import AnimateDiffGenerator, SimpleAnimateDiff
    ANIMATEDIFF_AVAILABLE = True
except ImportError as e:
    ANIMATEDIFF_AVAILABLE = False
    print(f"[WARNING] AnimateDiff not available: {e}")
    print("[WARNING] Video generation will not work without AnimateDiff or falling back to API")

class GeminiModel(AIModel):
    def __init__(self, api_key: str, bot_name: str = "Bartholemew"):
        self.client = genai.Client(api_key=api_key)
        self.api_key = api_key
        self.bot_name = bot_name
        self.animatediff = None
    
    def generate_response(self, messages: List[Dict], system_prompt: str, file_data: Optional[Dict] = None):
        formatted_history = []
        for m in messages[:-1]:
            gemini_role = "model" if m["role"] == "assistant" else "user"
            content = m["content"]
            clean_text = content if not str(content).startswith("IMAGE_DATA:") else "[Image]"
            formatted_history.append({"role": gemini_role, "parts": [{"text": clean_text}]})
        
        chat_session = self.client.chats.create(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=[types.Tool(google_search=types.GoogleSearch())]
            ),
            history=formatted_history
        )

        last_prompt = messages[-1]["content"]
        content_to_send = [last_prompt]
        
        if file_data:
            file_bytes = file_data.get('data') or file_data.get('bytes')
            file_mime = file_data.get('type') or file_data.get('mime')
            
            if file_bytes and file_mime:
                content_to_send.append(types.Part.from_bytes(
                    data=file_bytes,
                    mime_type=file_mime
                ))

        response = chat_session.send_message(content_to_send, stream=True)
        for chunk in response:
            yield chunk.text

    def generate_image(self, prompt: str) -> bytes:
        response = self.client.models.generate_content(
            model="imagen-3.0-generate-001",
            contents=f"Generate a high-quality image of: {prompt}"
        )
        return response.generated_images[0].image_bytes

    def generate_video(self, prompt: str, image_data: bytes = None) -> bytes:
        if not image_data:
            raise ValueError("Please upload an image first to animate it.")

        if not ANIMATEDIFF_AVAILABLE:
            raise Exception(
                "AnimateDiff is not available. This could be because:\n"
                "1. AnimateDiff.py file is missing\n"
                "2. Required dependencies are not installed (torch, diffusers)\n"
                "3. Running on Streamlit Cloud (doesn't support local GPU models)\n\n"
                "Solutions:\n"
                "- For local use: Install dependencies and ensure AnimateDiff.py is present\n"
                "- For Streamlit Cloud: Use a cloud API service like Replicate instead"
            )

        try:
            if self.animatediff is None:
                print("[GeminiModel] Initializing AnimateDiff...")
                use_simple = os.getenv("USE_SIMPLE_ANIMATEDIFF", "true").lower() == "true"
                
                if use_simple:
                    self.animatediff = SimpleAnimateDiff()
                    print("[GeminiModel] Using SimpleAnimateDiff (faster)")
                else:
                    self.animatediff = AnimateDiffGenerator()
                    print("[GeminiModel] Using AnimateDiffGenerator (better quality)")
            
            print(f"[GeminiModel] Starting local video generation")
            print(f"[GeminiModel] Prompt: '{prompt}'")
            
            num_frames = int(os.getenv("ANIMATEDIFF_FRAMES", "16"))
            fps = int(os.getenv("ANIMATEDIFF_FPS", "8"))
            
            video_data = self.animatediff.generate_from_image(
                image_data=image_data,
                prompt=prompt if prompt else "smooth subtle motion, high quality",
                num_frames=num_frames,
                fps=fps
            )
            
            print(f"[GeminiModel] Video generated: {len(video_data)} bytes")
            
            if len(video_data) < 1000:
                raise Exception(f"Video file is too small ({len(video_data)} bytes), likely corrupted")
            
            return video_data
                
        except Exception as e:
            error_msg = f"Video generation failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            raise Exception(error_msg)