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
from ltx_video_api import LTX2VideoGenerator

class GeminiModel(AIModel):
    def __init__(self, api_key: str, bot_name: str = "Bartholemew"):
        self.client = genai.Client(api_key=api_key)
        self.api_key = api_key
        self.bot_name = bot_name
        print("[GeminiBartModel] Initializing video generator with FORCED direct method...")
        self.video_generator = LTX2VideoGenerator(force_method="direct")
        print(f"[GeminiBartModel] Video generator method is: {self.video_generator.method}")
    
    def generate_response(self, messages: List[Dict], system_prompt: str, file_data: Optional[Dict] = None):
        formatted_history = []
        for m in messages[:-1]:
            gemini_role = "model" if m["role"] == "assistant" else "user"
            content = m["content"]
            clean_text = content if not str(content).startswith("IMAGE_DATA:") else "[Image]"
            formatted_history.append({"role": gemini_role, "parts": [{"text": clean_text}]})
        
        chat_session = self.client.chats.create(
            model="gemini-2.5-flash-lite",
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
        response = self.client.models.generate_images(
            model="imagen-3.0-generate-001",
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="1:1",
            safety_filter_level="block_only_high",
            person_generation="allow_adult"
        )
        return response.generated_images[0].image.data

    def generate_video(self, prompt: str, image_data: bytes = None) -> bytes:
        if not image_data:
            raise ValueError("Please upload an image first to animate it.")

        try:
            print(f"[GeminiModel] Starting video generation")
            print(f"[GeminiModel] Method: {self.video_generator.method}")
            print(f"[GeminiModel] Prompt: '{prompt}'")
            
            video_data = self.video_generator.generate_video(image_data, prompt)
            
            print(f"[GeminiModel] Video generated: {len(video_data)} bytes")
            
            if len(video_data) < 1000:
                raise Exception(f"Video file is too small ({len(video_data)} bytes)")
            
            return video_data
                
        except Exception as e:
            error_msg = f"Video generation failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            raise Exception(error_msg)