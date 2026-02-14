from ai_models import AIModel
from google import genai
from google.genai import types
from typing import List, Dict, Optional
import streamlit as st
import time
import requests
import os

class GeminiModel(AIModel):
    def __init__(self, api_key: str, bot_name: str = "Bartholemew"):
        self.client = genai.Client(api_key=api_key)
        self.bot_name = bot_name
    
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
            content_to_send.append(types.Part.from_bytes(
                data=file_data['data'],
                mime_type=file_data['type']
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

        try:
            operation = self.client.models.generate_videos(
                model="veo-3.1-fast-generate-preview",
                prompt=prompt if prompt else "animate this image naturally",
                image=types.Image(image_bytes=image_data, mime_type="image/png"),
                config=types.GenerateVideosConfig(
                    aspect_ratio="16:9",
                    duration_seconds=8,
                    resolution="720p"
                )
            )

            while not operation.done:
                time.sleep(5)
                operation = self.client.operations.get(operation)
            
            if operation.result and operation.result.generated_videos:
                video_uri = operation.result.generated_videos[0].video.uri
                return requests.get(video_uri).content

            if operation.result and operation.result.generated_videos:
            video_uri = operation.result.generated_videos[0].video.uri
            
            if video_uri.startswith("gs://"):
                raise Exception("Video is in private cloud storage. Switch to 'fast-generate' for direct links.")
            
            video_response = requests.get(
                video_uri, 
                headers={"x-goog-api-key": self.client.api_key} 
            )
            
            if video_response.status_code == 200:
                return video_response.content
            else:
                raise Exception(f"Failed to download video: {video_response.status_code}")    
            raise Exception("No video was generated.")
        except Exception as e:
            raise Exception(f"Failed to generate video: {str(e)}")
            