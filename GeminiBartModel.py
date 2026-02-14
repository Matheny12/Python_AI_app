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

class GeminiModel(AIModel):
    def __init__(self, api_key: str, bot_name: str = "Bartholemew"):
        self.client = genai.Client(api_key=api_key)
        self.api_key = api_key
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
            try:
                img = PILImage.open(BytesIO(image_data))
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
                    buffered = BytesIO()
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
                prompt=prompt if prompt else "animate this image naturally with subtle movement",
                image=types.Image(image_bytes=image_data, mime_type=mime_type),
                config=types.GenerateVideosConfig(
                    aspect_ratio="16:9",
                    duration_seconds=8,
                    resolution="720p"
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
            
            video_uri = operation.result.generated_videos[0].video.uri
            print(f"[DEBUG] Video URI: {video_uri}")
            
            headers = {
                'x-goog-api-key': self.api_key
            }
            video_response = requests.get(video_uri, headers=headers, timeout=60)
            video_response.raise_for_status()
            video_data = video_response.content
            
            print(f"[DEBUG] Video downloaded: {len(video_data)} bytes")
            
            if len(video_data) < 1000:
                raise Exception(f"Video file is too small ({len(video_data)} bytes), likely corrupted")
            
            return video_data
                
        except Exception as e:
            error_msg = f"Video generation failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            raise Exception(error_msg)