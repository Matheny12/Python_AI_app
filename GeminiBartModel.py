from ai_models import AIModel
from google import genai
from google.genai import types
from typing import List, Dict, Optional
import streamlit as st

class GeminiModel(AIModel):
    def __init__(self, api_key: str, bot_name: str = "Bartholemew"):
        self.client = genai.Client(api_key=api_key)
        self.bot_name = bot_name
    
    def generate_response(self, messages: List[Dict], system_prompt: str, file_data: Optional[Dict] = None):
        formatted_history = []
        for m in messages[:-1]:
            gemini_role = "model" if m["role"] == "assistant" else "user"
            clean_text = m["content"] if not str(m["content"]).startswith("IMAGE_DATA:") else "[Image]"
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
            content_to_send.append(
                types.Part.from_bytes(
                    data=file_data["bytes"],
                    mime_type=file_data["mime"]
                )
            )
            
        response_stream = chat_session.send_message_stream(content_to_send)
        
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    def generate_image(self, prompt: str) -> bytes:
        safe_prompt = self._refine_prompt(prompt)
        
        model_options = ['imagen-4.0-generate-001']
        last_error = ""
        
        for model_id in model_options:
            try:
                response = self.client.models.generate_images(
                    model=model_id,
                    prompt=safe_prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio="1:1",
                        person_generation="ALLOW_ADULT",
                        safety_filter_level="BLOCK_LOW_AND_ABOVE"
                    )
                )
                
                if response and hasattr(response, 'generated_images') and response.generated_images:
                    return response.generated_images[0].image.image_bytes
                else:
                    last_error = "Safety filters blocked the generation or no image was returned"
            except Exception as e:
                last_error = str(e)
                continue
        
        raise Exception(f"Image generation failed: {last_error}")
    
    def _refine_prompt(self, prompt: str) -> str:
        try:
            refine_chat = self.client.chats.create(model="gemini-2.5-flash-lite")
            refine_res = refine_chat.send_message(
                "You are an artist's prompt engineer. Create a highly detailed, "
                "cinematic physical description of the following subject. "
                "IMPORTANT: Remove all names of real people, politicians, or celebrities. "
                "Describe their facial features, hair, clothing, and the lighting style "
                f"generically so an artist can paint it without knowing who it is: '{prompt}'"
            )
            if refine_res.text:
                return refine_res.text
        except Exception:
            pass
        
        return prompt
    
    def generate_video(self, prompt: str, image_data: bytes = None) -> bytes:
        import time
        import io
        import tempfile
        import os
        import requests
        from PIL import Image as PILImage
        
        if not image_data:
            raise NotImplementedError(
                "Text-to-video is not supported. Please upload an image first, then use /video to animate it."
            )
        
        try:
            image = PILImage.open(io.BytesIO(image_data))
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                image.save(tmp_file.name, format='PNG')
                tmp_path = tmp_file.name
            
            try:
                uploaded_image = self.client.files.upload(file=tmp_path)
                
                operation = self.client.models.generate_videos(
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
                    operation = self.client.operations.get(operation)
                
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