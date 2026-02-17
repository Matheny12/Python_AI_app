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
from ltx_video_api import LTX2VideoGenerator

class BartBotModel(AIModel):
    @st.cache_resource
    def _get_llm(_self):
        model_name = "Llama-3.2-1B-Instruct-Q4_0.gguf"
        return GPT4All(model_name=model_name, allow_download=True)
    
    def __init__(self):
        self.llm = self._get_llm()
        try:
            self.api_key = st.secrets["GEMINI_KEY"]
        except (KeyError, AttributeError):
            self.api_key = os.getenv("GEMINI_KEY")
        
        from google import genai
        self.client = genai.Client(api_key=self.api_key)
        print("[BartBotModel] Initializing video generator with FORCED direct method...")
        self.video_generator = LTX2VideoGenerator(force_method="direct")
        print(f"[BartBotModel] Video generator method is: {self.video_generator.method}")

    def generate_response(self, messages: List[Dict], system_prompt: str, file_data: Optional[Dict] = None) -> Generator:
        from google.genai import types

        if file_data:
            print("[BartBotModel] File detected, using Gemini for vision")
            file_bytes = file_data.get('data') or file_data.get('bytes')
            file_mime = file_data.get('type') or file_data.get('mime')
            
            if file_bytes and file_mime:
                try:
                    response = self.client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=[
                            messages[-1]["content"],
                            types.Part.from_bytes(data=file_bytes, mime_type=file_mime)
                        ]
                    )
                    yield response.text
                    return
                except Exception as e:
                    yield f"Error analyzing image: {str(e)}"
                    return

        try:
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

            response = chat_session.send_message(messages[-1]["content"])

            if hasattr(response, 'text'):
                yield response.text
            elif isinstance(response, tuple):
                for item in response:
                    if hasattr(item, 'text'):
                        yield item.text
            else:
                yield str(response)

        except Exception as e:
            print(f"[BartBotModel] Gemini failed, falling back to local LLM: {e}")
            with self.llm.chat_session(system_prompt):
                user_input = messages[-1]["content"]
                response_generator = self.llm.generate(user_input, max_tokens=1024, streaming=True)
                for token in response_generator:
                    yield token

    def generate_image(self, prompt: str) -> bytes:
        from huggingface_hub import InferenceClient
        from google.genai import types

        try:
            print(f"[BartBotModel] Searching web to enhance image prompt: '{prompt}'")
            search_response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"""Search the web and find detailed visual descriptions for: {prompt}
                
Then write an enhanced, highly detailed image generation prompt based on what you find.
Focus on: colors, textures, lighting, style, composition, specific visual details.
Return ONLY the enhanced prompt, nothing else.""",
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
            )
            enhanced_prompt = search_response.text.strip()
            print(f"[BartBotModel] Enhanced prompt: '{enhanced_prompt}'")
        except Exception as e:
            print(f"[BartBotModel] Web search failed, using original prompt: {e}")
            enhanced_prompt = prompt

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            raise Exception("HF_TOKEN missing for image generation.")
        
        client = InferenceClient(api_key=hf_token)
        image = client.text_to_image(enhanced_prompt, model="black-forest-labs/FLUX.1-schnell")
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()
    
    def generate_video(self, prompt: str, image_data: bytes = None) -> bytes:
        if not image_data:
            raise ValueError("Please upload an image first.")

        try:
            print(f"[BartBotModel] Starting video generation")
            print(f"[BartBotModel] Method: {self.video_generator.method}")
            print(f"[BartBotModel] Prompt: '{prompt}'")
            
            video_data = self.video_generator.generate_video(image_data, prompt)
            
            print(f"[BartBotModel] Video generated: {len(video_data)} bytes")
            
            if len(video_data) < 1000:
                raise Exception(f"Video file is too small ({len(video_data)} bytes)")
            
            return video_data
            
        except Exception as e:
            error_msg = f"Video generation failed: {e}"
            print(f"[ERROR] {error_msg}")
            raise Exception(error_msg)