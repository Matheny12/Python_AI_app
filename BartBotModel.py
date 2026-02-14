from ai_models import AIModel
from typing import List, Dict, Optional, Generator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from diffusers import StableDiffusionPipeline
from PIL import Image
import io
import os
import streamlit as st
from threading import Thread

class BartBotModel(AIModel):
    @st.cache_resource
    def _get_model_and_tokenizer(_self, model_path, hf_token):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=hf_token,
            trust_remote_code=True
        )
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token,
            trust_remote_code=True
        )
        return tokenizer, model

    def __init__(self, model_path: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        hf_token = os.getenv("HF_TOKEN")
        self.tokenizer, self.model = self._get_model_and_tokenizer(model_path, hf_token)
        self.image_model = None
        self.vision_model = None

    def generate_response(self, messages: List[Dict], system_prompt: str, file_data: Optional[Dict] = None) -> Generator:
        prompt = self._format_messages(messages, system_prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

    def _format_messages(self, messages: List[Dict], system_prompt: str) -> str:
        prompt_parts = []
        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}\n")
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Bartholemew"
            content = msg["content"]
            if isinstance(content, str) and not content.startswith("IMAGE_DATA:"):
                prompt_parts.append(f"{role}: {content}\n")
        prompt_parts.append("Bartholemew:")
        return "\n".join(prompt_parts)
    
    def generate_image(self, prompt: str) -> bytes:
        if self.image_model is None:
            hf_token = os.getenv("HF_TOKEN")
            self.image_model = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16,
                token=hf_token
            ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        image = self.image_model(
            prompt,
            num_inference_steps=25,
            guidance_scale=7.5
        ).images[0]

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()