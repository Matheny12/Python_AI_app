from ai_models import AIModel
from typing import List, Dict, Optional, Generator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from diffusers import StableDiffusionPipeline
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
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
            token=hf_token,
            trust_remote_code=True,
            attn_implementation="sdpa" 
        )
        return tokenizer, model

    def __init__(self, model_path: str = "meta-llama/Llama-3.2-3B-Instruct"):
        hf_token = os.getenv("HF_TOKEN")
        self.tokenizer, self.model = self._get_model_and_tokenizer(model_path, hf_token)
        self.image_model = None
        self.vision_model = None

    def generate_response(self, messages: List[Dict], system_prompt: str, file_data: Optional[Dict] = None) -> Generator:
        formatted_messages = self._format_for_llama(messages, system_prompt)
        
        inputs = self.tokenizer.apply_chat_template(
            formatted_messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.model.device)
        
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            input_ids=inputs,
            streamer=streamer,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

    def _format_for_llama(self, messages: List[Dict], system_prompt: str) -> List[Dict]:
        llama_msgs = []
        if system_prompt:
            llama_msgs.append({"role": "system", "content": system_prompt})
        
        for msg in messages:
            role = "user" if msg["role"] == "user" else "assistant"
            content = msg["content"]
            
            if isinstance(content, str) and not content.startswith("IMAGE_DATA:"):
                llama_msgs.append({"role": role, "content": content})
        
        return llama_msgs

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