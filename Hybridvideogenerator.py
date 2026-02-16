"""
Hybrid Video Generator with built-in FREE methods
No separate files needed!
"""

import os
from typing import Optional
import base64
from io import BytesIO
import time
import requests
import tempfile

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from PIL import Image as PILImage, ImageEnhance

try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False

try:
    import imageio
    import numpy as np
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("[HybridVideoGen] imageio not available - simple animations disabled")


class HybridVideoGenerator:
    """
    Smart video generator with FREE built-in methods
    Priority: FREE > Replicate > Local
    """
    
    def __init__(self):
        self.method = None
        self._detect_best_method()
    
    def _detect_best_method(self):
        """Automatically detect which method to use - PREFER FREE!"""
        
        force_method = os.getenv("FORCE_VIDEO_METHOD", "").lower()
        if force_method == "replicate":
            self.method = "replicate"
            print("[HybridVideoGen] Forced to REPLICATE")
            return
        
        self.method = "free"
        print("[HybridVideoGen] Using FREE video generation (built-in)")
    
    def generate_video(self, image_data: bytes, prompt: str = None) -> bytes:
        """Generate video using best available method"""
        
        if self.method == "free":
            return self._generate_free(image_data, prompt)
        elif self.method == "replicate":
            return self._generate_replicate(image_data, prompt)
        else:
            return self._generate_free(image_data, prompt)
    
    def _generate_free(self, image_data: bytes, prompt: str) -> bytes:
        """
        FREE video generation with multiple fallbacks:
        1. Try HuggingFace Inference API (free, public)
        2. Fall back to simple animation (always works)
        """
        
        try:
            print("[FreeVideoGen] Trying HuggingFace Inference API...")
            return self._generate_huggingface(image_data)
        except Exception as e:
            print(f"[FreeVideoGen] HuggingFace failed: {e}")
        
        print("[FreeVideoGen] Using fallback animation...")
        return self._generate_simple_animation(image_data)
    
    def _generate_huggingface(self, image_data: bytes) -> bytes:
        """Try HuggingFace Inference API (FREE, no token)"""
        
        img = PILImage.open(BytesIO(image_data))
        if img.mode not in ('RGB', 'RGBA'):
            img = img.convert('RGB')
        
        img.thumbnail((512, 512), PILImage.LANCZOS)
        
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        
        img_base64 = base64.b64encode(img_bytes).decode()
        
        api_url = "https://api-inference.huggingface.co/models/ali-vilab/i2vgen-xl"
        headers = {"Content-Type": "application/json"}
        payload = {"inputs": img_base64}
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 503:
            print("[HuggingFace] Model loading, waiting 20s...")
            time.sleep(20)
            response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            print(f"[HuggingFace] Success! {len(response.content)} bytes")
            return response.content
        else:
            raise Exception(f"API error {response.status_code}")
    
    def _generate_simple_animation(self, image_data: bytes) -> bytes:
        """
        Simple animation fallback - ALWAYS WORKS
        Creates a zoom effect
        """
        
        if not IMAGEIO_AVAILABLE:
            raise Exception("imageio not available for fallback animation")
        
        print("[SimpleAnim] Creating zoom animation...")
        
        img = PILImage.open(BytesIO(image_data))
        img = img.convert('RGB')
        img.thumbnail((512, 512), PILImage.LANCZOS)
        
        frames = []
        num_frames = 16
        
        for i in range(num_frames):
            zoom = 1.0 + (i / num_frames) * 0.15
            
            w, h = img.size
            new_w, new_h = int(w * zoom), int(h * zoom)
            zoomed = img.resize((new_w, new_h), PILImage.LANCZOS)
            
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            cropped = zoomed.crop((left, top, left + w, top + h))
            
            enhancer = ImageEnhance.Brightness(cropped)
            adjusted = enhancer.enhance(0.95 + (i / num_frames) * 0.1)
            
            frames.append(np.array(adjusted))
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name
        
        imageio.mimsave(tmp_path, frames, fps=8, codec='libx264')
        
        with open(tmp_path, 'rb') as f:
            video_bytes = f.read()
        
        os.unlink(tmp_path)
        
        print(f"[SimpleAnim] Created! {len(video_bytes)} bytes")
        return video_bytes
    
    def _generate_replicate(self, image_data: bytes, prompt: str) -> bytes:
        """Generate using Replicate API (requires token)"""
        try:
            api_token = None
            if STREAMLIT_AVAILABLE:
                try:
                    api_token = st.secrets["REPLICATE_API_TOKEN"]
                except (KeyError, AttributeError):
                    pass
            
            if not api_token:
                api_token = os.getenv("REPLICATE_API_TOKEN")
            
            if not api_token:
                raise ValueError("REPLICATE_API_TOKEN not found - using FREE method instead")
            
            os.environ["REPLICATE_API_TOKEN"] = api_token
            
            img = PILImage.open(BytesIO(image_data))
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_uri = f"data:image/png;base64,{img_base64}"
            
            print("[Replicate] Calling API...")
            
            output = replicate.run(
                "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438",
                input={
                    "input_image": image_uri,
                    "cond_aug": 0.02,
                    "decoding_t": 7,
                    "video_length": "14_frames_with_svd",
                    "sizing_strategy": "maintain_aspect_ratio",
                    "motion_bucket_id": 127,
                    "frames_per_second": 7
                }
            )
            
            video_url = output if isinstance(output, str) else output[0]
            video_response = requests.get(video_url, timeout=60)
            video_response.raise_for_status()
            
            return video_response.content
            
        except Exception as e:
            print(f"[Replicate] Failed: {e}, falling back to FREE...")
            return self._generate_free(image_data, prompt)
    
    def get_method_info(self) -> dict:
        """Get information about current method"""
        return {
            "method": self.method,
            "description": {
                "free": "FREE Built-in (HuggingFace + Animation)",
                "replicate": "Replicate API (~$0.10/video)",
            }.get(self.method)
        }