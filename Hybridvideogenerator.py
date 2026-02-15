"""
Hybrid Video Generator - Works both locally and on cloud

Uses AnimateDiff when available (local with GPU), falls back to Replicate (cloud)
"""

import os
import streamlit as st
from typing import Optional
import base64
from io import BytesIO
from PIL import Image as PILImage

try:
    from AnimateDiff import AnimateDiffGenerator, SimpleAnimateDiff
    ANIMATEDIFF_AVAILABLE = True
except ImportError:
    ANIMATEDIFF_AVAILABLE = False
    print("AnimateDiff not available, will use cloud API")

try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False
    print("Replicate not available")

import requests


class HybridVideoGenerator:
    """
    Smart video generator that uses the best available method:
    1. Local AnimateDiff (if GPU available) - FREE & UNLIMITED
    2. Replicate API (if AnimateDiff unavailable) - PAID but works everywhere
    """
    
    def __init__(self):
        self.animatediff = None
        self.method = None
        self._detect_best_method()
    
    def _detect_best_method(self):
        """Automatically detect which method to use"""
        
        force_method = os.getenv("FORCE_VIDEO_METHOD", "").lower()
        if force_method == "local":
            self.method = "local"
            print("Forced to use LOCAL (AnimateDiff)")
            return
        elif force_method == "replicate":
            self.method = "replicate"
            print("Forced to use REPLICATE (cloud)")
            return
        
        if REPLICATE_AVAILABLE:
            self.method = "replicate"
            print("Using REPLICATE (fast, works everywhere)")
            print("To use local AnimateDiff, set FORCE_VIDEO_METHOD=local")
            return
        
        if ANIMATEDIFF_AVAILABLE:
            try:
                import torch
                if torch.cuda.is_available():
                    self.method = "local"
                    print("GPU detected! Using LOCAL AnimateDiff")
                    print("Note: First run will download ~5GB models (slow)")
                    return
                else:
                    print("WARNING: No GPU and no Replicate - AnimateDiff will be VERY slow")
                    self.method = "local"
                    return
            except:
                pass
        
        self.method = None
        print("ERROR: No video generation method available!")
    
    def generate_video(self, image_data: bytes, prompt: str = None) -> bytes:
        """Generate video using best available method"""
        
        if self.method == "local":
            return self._generate_local(image_data, prompt)
        elif self.method == "replicate":
            return self._generate_replicate(image_data, prompt)
        else:
            raise Exception(
                "No video generation method available!\n\n"
                "For LOCAL (free): Install torch, diffusers, and AnimateDiff.py\n"
                "For CLOUD: Install replicate and add REPLICATE_API_TOKEN to secrets"
            )
    
    def _generate_local(self, image_data: bytes, prompt: str) -> bytes:
        """Generate using local AnimateDiff"""
        try:
            if self.animatediff is None:
                use_simple = os.getenv("USE_SIMPLE_ANIMATEDIFF", "true").lower() == "true"
                if use_simple:
                    self.animatediff = SimpleAnimateDiff()
                    print("Loaded SimpleAnimateDiff")
                else:
                    self.animatediff = AnimateDiffGenerator()
                    print("Loaded AnimateDiffGenerator")
            
            num_frames = int(os.getenv("ANIMATEDIFF_FRAMES", "16"))
            fps = int(os.getenv("ANIMATEDIFF_FPS", "8"))
            
            return self.animatediff.generate_from_image(
                image_data=image_data,
                prompt=prompt or "smooth motion",
                num_frames=num_frames,
                fps=fps
            )
        except Exception as e:
            print(f"Local generation failed: {e}")
            if REPLICATE_AVAILABLE:
                print("Falling back to Replicate...")
                self.method = "replicate"
                return self._generate_replicate(image_data, prompt)
            raise
    
    def _generate_replicate(self, image_data: bytes, prompt: str) -> bytes:
        """Generate using Replicate API"""
        try:
            api_token = st.secrets.get("REPLICATE_API_TOKEN") or os.getenv("REPLICATE_API_TOKEN")
            if not api_token:
                raise ValueError("REPLICATE_API_TOKEN not found in secrets or environment")
            
            os.environ["REPLICATE_API_TOKEN"] = api_token
            
            img = PILImage.open(BytesIO(image_data))
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_uri = f"data:image/png;base64,{img_base64}"
            
            print("Calling Replicate API...")
            
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
            
            video_url = output if isinstance(output, str) else output[0] if isinstance(output, list) else str(output)
            video_response = requests.get(video_url, timeout=60)
            video_response.raise_for_status()
            
            return video_response.content
            
        except Exception as e:
            raise Exception(f"Replicate generation failed: {str(e)}")
    
    def get_method_info(self) -> dict:
        """Get information about current method"""
        return {
            "method": self.method,
            "animatediff_available": ANIMATEDIFF_AVAILABLE,
            "replicate_available": REPLICATE_AVAILABLE,
            "description": {
                "local": "LOCAL AnimateDiff (GPU) - Unlimited & Free",
                "replicate": "Replicate Cloud API - Paid (~$0.10/video)",
                None: "No method available - please configure"
            }.get(self.method)
        }