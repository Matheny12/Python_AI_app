"""
LTX-2 Video Generator with Audio Support
Supports both Replicate and direct LTX-2 API
"""

import os
import base64
import time
import requests
from io import BytesIO
from PIL import Image as PILImage

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False


class LTX2VideoGenerator:
    """
    LTX-2 Video Generation with Audio
    - Up to 60 seconds of video
    - Synchronized audio generation
    - 4K at 50fps support
    """
    
    def __init__(self):
        self.method = self._detect_method()
    
    def _detect_method(self):
        """Detect which API to use"""
        
        force = os.getenv("LTX_METHOD", "").lower()
        if force == "direct":
            return "direct"
        elif force == "replicate":
            return "replicate"
        
        if REPLICATE_AVAILABLE:
            return "replicate"
        
        return "direct"
    
    def generate_video(self, image_data: bytes, prompt: str = None) -> bytes:
        """
        Generate video from image using LTX-2
        
        Args:
            image_data: Image as bytes
            prompt: Text description for animation
        
        Returns:
            Video bytes (MP4 with audio)
        """
        
        if self.method == "replicate":
            return self._generate_replicate(image_data, prompt)
        else:
            return self._generate_direct(image_data, prompt)
    
    def _generate_replicate(self, image_data: bytes, prompt: str) -> bytes:
        """Generate using Replicate API (EASIEST)"""
        
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
                raise ValueError(
                    "REPLICATE_API_TOKEN required!\n"
                    "Add to secrets: REPLICATE_API_TOKEN = 'r8_your_token'"
                )
            
            os.environ["REPLICATE_API_TOKEN"] = api_token
            
            print("[LTX-2] Processing image...")
            
            img = PILImage.open(BytesIO(image_data))
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_uri = f"data:image/png;base64,{img_base64}"
            
            print(f"[LTX-2] Generating video with prompt: '{prompt}'")
            print("[LTX-2] This may take 1-3 minutes...")
            
            output = replicate.run(
                "lightricks/ltx-video:cae196b9527e2aecd9d86fa09c120d63a0f4ca11bb172e6e838f3bcaa83c2b05",
                input={
                    "prompt": prompt or "smooth natural motion, high quality",
                    "image": image_uri,
                    "num_frames": 121,
                    "num_inference_steps": 30,
                }
            )
            
            video_url = output if isinstance(output, str) else output[0] if isinstance(output, list) else str(output)
            
            print(f"[LTX-2] Downloading video from: {video_url}")
            
            video_response = requests.get(video_url, timeout=120)
            video_response.raise_for_status()
            video_data = video_response.content
            
            print(f"[LTX-2] Success! Generated {len(video_data)} bytes")
            
            return video_data
            
        except Exception as e:
            raise Exception(f"LTX-2 generation failed: {str(e)}")
    
    def _generate_direct(self, image_data: bytes, prompt: str) -> bytes:
        """Generate using direct LTX-2 API"""
        
        try:
            api_key = None
            if STREAMLIT_AVAILABLE:
                try:
                    api_key = st.secrets["LTX_API_KEY"]
                except (KeyError, AttributeError):
                    pass
            
            if not api_key:
                api_key = os.getenv("LTX_API_KEY")
            
            if not api_key:
                raise ValueError(
                    "LTX_API_KEY required!\n"
                    "Get it from: https://ltx-2api.com/\n"
                    "Add to secrets: LTX_API_KEY = 'your_key'"
                )
            
            print("[LTX-2 Direct] Processing image...")
            
            img = PILImage.open(BytesIO(image_data))
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            api_url = "https://ltx-2api.com/api/generate"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "prompt": prompt or "smooth natural motion with audio",
                "image": img_base64,
                "duration": 8,
                "resolution": "1080p",
                "mode": "fast"
            }
            
            print(f"[LTX-2 Direct] Submitting generation request...")
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            task_data = response.json()["data"]
            task_id = task_data["task_id"]
            
            print(f"[LTX-2 Direct] Task ID: {task_id}")
            print("[LTX-2 Direct] Polling for completion (1-3 minutes)...")
            
            status_url = f"https://ltx-2api.com/api/status?task_id={task_id}"
            
            for i in range(60):
                time.sleep(5)
                
                status_resp = requests.get(status_url, timeout=30)
                status_resp.raise_for_status()
                status_data = status_resp.json()["data"]
                
                status = status_data["status"]
                
                if status == "SUCCESS":
                    video_url = status_data["response"][0]
                    print(f"[LTX-2 Direct] Success! Downloading from {video_url}")
                    
                    video_resp = requests.get(video_url, timeout=120)
                    video_resp.raise_for_status()
                    
                    print(f"[LTX-2 Direct] Generated {len(video_resp.content)} bytes")
                    return video_resp.content
                    
                elif status == "FAILED":
                    error_msg = status_data.get("error_message", "Unknown error")
                    raise Exception(f"Generation failed: {error_msg}")
                
                print(f"[LTX-2 Direct] Status: {status} (attempt {i+1}/60)")
            
            raise Exception("Timeout waiting for video generation")
            
        except Exception as e:
            raise Exception(f"LTX-2 Direct API failed: {str(e)}")
    
    def get_info(self):
        """Get info about current configuration"""
        return {
            "method": self.method,
            "description": {
                "replicate": "LTX-Video via Replicate (~5sec videos, uses existing token)",
                "direct": "LTX-2 Direct API (10-60sec videos with audio, requires LTX_API_KEY)"
            }.get(self.method)
        }
