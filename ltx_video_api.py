"""
LTX-2 Video Generator with Audio Support
Supports both Replicate and direct LTX-2 API

VERSION: 2025-02-16-v2 (FIXED - Forces Direct LTX API)
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
    REPLICATE_AVAILABLE = False
except ImportError:
    REPLICATE_AVAILABLE = False


class LTX2VideoGenerator:
    """
    LTX-2 Video Generation with Audio
    - Official API: https://api.ltx.video (requires API key from console.ltx.video)
    - Replicate: Fallback option using replicate.com
    - Up to 60 seconds of video
    - Synchronized audio generation
    - 4K at 50fps support
    """
    
    def __init__(self, force_method=None):
        """
        Initialize LTX Video Generator
        
        Args:
            force_method: Optional - "direct" or "replicate" to force a specific method
        """
        self.method = self._detect_method(force_method)
    
    def _detect_method(self, force_method=None):
        """Detect which API to use"""
        
        print("=" * 60)
        print("[LTX DEBUG] Method detection starting...")
        print(f"[LTX DEBUG] force_method parameter: {force_method}")
        print(f"[LTX DEBUG] REPLICATE_AVAILABLE: {REPLICATE_AVAILABLE}")
        
        if force_method:
            print(f"[LTX] Forced method: {force_method}")
            return force_method
        
        force = os.getenv("LTX_METHOD", "").lower()
        if force == "direct":
            print("[LTX] Using direct method from LTX_METHOD env var")
            return "direct"
        elif force == "replicate":
            print("[LTX] Using replicate method from LTX_METHOD env var")
            return "replicate"
        
        if STREAMLIT_AVAILABLE:
            try:
                force_secret = st.secrets.get("LTX_METHOD", "").lower()
                if force_secret == "direct":
                    print("[LTX] Using direct method from Streamlit secrets")
                    return "direct"
                elif force_secret == "replicate":
                    print("[LTX] Using replicate method from Streamlit secrets")
                    return "replicate"
            except:
                pass
        
        ltx_key = None
        if STREAMLIT_AVAILABLE:
            try:
                ltx_key = st.secrets.get("LTX_API_KEY")
            except:
                pass
        if not ltx_key:
            ltx_key = os.getenv("LTX_API_KEY")
        
        if ltx_key and ltx_key.startswith("ltxv_"):
            print("[LTX] Found LTX_API_KEY, using direct API")
            return "direct"
        
        if REPLICATE_AVAILABLE:
            print("[LTX] Using Replicate as fallback")
            return "replicate"
        
        print("[LTX] Defaulting to direct method")
        print("=" * 60)
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
        """Generate using official LTX Video API at api.ltx.video"""
        
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
                    "Get it from: https://console.ltx.video/api-keys\n"
                    "Add to secrets: LTX_API_KEY = 'ltxv_your_key'"
                )
            
            print("[LTX Video API] Processing image...")
            
            img = PILImage.open(BytesIO(image_data))
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            
            max_dimension = 512
            if max(img.size) > max_dimension:
                ratio = max_dimension / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, PILImage.Resampling.LANCZOS)
                print(f"[LTX Video API] Resized image to {new_size}")
            
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=70, optimize=True)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_uri = f"data:image/jpeg;base64,{img_base64}"
            
            payload_size_mb = len(img_base64) / (1024 * 1024)
            print(f"[LTX Video API] Payload size: {payload_size_mb:.2f} MB")
            
            api_url = "https://api.ltx.video/v1/image-to-video"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "image_uri": image_uri,
                "prompt": prompt or "smooth natural motion with synchronized audio",
                "model": "ltx-2-pro",
                "duration": 8,
                "resolution": "1920x1080"
            }
            
            print(f"[LTX Video API] Generating video with prompt: '{prompt}'")
            print("[LTX Video API] This may take 1-2 minutes...")
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            
            video_data = response.content
            
            print(f"[LTX Video API] Success! Generated {len(video_data)} bytes")
            
            return video_data
            
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_json = e.response.json()
                error_detail = f": {error_json}"
            except:
                error_detail = f" (Status {e.response.status_code})"
            raise Exception(f"LTX Video API request failed{error_detail}")
        except Exception as e:
            raise Exception(f"LTX Video API failed: {str(e)}")
    
    def get_info(self):
        """Get info about current configuration"""
        return {
            "method": self.method,
            "description": {
                "replicate": "LTX-Video via Replicate (~5sec videos)",
                "direct": "Official LTX Video API (8-60sec videos with audio, 4K support)"
            }.get(self.method),
            "api_url": {
                "replicate": "replicate.com/lightricks/ltx-video",
                "direct": "https://api.ltx.video/v1/image-to-video"
            }.get(self.method)
        }