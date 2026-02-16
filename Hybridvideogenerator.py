"""
Simple Video Generator using Hugging Face Inference API
FREE - No authentication needed for public models
"""

import requests
import time
from io import BytesIO
from PIL import Image
import base64

class SimpleVideoGenerator:
    """
    FREE video generation using Hugging Face Inference API
    No API token required!
    """
    
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/ali-vilab/i2vgen-xl"
        print("[SimpleVideoGen] Using Hugging Face Inference API (FREE)")
    
    def generate_video(self, image_data: bytes, prompt: str = None) -> bytes:
        """
        Generate video from image using Hugging Face
        
        Args:
            image_data: Image as bytes
            prompt: Optional text prompt (not used by this model)
        
        Returns:
            Video as bytes
        """
        try:
            img = Image.open(BytesIO(image_data))
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            
            max_size = 512
            img.thumbnail((max_size, max_size), Image.LANCZOS)
            
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            
            print(f"[SimpleVideoGen] Generating video (this may take 30-60 seconds)...")
            
            headers = {"Content-Type": "application/json"}
            
            img_base64 = base64.b64encode(img_bytes).decode()
            
            payload = {
                "inputs": img_base64
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 503:
                print("[SimpleVideoGen] Model is loading, waiting 20 seconds...")
                time.sleep(20)
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=120
                )
            
            if response.status_code == 200:
                video_data = response.content
                print(f"[SimpleVideoGen] Video generated: {len(video_data)} bytes")
                return video_data
            else:
                raise Exception(f"API returned status {response.status_code}: {response.text}")
                
        except Exception as e:
            raise Exception(f"Video generation failed: {str(e)}")


class FallbackVideoGenerator:
    """
    Fallback: Create a simple animated GIF from the image
    Always works, no API needed
    """
    
    def generate_video(self, image_data: bytes, prompt: str = None) -> bytes:
        """
        Create a simple zoom/pan animation as fallback
        Returns MP4 video
        """
        try:
            import imageio
            import numpy as np
            from PIL import Image, ImageEnhance
            import tempfile
            import os
            
            print("[FallbackVideoGen] Creating simple animation...")
            
            img = Image.open(BytesIO(image_data))
            img = img.convert('RGB')
            
            img.thumbnail((512, 512), Image.LANCZOS)
            
            frames = []
            num_frames = 16
            
            for i in range(num_frames):
                zoom = 1.0 + (i / num_frames) * 0.1
                
                w, h = img.size
                new_w, new_h = int(w * zoom), int(h * zoom)
                zoomed = img.resize((new_w, new_h), Image.LANCZOS)
                
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
            
            print(f"[FallbackVideoGen] Simple animation created: {len(video_bytes)} bytes")
            
            return video_bytes
            
        except Exception as e:
            raise Exception(f"Fallback generation failed: {str(e)}")


class UniversalVideoGenerator:
    """
    Try multiple methods until one works
    """
    
    def __init__(self):
        self.methods = [
            ("HuggingFace", SimpleVideoGenerator()),
            ("Fallback", FallbackVideoGenerator())
        ]
        print("[UniversalVideoGen] Initialized with multiple fallback methods")
    
    def generate_video(self, image_data: bytes, prompt: str = None) -> bytes:
        """Try each method until one works"""
        
        last_error = None
        
        for method_name, generator in self.methods:
            try:
                print(f"[UniversalVideoGen] Trying {method_name}...")
                return generator.generate_video(image_data, prompt)
            except Exception as e:
                print(f"[UniversalVideoGen] {method_name} failed: {str(e)}")
                last_error = e
                continue
        
        raise Exception(f"All video generation methods failed. Last error: {last_error}")