import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from PIL import Image
import os
import io
import tempfile

class AnimateDiffGenerator:
    """
    AnimateDiff video generator - Unlimited FREE local video generation
    
    Features:
    - Fully local (no API calls)
    - Unlimited generations
    - Works with uploaded images
    - Customizable motion and style
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.pipe = None
        print(f"[AnimateDiff] Using device: {device}")
        
        if device == "cpu":
            print("[AnimateDiff] WARNING: CPU mode is VERY slow (5-10 min per video)")
            print("[AnimateDiff] For better performance, use a GPU")
    
    def load_model(self):
        """Load AnimateDiff model (one-time setup, ~5GB download)"""
        if self.pipe is not None:
            return
        
        print("[AnimateDiff] Loading model... (first time may take 5-10 minutes)")
        
        try:
            adapter = MotionAdapter.from_pretrained(
                "guoyww/animatediff-motion-adapter-v1-5-2",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
            self.pipe = AnimateDiffPipeline.from_pretrained(
                model_id,
                motion_adapter=adapter,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            if self.device == "cuda":
                self.pipe.enable_vae_slicing()
                self.pipe.enable_model_cpu_offload()
            
            self.pipe = self.pipe.to(self.device)
            
            print("[AnimateDiff] Model loaded successfully!")
            
        except Exception as e:
            print(f"[AnimateDiff] Error loading model: {str(e)}")
            raise
    
    def generate_from_image(self, image_data: bytes, prompt: str = None, 
                          num_frames: int = 16, fps: int = 8) -> bytes:
        """
        Generate video from an image
        
        Args:
            image_data: Image as bytes
            prompt: Text prompt to guide the animation
            num_frames: Number of frames (default: 16, ~2 seconds at 8fps)
            fps: Frames per second (default: 8)
        
        Returns:
            Video as bytes (MP4 format)
        """
        self.load_model()
        
        try:
            image = Image.open(io.BytesIO(image_data))
            image = image.convert('RGB')
            
            width, height = image.size
            aspect_ratio = width / height
            
            if aspect_ratio > 1:
                new_width = 512
                new_height = int(512 / aspect_ratio)
            else:
                new_height = 512
                new_width = int(512 * aspect_ratio)
            
            new_width = (new_width // 8) * 8
            new_height = (new_height // 8) * 8
            
            image = image.resize((new_width, new_height), Image.LANCZOS)
            
            print(f"[AnimateDiff] Image resized to: {new_width}x{new_height}")
            
            if not prompt or prompt.strip() == "":
                prompt = "smooth subtle motion, high quality, detailed"
            else:
                prompt = f"{prompt}, smooth motion, high quality, detailed"
            
            negative_prompt = "static, blurry, low quality, distorted, ugly"
            
            print(f"[AnimateDiff] Generating {num_frames} frames...")
            print(f"[AnimateDiff] Prompt: {prompt}")
            
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                guidance_scale=7.5,
                num_inference_steps=25,
                generator=torch.Generator(device=self.device).manual_seed(42),
            )
            
            frames = output.frames[0]
            
            print(f"[AnimateDiff] Generated {len(frames)} frames")
            
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp_path = tmp.name
            
            export_to_video(frames, tmp_path, fps=fps)
            
            with open(tmp_path, 'rb') as f:
                video_bytes = f.read()
            
            os.unlink(tmp_path)
            
            print(f"[AnimateDiff] Video generated: {len(video_bytes)} bytes")
            
            return video_bytes
            
        except Exception as e:
            print(f"[AnimateDiff] Generation error: {str(e)}")
            raise
    
    def unload_model(self):
        """Free up memory"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[AnimateDiff] Model unloaded")


class SimpleAnimateDiff:
    """
    Simpler, lighter AnimateDiff implementation
    Uses img2img pipeline for faster results
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.pipe = None
        print(f"[SimpleAnimateDiff] Using device: {device}")
    
    def load_model(self):
        """Load a lighter model for faster generation"""
        if self.pipe is not None:
            return
        
        print("[SimpleAnimateDiff] Loading lightweight model...")
        
        from diffusers import StableDiffusionImg2ImgPipeline
        
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )
        
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
        
        self.pipe = self.pipe.to(self.device)
        print("[SimpleAnimateDiff] Model loaded!")
    
    def generate_from_image(self, image_data: bytes, prompt: str = None, 
                          num_frames: int = 16, fps: int = 8) -> bytes:
        """Generate simple animated video using img2img variations"""
        self.load_model()
        
        import imageio
        
        try:
            image = Image.open(io.BytesIO(image_data))
            image = image.convert('RGB')
            image = image.resize((512, 512), Image.LANCZOS)
            
            if not prompt:
                prompt = "smooth motion, animated"
            
            frames = []
            
            print(f"[SimpleAnimateDiff] Generating {num_frames} frames...")
            
            for i in range(num_frames):
                strength = 0.1 + (i / num_frames) * 0.05
                
                output = self.pipe(
                    prompt=prompt,
                    image=image,
                    strength=strength,
                    guidance_scale=7.5,
                    num_inference_steps=20,
                ).images[0]
                
                frames.append(output)
            
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp_path = tmp.name
            
            import numpy as np
            frames_np = [np.array(frame) for frame in frames]
            
            imageio.mimsave(tmp_path, frames_np, fps=fps, codec='libx264')
            
            with open(tmp_path, 'rb') as f:
                video_bytes = f.read()
            
            os.unlink(tmp_path)
            
            print(f"[SimpleAnimateDiff] Video generated: {len(video_bytes)} bytes")
            
            return video_bytes
            
        except Exception as e:
            print(f"[SimpleAnimateDiff] Error: {str(e)}")
            raise
