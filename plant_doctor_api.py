#!/usr/bin/env python3
"""
Plant Doctor API - FastAPI wrapper for Jetson plant disease detection model
Single-threaded, blocking API for memory efficiency on Jetson devices
"""

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os
import gc
import time
import base64
import tempfile
import threading
from pathlib import Path
from PIL import Image
import torch
from typing import Optional
from contextlib import asynccontextmanager

# Jetson optimizations
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
os.environ["CUDA_CACHE_DISABLE"] = "0"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from vllm import LLM, SamplingParams, EngineArgs
from transformers import AutoTokenizer, AutoImageProcessor

# Configuration
MODEL_PATH = "/mnt/nvme/workspace/plant-examiner/model"
TEMP_DIR = "/tmp/plant_doctor"

# Global model instance and lock
model_instance = None
model_lock = threading.Lock()

class AnalysisRequest(BaseModel):
    image_base64: str
    language: str = "English"

class AnalysisResponse(BaseModel):
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    processing_time: float

class ModelManager:
    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self.image_processor = None
        self.loaded = False
    
    def get_compatible_dtype(self):
        """Determine the best dtype for this model on Jetson"""
        print("Checking dtype compatibility...")
        bf16 = torch.cuda.is_bf16_supported()
        print(f"BFloat16 supported: {bf16}")
        return "bfloat16" if bf16 else "auto"
    
    def load_model(self):
        """Load the model, tokenizer, and image processor"""
        if self.loaded:
            return True
            
        print("Loading Plant Doctor model...")
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
            print("✅ Tokenizer loaded")
            
            # Load image processor (optional)
            try:
                self.image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
                print("✅ Image processor loaded")
            except Exception as e:
                print(f"⚠️ Could not load image processor: {e}")
                self.image_processor = None
            
            # Load vLLM model
            dtype = self.get_compatible_dtype()
            args = EngineArgs(
                model=MODEL_PATH,
                dtype=dtype,
                max_model_len=2048,
                gpu_memory_utilization=0.75,
                enforce_eager=True,
                disable_custom_all_reduce=True,
                limit_mm_per_prompt={"image": 1},
            )
            
            print("Loading vLLM model...")
            start_time = time.time()
            self.llm = LLM(**vars(args))
            load_time = time.time() - start_time
            print(f"✅ Model loaded in {load_time:.2f}s")
            
            # Cleanup
            torch.cuda.empty_cache()
            gc.collect()
            
            self.loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def analyze_image(self, image_path: str, language: str = "English") -> tuple:
        """Analyze plant image and return diagnosis"""
        if not self.loaded:
            return False, "Model not loaded", 0
            
        instruction = (
            f"Examine this image returning information about the plant "
            f"and any state (disease or healthy), return text in {language}"
        )
        
        print(f"Analyzing image: {image_path}")
        print(f"Language: {language}")
        
        try:
            # Load image
            img = Image.open(image_path).convert("RGB")
            
            # Create multimodal prompt with correct token format for Gemma-3n
            prompt_text = f"<bos><start_of_turn>user\n<image_soft_token>{instruction}<end_of_turn>\n<start_of_turn>model\n"
            
            # Create multimodal request
            multimodal_request = {
                "prompt": prompt_text,
                "multi_modal_data": {"image": img}
            }
            
            sampling = SamplingParams(
                max_tokens=256,
                temperature=0.7,
                top_p=0.95,
                top_k=16,
                stop=["<end_of_turn>"],
                skip_special_tokens=True,
            )
            
            print("🤖 Generating analysis...")
            start_time = time.time()
            output = self.llm.generate([multimodal_request], sampling)
            processing_time = time.time() - start_time
            
            response = output[0].outputs[0].text.strip()
            print(f"✅ Analysis completed in {processing_time:.2f}s")
            
            # Cleanup
            torch.cuda.empty_cache()
            gc.collect()
            
            return True, response, processing_time
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return False, str(e), 0

# Global model manager
model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🌱 Starting Plant Doctor API...")
    
    # Create temp directory
    Path(TEMP_DIR).mkdir(exist_ok=True)
    
    # Load model
    if not model_manager.load_model():
        print("❌ Failed to load model on startup")
        raise RuntimeError("Model loading failed")
    
    print("✅ Plant Doctor API ready!")
    yield
    
    # Shutdown
    print("🧹 Shutting down...")
    if model_manager.llm:
        del model_manager.llm
    if model_manager.tokenizer:
        del model_manager.tokenizer
    if model_manager.image_processor:
        del model_manager.image_processor
    torch.cuda.empty_cache()
    gc.collect()
    print("✅ Cleanup completed")

# FastAPI app
app = FastAPI(
    title="Plant Doctor API",
    description="AI-powered plant disease detection API for Jetson devices",
    version="1.0.0",
    lifespan=lifespan
)

# Allow *everything* for CORS:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # allow any origin
    allow_credentials=True,         # allow cookies, authorization headers, etc.
    allow_methods=["*"],            # allow all HTTP methods (GET, POST, OPTIONS…)
    allow_headers=["*"],            # allow any request headers
)

def save_base64_image(base64_data: str) -> str:
    """Save base64 image to temporary file"""
    try:
        # Remove data URL prefix if present
        if base64_data.startswith('data:image'):
            base64_data = base64_data.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_data)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix='.jpg', 
            dir=TEMP_DIR
        )
        temp_file.write(image_data)
        temp_file.close()
        
        return temp_file.name
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Plant Doctor API",
        "version": "1.0.0",
        "status": "ready" if model_manager.loaded else "loading"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    gpu_info = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_info = {
            "gpu_name": props.name,
            "total_memory_gb": props.total_memory / 1e9,
            "allocated_memory_gb": torch.cuda.memory_allocated(0) / 1e9,
            "reserved_memory_gb": torch.cuda.memory_reserved(0) / 1e9,
        }
    
    return {
        "status": "healthy",
        "model_loaded": model_manager.loaded,
        "cuda_available": torch.cuda.is_available(),
        "gpu_info": gpu_info
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_plant(request: AnalysisRequest):
    """Analyze plant image for disease detection"""
    
    # Use lock to ensure single-threaded processing
    with model_lock:
        start_time = time.time()
        temp_image_path = None
        
        try:
            # Check if model is loaded
            if not model_manager.loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            # Save base64 image to temp file
            temp_image_path = save_base64_image(request.image_base64)
            
            # Analyze image
            success, result, processing_time = model_manager.analyze_image(
                temp_image_path, 
                request.language
            )
            
            total_time = time.time() - start_time
            
            if success:
                return AnalysisResponse(
                    success=True,
                    result=result,
                    processing_time=total_time
                )
            else:
                return AnalysisResponse(
                    success=False,
                    error=result,
                    processing_time=total_time
                )
                
        except HTTPException:
            raise
        except Exception as e:
            total_time = time.time() - start_time
            return AnalysisResponse(
                success=False,
                error=str(e),
                processing_time=total_time
            )
        finally:
            # Cleanup temp file
            if temp_image_path and os.path.exists(temp_image_path):
                try:
                    os.unlink(temp_image_path)
                except Exception as e:
                    print(f"Warning: Could not delete temp file {temp_image_path}: {e}")

@app.post("/analyze-file")
async def analyze_plant_file(
    file: UploadFile = File(...),
    language: str = Form("English")
):
    """Analyze plant image uploaded as file"""
    
    # Use lock to ensure single-threaded processing
    with model_lock:
        start_time = time.time()
        temp_image_path = None
        
        try:
            # Check if model is loaded
            if not model_manager.loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            # Validate file type
            if not file.content_type or not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            # Save uploaded file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix='.jpg', 
                dir=TEMP_DIR
            )
            
            content = await file.read()
            temp_file.write(content)
            temp_file.close()
            temp_image_path = temp_file.name
            
            # Analyze image
            success, result, processing_time = model_manager.analyze_image(
                temp_image_path, 
                language
            )
            
            total_time = time.time() - start_time
            
            if success:
                return AnalysisResponse(
                    success=True,
                    result=result,
                    processing_time=total_time
                )
            else:
                return AnalysisResponse(
                    success=False,
                    error=result,
                    processing_time=total_time
                )
                
        except HTTPException:
            raise
        except Exception as e:
            total_time = time.time() - start_time
            return AnalysisResponse(
                success=False,
                error=str(e),
                processing_time=total_time
            )
        finally:
            # Cleanup temp file
            if temp_image_path and os.path.exists(temp_image_path):
                try:
                    os.unlink(temp_image_path)
                except Exception as e:
                    print(f"Warning: Could not delete temp file {temp_image_path}: {e}")

if __name__ == "__main__":
    import uvicorn
    
    print("🌱 Starting Plant Doctor API Server...")
    uvicorn.run(
        "plant_doctor_api:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for memory efficiency
        reload=False
    )