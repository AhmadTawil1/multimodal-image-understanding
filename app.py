from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import requests
from typing import Optional

# ============================================================================
# CONFIGURATION
# ============================================================================
# IMPORTANT: This URL is from your Colab ngrok tunnel
COLAB_SERVER_URL = "https://kenna-explosible-nonmonistically.ngrok-free.dev"

# ============================================================================
# FastAPI App
# ============================================================================
app = FastAPI(
    title="LLaVA Local Client",
    description="Local client that forwards image analysis requests to Colab server",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LLaVA Local Client",
        "status": "running",
        "colab_server": COLAB_SERVER_URL,
        "endpoints": {
            "analyze": "/v1/analyze",
            "embed": "/v1/embed",
            "cosine_sim": "/v1/cosine-sim",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Check health of both local and Colab servers"""
    try:
        # Check Colab server health
        response = requests.get(f"{COLAB_SERVER_URL}/health", timeout=5)
        colab_health = response.json()
        
        return {
            "local_status": "healthy",
            "colab_status": "connected",
            "colab_details": colab_health
        }
    except requests.exceptions.RequestException as e:
        return {
            "local_status": "healthy",
            "colab_status": "disconnected",
            "error": str(e),
            "message": "Make sure to update COLAB_SERVER_URL with your ngrok URL"
        }

@app.post("/v1/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    prompt: str = Form(default="Describe this image in one concise sentence.")
):
    """
    Analyze an uploaded image using LLaVA model on Colab server
    
    Args:
        file: Image file to analyze
        prompt: Text prompt/instruction for the model (optional)
    
    Returns:
        JSON response with analysis results from LLaVA
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        print(f"Received image: {file.filename}")
        print(f"Prompt: {prompt}")
        print(f"Forwarding to Colab server: {COLAB_SERVER_URL}")
        
        # Read file content
        file_content = await file.read()
        
        # Prepare files and data for forwarding
        files = {
            "file": (file.filename, file_content, file.content_type)
        }
        data = {
            "prompt": prompt
        }
        
        # Forward request to Colab server
        response = requests.post(
            f"{COLAB_SERVER_URL}/v1/analyze",
            files=files,
            data=data,
            timeout=60  # 60 second timeout for model inference
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        # Return the response from Colab
        result = response.json()
        print(f"Response from Colab: {result.get('response', 'N/A')}")
        
        return result
        
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Colab server at {COLAB_SERVER_URL}. "
                   f"Make sure the server is running and COLAB_SERVER_URL is correct."
        )
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Request to Colab server timed out. The model might be loading."
        )
    except requests.exceptions.HTTPError as e:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Colab server error: {response.text}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@app.post("/v1/embed")
async def embed_image(
    file: UploadFile = File(...)
):
    """
    Generate embeddings for an uploaded image using CLIP model on Colab server
    
    Args:
        file: Image file to generate embeddings for
    
    Returns:
        JSON response with embedding vector from CLIP model
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        print(f"Received image for embedding: {file.filename}")
        print(f"Forwarding to Colab server: {COLAB_SERVER_URL}")
        
        # Read file content
        file_content = await file.read()
        
        # Prepare files for forwarding
        files = {
            "file": (file.filename, file_content, file.content_type)
        }
        
        # Forward request to Colab server
        response = requests.post(
            f"{COLAB_SERVER_URL}/v1/embed",
            files=files,
            timeout=60  # 60 second timeout for embedding generation
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        # Return the response from Colab
        result = response.json()
        print(f"Embedding generated successfully, shape: {result.get('embedding_shape', 'N/A')}")
        
        return result
        
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Colab server at {COLAB_SERVER_URL}. "
                   f"Make sure the server is running and COLAB_SERVER_URL is correct."
        )
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Request to Colab server timed out. The model might be loading."
        )
    except requests.exceptions.HTTPError as e:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Colab server error: {response.text}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@app.post("/v1/cosine-sim")
async def calculate_cosine_similarity(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    """
    Calculate cosine similarity between two uploaded images using CLIP model on Colab server
    
    Args:
        file1: First image file
        file2: Second image file
    
    Returns:
        JSON response with cosine similarity score from CLIP model
    """
    try:
        # Validate file types
        if not file1.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File1 must be an image")
        if not file2.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File2 must be an image")
        
        print(f"Received images for similarity: {file1.filename} and {file2.filename}")
        print(f"Forwarding to Colab server: {COLAB_SERVER_URL}")
        
        # Read file contents
        file1_content = await file1.read()
        file2_content = await file2.read()
        
        # Prepare files for forwarding
        files = [
            ("file1", (file1.filename, file1_content, file1.content_type)),
            ("file2", (file2.filename, file2_content, file2.content_type))
        ]
        
        # Forward request to Colab server
        response = requests.post(
            f"{COLAB_SERVER_URL}/v1/cosine-sim",
            files=files,
            timeout=60  # 60 second timeout for embedding generation and similarity calculation
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        # Return the response from Colab
        result = response.json()
        print(f"Cosine similarity: {result.get('cosine_similarity', 'N/A')}")
        
        return result
        
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Colab server at {COLAB_SERVER_URL}. "
                   f"Make sure the server is running and COLAB_SERVER_URL is correct."
        )
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Request to Colab server timed out. The model might be loading."
        )
    except requests.exceptions.HTTPError as e:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Colab server error: {response.text}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("Starting LLaVA Local Client")
    print("=" * 70)
    print(f"Colab Server: {COLAB_SERVER_URL}")
    print(f"Local Server: http://127.0.0.1:8001")
    print("=" * 70)
    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=True)