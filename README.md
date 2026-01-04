# LLaVA Image Analysis System - Setup Guide

This project allows you to run the LLaVA vision-language model in Google Colab (for GPU access) and interact with it from your local machine via FastAPI.

## Architecture

```
┌─────────────────┐         HTTP POST          ┌──────────────────┐
│  Local Machine  │ ─────────────────────────> │  Google Colab    │
│    (app.py)     │  (image + prompt)          │ (vlm_server.py)  │
│                 │                             │                  │
│  FastAPI Client │ <───────────────────────── │  LLaVA Model     │
└─────────────────┘    JSON Response           └──────────────────┘
                                                        │
                                                   ngrok tunnel
```

## Prerequisites

### Local Machine
- Python 3.8+
- FastAPI
- Requests
- Uvicorn

### Google Colab
- Free Google account
- Ngrok account (free tier is fine)

## Setup Instructions

### Step 1: Get Ngrok Auth Token

1. Go to [ngrok.com](https://ngrok.com/)
2. Sign up for a free account
3. Go to [Your Authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)
4. Copy your authtoken

### Step 2: Setup Colab Server

1. **Open Google Colab**
   - Go to [colab.research.google.com](https://colab.research.google.com/)
   - Create a new notebook or upload `vlm_server.py`

2. **Upload the Server File**
   - Upload `vlm_server.py` to Colab
   - Or copy-paste the entire content into a Colab cell

3. **Add Your Ngrok Token**
   - Find this line in `vlm_server.py`:
     ```python
     # ngrok.set_auth_token("YOUR_NGROK_TOKEN_HERE")
     ```
   - Uncomment it and replace with your token:
     ```python
     ngrok.set_auth_token("your_actual_token_here")
     ```

4. **Run the Server**
   - Execute the cell in Colab
   - Wait 2-5 minutes for the model to load
   - Look for output like:
     ```
     ======================================================================
     🌐 PUBLIC URL: https://1234-56-78-90-123.ngrok-free.app
     ======================================================================
     ```
   - **COPY THIS URL!** You'll need it for the next step

### Step 3: Setup Local Client

1. **Install Dependencies**
   ```bash
   pip install fastapi uvicorn requests python-multipart
   ```

2. **Update Configuration**
   - Open `app.py`
   - Find this line:
     ```python
     COLAB_SERVER_URL = "http://localhost:8000"
     ```
   - Replace with your ngrok URL from Step 2:
     ```python
     COLAB_SERVER_URL = "https://1234-56-78-90-123.ngrok-free.app"
     ```

3. **Start Local Server**
   ```bash
   python app.py
   ```
   - Server will start at `http://127.0.0.1:8000`

### Step 4: Test the System

#### Option 1: Using FastAPI Docs (Recommended)

1. Open browser: `http://127.0.0.1:8000/docs`
2. Click on `/v1/analyze` endpoint
3. Click "Try it out"
4. Upload an image file
5. Enter a prompt (or use default)
6. Click "Execute"
7. See the response from LLaVA!

#### Option 2: Using curl

```bash
curl -X POST "http://127.0.0.1:8000/v1/analyze" \
  -F "file=@path/to/your/image.jpg" \
  -F "prompt=What objects are in this image?"
```

#### Option 3: Using Python

```python
import requests

# Test health check first
health = requests.get("http://127.0.0.1:8000/health")
print(health.json())

# Analyze an image
with open("path/to/your/image.jpg", "rb") as f:
    files = {"file": f}
    data = {"prompt": "Describe this image in detail"}
    response = requests.post(
        "http://127.0.0.1:8000/v1/analyze",
        files=files,
        data=data
    )
    print(response.json())
```

## Example Prompts

Try these prompts with your images:

- `"Describe this image in one concise sentence."`
- `"List all objects visible in this image."`
- `"What is the main subject of this image?"`
- `"Is there any text in this image? If yes, transcribe it."`
- `"Describe the mood and atmosphere of this image."`
- `"What colors are dominant in this image?"`

## Troubleshooting

### "Cannot connect to Colab server"
- Make sure Colab server is running
- Check that you updated `COLAB_SERVER_URL` with the correct ngrok URL
- Verify ngrok tunnel is active in Colab output

### "Request timed out"
- Model might still be loading (wait 2-5 minutes on first run)
- Check Colab output for errors
- Try the health check endpoint: `http://127.0.0.1:8000/health`

### "Model not loaded yet"
- Wait for model to finish loading in Colab
- Check Colab output for "Model loaded successfully!" message

### Ngrok URL expired
- Free ngrok URLs expire after 2 hours or when you restart
- Get new URL from Colab output
- Update `COLAB_SERVER_URL` in `app.py`
- Restart local server

## API Endpoints

### Local Server (app.py)

- `GET /` - Server info
- `GET /health` - Health check (checks both local and Colab)
- `POST /v1/analyze` - Analyze image
  - **file**: Image file (required)
  - **prompt**: Text prompt (optional, default: "Describe this image in one concise sentence.")

### Colab Server (vlm_server.py)

- `GET /` - Server info
- `GET /health` - Health check
- `POST /v1/analyze` - Analyze image with LLaVA model

## Response Format

```json
{
  "success": true,
  "filename": "example.jpg",
  "image_size": {
    "width": 1920,
    "height": 1080
  },
  "prompt": "Describe this image",
  "response": "A beautiful sunset over the ocean with orange and pink clouds."
}
```

## Notes

- **GPU**: Colab provides free GPU access, making LLaVA inference much faster
- **Model Size**: LLaVA is ~13GB, first load takes time
- **Session**: Keep Colab tab open while using the system
- **Free Tier**: Colab free tier has usage limits, sessions may disconnect after inactivity
- **Ngrok**: Free tier provides temporary URLs that change on restart

## Files

- `app.py` - Local FastAPI client
- `vlm_server.py` - Colab FastAPI server with LLaVA model
- `vlm.py` - Original Colab notebook (reference only)
- `README_SETUP.md` - This file

## Support

If you encounter issues:
1. Check Colab output for errors
2. Verify ngrok URL is correct and active
3. Test health endpoint first
4. Check that all dependencies are installed
