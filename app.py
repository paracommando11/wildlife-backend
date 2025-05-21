import cv2
import numpy as np
import google.genai as genai
import requests
import threading
import time
import base64
from google.genai import types
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from twilio.rest import Client
from PIL import Image
from io import BytesIO
from starlette.responses import JSONResponse
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# ✅ Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Configure Google Gemini AI
genai_client = genai.Client(api_key="google api key")

# ✅ Configure Twilio API
TWILIO_SID = os.getenv("ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("AUTH_TOKEN")
TWILIO_PHONE = "enter here"
EMERGENCY_PHONE = "your phone"

client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# 📌 Replace with your IP Webcam URL
phone_camera_url = "enter webacm url"
cap = cv2.VideoCapture(phone_camera_url)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 15)

# ✅ Global Variables
detected_label = "Processing..."
last_captured_frame = None
lock = threading.Lock()

def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

bounding_box_system_instructions = """
    Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
    If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
"""

safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]

def send_alert_sms(message):
    try:
        client.messages.create(
            body=message,
            from_=TWILIO_PHONE,
            to=EMERGENCY_PHONE,
        )
        print(f"📩 SMS Sent: {message}")
    except Exception as e:
        print(f"⚠️ Twilio SMS Error: {str(e)}")

def make_alert_call():
    try:
        call = client.calls.create(
            twiml=f'<Response><Say>Emergency Alert: Please check the situation immediately.</Say></Response>',
            from_=TWILIO_PHONE,
            to=EMERGENCY_PHONE,
        )
        print(f"📞 Call initiated: {call.sid}")
    except Exception as e:
        print(f"⚠️ Twilio Call Error: {str(e)}")

def analyze_with_gemini(image_path):
    global detected_label

    with open(image_path, "rb") as img_file:
        im = Image.open(BytesIO(img_file.read()))
        im.thumbnail([1024, 1024], Image.Resampling.LANCZOS)

    try:
        model_name = "gemini-1.5-flash"
        prompt = (
            "Identify the subject in this image (Tiger, Rhino, Poacher, or Forest Fire), and analyze its condition. "
            "If a poacher is detected, make sure it has a weapon and do not confuse between poacher and forest rangers. Respond clearly as 'Poacher detected'. "
            "If it's an animal, determine if it is healthy or in poor condition (injured, malnourished, or sick). "
            "If a forest fire is detected, respond as 'Forest fire detected'. Provide a brief explanation."
        )

        response = genai_client.models.generate_content(
            model=model_name,
            contents=[prompt, im],
            config=types.GenerateContentConfig(
                system_instruction=bounding_box_system_instructions,
                temperature=0.5,
                safety_settings=safety_settings,
            )
        )

        if response and hasattr(response, "text"):
            detected_label = response.text.strip()
            print(f"🔍 Gemini detected: {detected_label}")

            if "Poacher detected" in detected_label:
                send_alert_sms("⚠️ ALERT: Poacher detected in the forest. Immediate action required!")
                make_alert_call()

            elif "Forest fire detected" in detected_label:
                send_alert_sms("🔥 EMERGENCY: A forest fire has been detected. Authorities must respond urgently!")
                make_alert_call()

            return detected_label

        return "⚠️ No response from Gemini AI"
    except Exception as e:
        return f"⚠️ Gemini AI Error: {str(e)}"

def process_frame():
    global last_captured_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Check the IP Webcam URL.")
            time.sleep(1)
            continue

        with lock:
            last_captured_frame = frame.copy()

        time.sleep(0.1)  # Capture every 100ms

def analyze_frame_periodically():
    while True:
        with lock:
            if last_captured_frame is None:
                time.sleep(5)
                continue

            image_path = "detected_subject.jpg"
            cv2.imwrite(image_path, last_captured_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        analyze_with_gemini(image_path)
        time.sleep(5)  # Analyze every 5 seconds

# ✅ Start Threads
threading.Thread(target=process_frame, daemon=True).start()
threading.Thread(target=analyze_frame_periodically, daemon=True).start()

@app.get("/latest-frame")
def get_latest_frame():
    with lock:
        if last_captured_frame is None:
            return JSONResponse(content={"error": "No frame available yet"}, status_code=404)
        image_base64 = encode_image_to_base64(last_captured_frame)
        return {"image": image_base64, "detection": detected_label}

@app.get("/")
def home():
    return {"message": "Wildlife AI FastAPI is running!"}
