import firebase_admin
from firebase_admin import credentials, firestore, messaging
from fastapi import FastAPI, HTTPException
from typing import List
import aiohttp
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pydantic import BaseModel

# Path to your Firebase service account key file
service_account_key_path = 'lostget-faafe-firebase-adminsdk-b6rif-ec885420b8.json'

cred = credentials.Certificate(service_account_key_path)
firebase_admin.initialize_app(cred)

db = firestore.client()

# retrive imageUrls
def get_document(field_value):
    # Reference to the document in the 'reportItems' collection
    collection_ref = db.collection('reportItems')
    
    return collection_ref.where('id', '==', field_value)

def get_image_url(query_ref):
    # Attempt to get the document
    docs = query_ref.stream()
    
    for doc in docs:
        print(f"Found document with ID: {doc.id}")
        doc_data = doc.to_dict()
        # Retrieve the 'imageUrls' list from the document
        image_urls = doc_data.get('imageUrls', [])
        break
    
    return image_urls

app = FastAPI()

model_path = 'imageclassifier_latest.h5'
model = load_model(model_path)

classes = ['Hentai', 'Neutral', 'Other', 'Porn', 'Sexy']

async def fetch_image(url: str) -> np.ndarray:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail=f"Failed to fetch image from {url}")
            image_data = await response.read()
            nparr = np.frombuffer(image_data, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img_np
        
def send_push_notification(token, title, body):
    # Create a message
    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body,
        ),
        token=token,
    )

    # Send the message
    messaging.send(message)

def get_device_token(uid):
     # Reference to the document in the 'reportItems' collection
    doc_ref = db.collection('fcm_token').document(uid)
    doc = doc_ref.get()
    if doc.exists:
        # Document data
        doc_data = doc.to_dict()
        # Retrieve the 'imageUrls' list from the document
        token = doc_data.get('fcmToken',str)
        return token
    
def update_report(id, update_fields):
    query_ref = get_document(id)
    
    docs = query_ref.stream()
    
    for doc in docs:
        doc_ref = db.collection('reportItems').document(doc.id)
        doc_ref.update(update_fields)

def preprocess_image(image: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resize = tf.image.resize(image_rgb, (256, 256))
    resize = resize / 255.0
    return np.expand_dims(resize, 0)

def predict_image(image: np.ndarray) -> str:
    predictions = model.predict(image)
    highest_index = np.argmax(predictions)
    return classes[highest_index]


class ImageCheckRequest(BaseModel):
    id: str
    uid: str

@app.post("/check-images/")
async def check_images(request: ImageCheckRequest):
    id = request.id
    uid = request.uid
    print(uid)
    query_ref = get_document(id)
    urls = get_image_url(query_ref)
    token = get_device_token(uid)
    print(urls)
    for url in urls:
        try:
            image = await fetch_image(url)
            
            preprocessed_image = preprocess_image(image)
            prediction = predict_image(preprocessed_image)
            if prediction in ['Sexy', 'Hentai', 'Porn']:
                send_push_notification(token, "Issue With your Report", "Your item can't be reported")
                update_report(id, {'flagged': True})
                return {"contains_prohibited_content": True}
        except Exception as e:
            # Handle exceptions, possibly logging them
            continue
    send_push_notification(token, "Report is live now", "Congratulations, Your item report has been published")
    update_report(id, {'published': True})
    return {"contains_prohibited_content": False}

@app.get("/")
async def read_root():
    return {"success": "Obscene Image Classifier is working correctly"}


import uvicorn
from threading import Thread

def run_app():
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")

thread = Thread(target=run_app)
thread.start()

