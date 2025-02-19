import os
import time
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from class_mapping import class_name_mapping  # Import class_name_mapping
from model import NeuralNet
from flask import Flask, request, render_template, jsonify, send_from_directory
from nltk_utils import tokenize, bag_of_words
from werkzeug.utils import secure_filename
import random
import json
from gtts import gTTS
import pygame  # To play the audio

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'secret_key'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize pygame for TTS playback
pygame.init()

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Plant Disease Detection Model
model = models.resnet18(pretrained=False)
num_classes = 39  # Update with actual class count
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, num_classes)

model_path = "plant_village_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Define transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to generate and play speech with unique filenames
def text_to_speech(text):
    timestamp = int(time.time() * 1000)  # Unique identifier using timestamp
    filename = f"audio_{timestamp}.mp3"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    tts = gTTS(text=text, lang='en')
    tts.save(filepath)
    
    pygame.mixer.music.load(filepath)
    pygame.mixer.music.play()
    
    return filename  # Return filename to serve it later

# Image Prediction Function
def predict_image(image_path, model, transform, device, class_mapping):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class_idx = predicted.item()
        predicted_class = list(class_mapping.keys())[predicted_class_idx]
        label = class_mapping.get(predicted_class, "Unknown Class")

    # Convert prediction result to speech and get the filename
    audio_filename = text_to_speech(f"The leaf belongs to a {label}.")
    
    return label, audio_filename

# Load Chatbot Model
def load_chatbot_model(filename='data.pth'):
    data = torch.load(filename)
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    
    return model, all_words, tags

# Chatbot Response Function
def chatbot_response(model, sentence, all_words, tags, intents, device):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, -1)
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.5:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                audio_filename = text_to_speech(response)  # Convert chatbot response to speech
                return response, audio_filename
    else:
        response = "I'm sorry, I couldn't understand that. Please provide more details."
        audio_filename = text_to_speech(response)
        return response, audio_filename

# Route for uploading and predicting plant disease
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part in the request"
        file = request.files['file']
        if file.filename == '':
            return "No file selected for uploading"
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Predict class and get audio filename
            result, audio_filename = predict_image(filepath, model, transform, device, class_name_mapping)
            image_url = filename
            
            return render_template('result.html', label=result, image_url=image_url, audio_filename=audio_filename)
    
    return render_template('index.html')

# Chatbot Route
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.form['message']
    intents_file = 'intents.json'
    model, all_words, tags = load_chatbot_model()
    
    with open(intents_file, 'r') as f:
        intents = json.load(f)
    
    response, audio_filename = chatbot_response(model, user_message, all_words, tags, intents, device)
    return jsonify({'response': response, 'audio_filename': audio_filename})

# Serve Uploaded Files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Serve Audio Files
@app.route('/audio/<filename>')
def get_audio(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
