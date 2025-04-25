# ------------ YOUR IMPORTS (same as you wrote) ------------

import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import webbrowser
import threading

app = Flask(__name__)
CORS(app)

# ------------ CONFIGURATION ------------

CROP_CSV_PATH = r"C:\Users\LENOVO\archive\Crop_recommendation.csv"
TRAIN_DIR = r"C:\Users\LENOVO\plant_disease_dataset\Train"
VAL_DIR = r"C:\Users\LENOVO\plant_disease_dataset\Validation"

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

CROP_GUIDELINES = {
    'rice': {
        'season': 'Kharif (June - November)',
        'soil': 'Clayey, Loamy soil with good water retention',
        'fertilizer': 'Urea, DAP, Potash',
        'watering': 'Requires flooded fields during early growth',
        'precautions': 'Protect from pests like stem borers and blast disease'
    },
    'wheat': {
        'season': 'Rabi (November - April)',
        'soil': 'Well-drained Loamy or Clay loam soil',
        'fertilizer': 'Nitrogen-rich fertilizers',
        'watering': 'Regular irrigation needed especially at flowering stage',
        'precautions': 'Control rust disease and aphids'
    },
    # Add other crops too...
}

# ------------ TRAINING FUNCTIONS ------------

def train_crop_model():
    df = pd.read_csv(CROP_CSV_PATH)
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, 'crop_model.pkl')

def train_disease_model():
    datagen = ImageDataGenerator(rescale=1./255)
    train_data = datagen.flow_from_directory(TRAIN_DIR, target_size=(150,150), batch_size=32, class_mode='categorical')
    val_data = datagen.flow_from_directory(VAL_DIR, target_size=(150,150), batch_size=32, class_mode='categorical')

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(train_data.class_indices), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, validation_data=val_data, epochs=5)
    model.save('plant_disease_model.keras')

    with open('disease_labels.txt', 'w') as f:
        for label in train_data.class_indices:
            f.write(label + '\n')

def load_models():
    global crop_model, disease_model, disease_labels
    crop_model = joblib.load('crop_model.pkl')
    disease_model = load_model('plant_disease_model.keras')
    with open('disease_labels.txt') as f:
        disease_labels = [line.strip() for line in f.readlines()]

# ------------ FILE CHECK ------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------ COMMON STYLES ------------

COMMON_STYLES = '''
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap" rel="stylesheet">
<style>
    body { margin: 0; font-family: 'Poppins', sans-serif; background: url('your-farming-background.jpg') no-repeat center center/cover; height: 100vh; color: white; }
    .overlay { background-color: rgba(0, 0, 0, 0.5); height: 100%; width: 100%; position: absolute; top: 0; left: 0; }
    .content { position: relative; text-align: center; top: 20%; z-index: 1; }
    h1, h2 { animation: fadeInDown 1s ease-in-out; }
    p, form { animation: fadeInUp 1s ease-in-out; }
    input, button { margin: 8px; padding: 10px; border-radius: 8px; border: none; }
    input[type="submit"], button { background: #4CAF50; color: white; cursor: pointer; transition: background 0.3s, transform 0.3s; }
    input[type="submit"]:hover, button:hover { background: #45a049; transform: scale(1.05); }
    @keyframes fadeInDown { from { opacity: 0; transform: translateY(-50px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(50px); } to { opacity: 1; transform: translateY(0); } }
</style>
'''

# ------------ ROUTES ------------

@app.route('/', methods=['GET'])
def index():
    return render_template_string(f'''
    <!DOCTYPE html>
    <html><head><meta charset="UTF-8"><title>Smart Agriculture</title>{COMMON_STYLES}</head>
    <body><div class="overlay"></div><div class="content">
    <h1>Empowering Farmers, Nurturing Nature 🌱</h1>
    <p>Smart solutions for predicting crops and protecting plants. Let's grow the future together!</p>
    <div class="buttons">
        <button onclick="window.location.href='/crop'">Predict Crop</button>
        <button onclick="window.location.href='/disease'">Detect Disease</button>
    </div>
    </div></body></html>
    ''')

@app.route('/crop', methods=['GET'])
def crop_page():
    return render_template_string(f'''
    <html><head><title>Crop Prediction</title>{COMMON_STYLES}</head><body><div class="overlay"></div>
    <div class="content">
    <h2>Crop Prediction</h2>
    <form action="/predict_crop" method="post">
        <input type="text" name="N" placeholder="Nitrogen" required><br>
        <input type="text" name="P" placeholder="Phosphorous" required><br>
        <input type="text" name="K" placeholder="Potassium" required><br>
        <input type="text" name="temperature" placeholder="Temperature (°C)" required><br>
        <input type="text" name="humidity" placeholder="Humidity (%)" required><br>
        <input type="text" name="ph" placeholder="pH" required><br>
        <input type="text" name="rainfall" placeholder="Rainfall (mm)" required><br><br>
        <input type="submit" value="Predict Crop">
    </form>
    <button onclick="window.location.href='/'">Back to Home</button>
    </div></body></html>
    ''')

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        features = [
            float(request.form['N']),
            float(request.form['P']),
            float(request.form['K']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['ph']),
            float(request.form['rainfall'])
        ]
        prediction = crop_model.predict([features])[0]
        guidelines = CROP_GUIDELINES.get(prediction.lower(), None)

        if guidelines:
            guideline_html = f'''
            <h3>📝 Farming Guidelines:</h3>
            <ul>
                <li><b>Best Season:</b> {guidelines['season']}</li>
                <li><b>Recommended Soil:</b> {guidelines['soil']}</li>
                <li><b>Fertilizer Suggestions:</b> {guidelines['fertilizer']}</li>
                <li><b>Water Requirements:</b> {guidelines['watering']}</li>
                <li><b>Precautions:</b> {guidelines['precautions']}</li>
            </ul>
            '''
        else:
            guideline_html = "<p>No specific guidelines available for this crop.</p>"

        return f'''
        <html><head>{COMMON_STYLES}</head><body><div class="overlay"></div>
        <div class="content">
        <h1>✅ Recommended Crop: {prediction}</h1>
        {guideline_html}
        <br><button onclick="window.location.href='/'">Back to Home</button>
        </div></body></html>
        '''
    except Exception as e:
        return f"<h1>❌ Error: {e}</h1><a href='/'>Back to Home</a>"

@app.route('/disease', methods=['GET'])
def disease_page():
    return render_template_string(f'''
    <html><head><title>Disease Detection</title>{COMMON_STYLES}</head><body><div class="overlay"></div>
    <div class="content">
    <h2>Plant Disease Detection</h2>
    <form action="/predict_disease" method="post" enctype="multipart/form-data">
        <input type="file" name="file" required><br><br>
        <input type="submit" value="Detect Disease">
    </form>
    <button onclick="window.location.href='/'">Back to Home</button>
    </div></body></html>
    ''')

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    if 'file' not in request.files:
        return "❌ No file uploaded"
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded.jpg')
        file.save(filepath)

        try:
            img = load_img(filepath, target_size=(150, 150))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            predictions = disease_model.predict(img_array)
            index = np.argmax(predictions[0])
            confidence = predictions[0][index]
            label = disease_labels[index]

            return f'''
            <html><head>{COMMON_STYLES}</head><body><div class="overlay"></div>
            <div class="content">
            <h1>🦠 Detected Disease: {label}</h1>
            <h3>Confidence: {confidence:.2f}</h3>
            <button onclick="window.location.href='/'">Back to Home</button>
            </div></body></html>
            '''
        except Exception as e:
            return f"<h1>❌ Prediction Error: {e}</h1><a href='/'>Back to Home</a>"
        finally:
            os.remove(filepath)
    else:
        return "❌ Invalid file type"

# ------------ AUTO OPEN BROWSER ------------

def open_browser():
    webbrowser.open("http://127.0.0.1:5000/")

# ------------ MAIN ------------

if __name__ == '__main__':
    print("📊 Training crop model...")
    train_crop_model()

    print("🌱 Training disease model...")
    train_disease_model()

    print("🔄 Loading models...")
    load_models()

    print("🚀 Starting app on http://127.0.0.1:5000")
    threading.Timer(1.5, open_browser).start()
    app.run(debug=False)
