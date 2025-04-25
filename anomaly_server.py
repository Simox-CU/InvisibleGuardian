import os
import threading
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, render_template
import json


app = Flask(__name__)
DATASET_DIR = "dataset/"
ANOMALY_DIR = os.path.join(DATASET_DIR, "anomalie")
NORMAL_DIR = os.path.join(DATASET_DIR, "normalne")
MODEL_PATH = "autoencoder_model6.keras"
STATS_FILE = "stats.json"
TEMP_DIR = "temp"


os.makedirs(ANOMALY_DIR, exist_ok=True)
os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)



def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64)) / 255.0
            images.append(np.expand_dims(img, axis=-1))
    return np.array(images)



def update_stats(key):
    with open(STATS_FILE, "r") as f:
        stats = json.load(f)

    stats[key] += 1
    stats["total_samples"] += 1

    with open(STATS_FILE, "w") as f:
        json.dump(stats, f)



def retrain_model():
    print("🔄 Preučujem model...")

    
    anomaly_images = load_images_from_folder(ANOMALY_DIR)
    normal_images = load_images_from_folder(NORMAL_DIR)

    if len(anomaly_images) == 0 or len(normal_images) == 0:
        print("❌ Nedostatok dát na preučenie!")
        return

    
    X = np.concatenate([anomaly_images, normal_images])
    y = np.array([1] * len(anomaly_images) + [0] * len(normal_images))

    
    model = tf.keras.models.load_model(MODEL_PATH)

   
    model.fit(X, y, epochs=5, batch_size=16, validation_split=0.2)

    
    model.save(MODEL_PATH)
    print("✅ Model bol úspešne preučený!")



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Uloženie do dočasného priečinka
    temp_filepath = os.path.join(TEMP_DIR, file.filename)
    file.save(temp_filepath)

    return "File successfully uploaded", 200



@app.route('/response', methods=['GET'])
def response():
    answer = request.args.get("answer")
    filename = request.args.get("filename")

    
    print(f"Answer: {answer}, Filename: {filename}")

    if not filename:
        return "Filename not provided", 400

    
    temp_filepath = os.path.join(TEMP_DIR, filename)

    
    if answer == "NIE":
        target_dir = ANOMALY_DIR
        category = "anomalie"
    elif answer == "ANO":
        target_dir = NORMAL_DIR
        category = "normalne"
    else:
        return "Neplatná odpoveď", 400

    
    counter = 1
    new_filename = f"{category}_screenshot{counter}.jpg"
    new_filepath = os.path.join(target_dir, new_filename)

    
    while os.path.exists(new_filepath):
        counter += 1
        new_filename = f"{category}_screenshot{counter}.jpg"
        new_filepath = os.path.join(target_dir, new_filename)

   
    os.rename(temp_filepath, new_filepath)

   
    msg = f"✅ Súbor bol úspešne presunutý do priečinka '{category}' ako '{new_filename}'."
    print(msg)

    
    if answer == "ANO":
        update_stats("confirmed_anomalies")
    else:
        update_stats("false_alarms")

    
    threading.Thread(target=retrain_model).start()

    return render_template("message.html", message=f"✅ Súbor bol presunutý do priečinka '{category}' ako '{new_filename}'.")



@app.route("/")
def home():
    with open(STATS_FILE, "r") as f:
        stats = json.load(f)
    return render_template("index.html", stats=stats)  


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
