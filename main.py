import os
import threading
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, render_template
import json

# Nastavenie aplikácie a ciest
app = Flask(__name__)
DATASET_DIR = "dataset/"
ANOMALY_DIR = os.path.join(DATASET_DIR, "anomalie")
NORMAL_DIR = os.path.join(DATASET_DIR, "normalne")
MODEL_PATH = "autoencoder_model6.keras"
STATS_FILE = "stats.json"
TEMP_DIR = "temp"

# Cesty k adresárom
os.makedirs(ANOMALY_DIR, exist_ok=True)
os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


# Funkcia na načítanie obrázkov
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64)) / 255.0
            images.append(np.expand_dims(img, axis=-1))
    return np.array(images)


# Funkcia na aktualizáciu štatistík
def update_stats(key):
    with open(STATS_FILE, "r") as f:
        stats = json.load(f)

    stats[key] += 1
    stats["total_samples"] += 1

    with open(STATS_FILE, "w") as f:
        json.dump(stats, f)


# Funkcia na preučenie modelu
def retrain_model():
    print("🔄 Preučujem model...")

    # Načítanie obrázkov z priečinkov s anomáliami a normálnymi obrázkami
    anomaly_images = load_images_from_folder(ANOMALY_DIR)
    normal_images = load_images_from_folder(NORMAL_DIR)

    if len(anomaly_images) == 0 or len(normal_images) == 0:
        print("❌ Nedostatok dát na preučenie!")
        return

    # Spojenie anomálnych a normálnych obrázkov pre trénovanie
    X = np.concatenate([anomaly_images, normal_images])
    y = np.array([1] * len(anomaly_images) + [0] * len(normal_images))

    # Načítanie modelu
    model = tf.keras.models.load_model(MODEL_PATH)

    # Tréning modelu
    model.fit(X, y, epochs=5, batch_size=16, validation_split=0.2)

    # Uloženie nového modelu
    model.save(MODEL_PATH)
    print("✅ Model bol úspešne preučený!")


# Route na upload obrázkov
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


# Route na spracovanie odpovedí a presun obrázkov
@app.route('/response', methods=['GET'])
def response():
    answer = request.args.get("answer")
    filename = request.args.get("filename")

    # Debug výpis pre kontrolu
    print(f"Answer: {answer}, Filename: {filename}")

    if not filename:
        return "Filename not provided", 400

    # Získanie cesty k dočasnému súboru
    temp_filepath = os.path.join(TEMP_DIR, filename)

    # Ak odpoveď je "ÁNO" alebo "NIE", presuň obrázok do príslušného priečinka
    if answer == "NIE":
        target_dir = ANOMALY_DIR
        category = "anomalie"
    elif answer == "ANO":
        target_dir = NORMAL_DIR
        category = "normalne"
    else:
        return "Neplatná odpoveď", 400

    # Vytvorenie unikátneho názvu súboru (pridanie čísla na koniec)
    counter = 1
    new_filename = f"{category}_screenshot{counter}.jpg"
    new_filepath = os.path.join(target_dir, new_filename)

    # Zistíme, či už súbor s týmto názvom existuje a zvýšime číslo
    while os.path.exists(new_filepath):
        counter += 1
        new_filename = f"{category}_screenshot{counter}.jpg"
        new_filepath = os.path.join(target_dir, new_filename)

    # Presunutie súboru
    os.rename(temp_filepath, new_filepath)

    # Informácie o presunutí súboru
    msg = f"✅ Súbor bol úspešne presunutý do priečinka '{category}' ako '{new_filename}'."
    print(msg)

    # Tieto štatistiky sa môžu aktualizovať podľa potreby
    if answer == "ANO":
        update_stats("confirmed_anomalies")
    else:
        update_stats("false_alarms")

    # Spustenie preučenia modelu na pozadí
    threading.Thread(target=retrain_model).start()

    return render_template("message.html", message=f"✅ Súbor bol presunutý do priečinka '{category}' ako '{new_filename}'.")


# 🔹 Hlavná stránka (index)
@app.route("/")
def home():
    # Načítanie štatistík zo súboru
    with open(STATS_FILE, "r") as f:
        stats = json.load(f)
    return render_template("index.html", stats=stats)  # Šablóna pre hlavnú stránku


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
