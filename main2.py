# main.py
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ===============================
# 1. Load Dataset
# ===============================
dataset_path = r"D:\FakeClothing\dataset\fake_clothing_scams.csv"
df = pd.read_csv(dataset_path)

print("✅ Dataset loaded successfully! Shape:", df.shape)

# ===============================
# 2. Load CNN Model
# ===============================
cnn_model_path = r"D:\FakeClothing\cnn_logo_model.keras"
cnn_model = tf.keras.models.load_model(cnn_model_path)
print("✅ Loaded CNN model from .keras file")

# ===============================
# 3. Feature Preparation
# ===============================
X_tabular, X_ensemble, y = [], [], []

for idx, row in df.iterrows():
    img_path = os.path.join(r"D:\FakeClothing\dataset\images", row.get("image_path", ""))
    logo_conf = 0.0

    # CNN Logo Prediction
    try:
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=(128, 128))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            pred = cnn_model.predict(img_array, verbose=0)[0][0]
            logo_conf = float(pred)
    except Exception as e:
        print(f"⚠️ Error processing {img_path}: {e}")

    brand_conf = float(row.get("label_brand", 0))
    qr_conf = float(row.get("label_qr", 0))

    label = row["Scam_Flag"]

    X_tabular.append([brand_conf, qr_conf])
    X_ensemble.append([logo_conf, brand_conf, qr_conf])
    y.append(label)

X_tabular = np.array(X_tabular)
X_ensemble = np.array(X_ensemble)
y = np.array(y)

# ===============================
# 4. Train/Test Split
# ===============================
X_train_tab, X_test_tab, y_train, y_test = train_test_split(
    X_tabular, y, test_size=0.2, random_state=42, stratify=y
)

X_train_ens, X_test_ens, _, _ = train_test_split(
    X_ensemble, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 5. Train Meta-Models
# ===============================
tab_model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
tab_model.fit(X_train_tab, y_train)
y_pred_tab = tab_model.predict(X_test_tab)

ens_model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
ens_model.fit(X_train_ens, y_train)
y_pred_ens = ens_model.predict(X_test_ens)

# ===============================
# 6. Evaluation
# ===============================
print("\n📊 Tabular-only Accuracy:", round(accuracy_score(y_test, y_pred_tab) * 100, 2), "%")
print("Classification Report:\n", classification_report(y_test, y_pred_tab, zero_division=0))

print("\n📊 Ensemble (Images + Tabular + QR) Accuracy:", round(accuracy_score(y_test, y_pred_ens) * 100, 2), "%")
print("Classification Report:\n", classification_report(y_test, y_pred_ens, zero_division=0))

# ===============================
# 7. Sample Predictions
# ===============================
print("\n🔍 Predictions on sample products:")
for idx, row in df.head(5).iterrows():
    brand_conf = float(row.get("label_brand", 0))
    qr_conf = float(row.get("label_qr", 0))
    logo_conf = 0.0
    img_path = os.path.join(r"D:\FakeClothing\dataset\images", row.get("image_path", ""))

    if os.path.exists(img_path):
        try:
            img = load_img(img_path, target_size=(128, 128))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            pred = cnn_model.predict(img_array, verbose=0)[0][0]
            logo_conf = float(pred)
        except:
            pass

    final_pred = ens_model.predict([[logo_conf, brand_conf, qr_conf]])[0]
    label = "Fake" if final_pred == 1 else "Real"
    print(f"Product: {row.get('Product_Name','Unknown')}, Prediction: {label}")
