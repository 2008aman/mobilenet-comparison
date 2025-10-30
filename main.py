import time
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import (
    MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large,
    mobilenet, mobilenet_v2, mobilenet_v3
)
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# ======= CONFIG =======
IMG_PATH = "dog.jpg"   # replace with your own image path

# ======= PREPARE IMAGE =======
img = image.load_img(IMG_PATH, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)  # shape (1, 224, 224, 3)

# ======= DEFINE MODELS & PREPROCESSORS =======
models = {
    "MobileNetV1": (MobileNet(weights="imagenet"), mobilenet.preprocess_input),
    "MobileNetV2": (MobileNetV2(weights="imagenet"), mobilenet_v2.preprocess_input),
    "MobileNetV3Small": (MobileNetV3Small(weights="imagenet"), mobilenet_v3.preprocess_input),
    "MobileNetV3Large": (MobileNetV3Large(weights="imagenet"), mobilenet_v3.preprocess_input),
}

results = []

# ======= RUN PREDICTIONS =======
for name, (model, preprocess_fn) in models.items():
    print(f"\n🔹 Evaluating {name} ...")

    # Apply correct preprocessing
    x_pre = preprocess_fn(x.copy())

    # Time the inference
    start = time.time()
    preds = model.predict(x_pre, verbose=0)
    elapsed = time.time() - start

    decoded = decode_predictions(preds, top=3)[0]
    label, confidence = decoded[0][1], decoded[0][2] * 100

    print(f"Top Prediction: {label} ({confidence:.2f}%)")
    print(f"Inference Time: {elapsed:.3f}s")
    print(f"Parameters: {model.count_params()/1e6:.2f}M")

    results.append({
        "Model": name,
        "Top Label": label,
        "Confidence (%)": confidence,
        "Time (s)": elapsed,
        "Params (M)": model.count_params()/1e6,
    })

# ======= SUMMARY TABLE =======
print("\n📊 COMPARISON SUMMARY:")
print(f"{'Model':<20}{'Confidence (%)':<18}{'Time (s)':<10}{'Params (M)':<10}")
for r in results:
    print(f"{r['Model']:<20}{r['Confidence (%)']:<18.2f}{r['Time (s)']:<10.3f}{r['Params (M)']:<10.2f}")
