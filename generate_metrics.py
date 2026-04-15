import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- CONFIGURATION ---
# IMPORTANT: Adjust these paths if needed
TEST_DIR = 'dataset/Training'
MODEL_DIR = 'models'
OUTPUT_FILE = 'app/static/metrics_data.json'


def generate():
    # 1. Verify Directories
    if not os.path.exists(TEST_DIR):
        print(f"[ERROR] Test directory not found at: {TEST_DIR}")
        print("Please make sure you have a 'dataset/Testing' folder.")
        return

    if not os.path.exists(MODEL_DIR):
        print(f"[ERROR] Models directory not found at: {MODEL_DIR}")
        return

    # 2. Load Test Data
    print(f"Loading test data from {TEST_DIR}...")
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    try:
        test_gen = test_datagen.flow_from_directory(
            TEST_DIR,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False  # Must be False for correct metrics
        )
    except Exception as e:
        print(f"[ERROR] Failed to load images: {e}")
        return

    if test_gen.samples == 0:
        print("[ERROR] No images found in Testing folder!")
        return

    y_true = test_gen.classes
    labels = list(test_gen.class_indices.keys())

    metrics_data = {}
    model_preds = []

    # 3. Evaluate Individual Models
    model_names = ['vgg16', 'resnet50', 'densenet121']

    for name in model_names:
        print(f"\nEvaluating {name}...")
        path = os.path.join(MODEL_DIR, f'{name}_best.keras')

        if not os.path.exists(path):
            print(f"  [WARNING] Model file not found: {path}")
            print(f"  Skipping {name}...")
            continue

        try:
            model = load_model(path)
            pred = model.predict(test_gen, verbose=1)
            model_preds.append(pred)

            y_pred = np.argmax(pred, axis=1)

            # Calculate metrics
            acc = float(accuracy_score(y_true, y_pred))
            report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
            cm = confusion_matrix(y_true, y_pred).tolist()

            metrics_data[name] = {
                'accuracy': acc,
                'report': report,
                'matrix': cm
            }
            print(f"  Accuracy: {acc * 100:.2f}%")

        except Exception as e:
            print(f"  [ERROR] Failed to evaluate {name}: {e}")

    # 4. Evaluate Ensemble
    if len(model_preds) > 0:
        print("\nEvaluating Ensemble (Average of all models)...")
        # Average the predictions (Soft Voting)
        avg_preds = np.mean(model_preds, axis=0)
        y_ens = np.argmax(avg_preds, axis=1)

        acc = float(accuracy_score(y_true, y_ens))
        report = classification_report(y_true, y_ens, target_names=labels, output_dict=True)
        cm = confusion_matrix(y_true, y_ens).tolist()

        metrics_data['Ensemble'] = {
            'accuracy': acc,
            'report': report,
            'matrix': cm
        }
        print(f"  Ensemble Accuracy: {acc * 100:.2f}%")
    else:
        print("[ERROR] No models were evaluated successfully. Cannot calculate Ensemble.")
        return

    # 5. Save Results
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(metrics_data, f)

    print(f"\n[SUCCESS] Metrics saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    generate()