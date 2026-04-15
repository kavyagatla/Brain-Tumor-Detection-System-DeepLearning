import os
import uuid
import json
import numpy as np
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app.extensions import db
from app.models import Scan
from sqlalchemy import func
from datetime import timedelta
from werkzeug.security import check_password_hash

# Tensorflow imports
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

main_bp = Blueprint('main', __name__)

# --- GLOBAL MODEL CACHE ---
loaded_models = {}


def get_model(model_name):
    if model_name not in loaded_models:
        # 1. First way to find the folder (Current Directory)
        path1 = os.path.join(os.getcwd(), 'models', f'{model_name}_best.keras')
        # 2. Second way to find the folder (App Directory)
        path2 = os.path.join(os.path.dirname(current_app.root_path), 'models', f'{model_name}_best.keras')

        target_path = None
        if os.path.exists(path1):
            target_path = path1
        elif os.path.exists(path2):
            target_path = path2

        if target_path:
            print(f"\n[SUCCESS] Found model at: {target_path}")
            try:
                loaded_models[model_name] = load_model(target_path)
            except Exception as e:
                print(f"[CRITICAL ERROR] Model file exists but failed to load: {e}")
        else:
            print(f"\n[ERROR] Could not find {model_name}_best.keras!")
            print(f"   --> I looked in: {path1}")
            print(f"   --> And also in: {path2}")

    return loaded_models.get(model_name)


# --- ROUTES ---

@main_bp.route('/')
def index():
    return render_template('main/index.html')


@main_bp.route('/dashboard')
@login_required
def dashboard():
    scans = Scan.query.filter_by(doctor_id=current_user.id).order_by(Scan.upload_date.desc()).all()
    total_scans = len(scans)

    distribution_query = db.session.query(
        Scan.tumor_type, func.count(Scan.tumor_type)
    ).filter_by(doctor_id=current_user.id).group_by(Scan.tumor_type).all()

    labels = [row[0] for row in distribution_query]
    chart_data = [row[1] for row in distribution_query]

    best_model_name = "Training Pending"
    best_model_acc = 0.0

    json_path = os.path.join(current_app.root_path, 'static', 'metrics_data.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                metrics = json.load(f)
            best_model_name = max(metrics, key=lambda k: metrics[k]['accuracy'])
            best_model_acc = round(metrics[best_model_name]['accuracy'], 2)
        except Exception:
            best_model_name = "Error Loading"

    return render_template('main/dashboard.html',
                           scans=scans,
                           total_scans=total_scans,
                           labels=labels,
                           chart_data=chart_data,
                           best_model=best_model_name,
                           best_acc=best_model_acc,
                           timedelta=timedelta)


@main_bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)

            new_scan = Scan(
                patient_id=request.form.get('patient_id', 'Unknown'),
                filename=unique_filename,
                tumor_type="Processing...",
                confidence=0.0,
                doctor_id=current_user.id
            )
            db.session.add(new_scan)
            db.session.commit()

            return redirect(url_for('main.result', scan_id=new_scan.id))

    return render_template('main/upload.html')


@main_bp.route('/result/<int:scan_id>')
@login_required
def result(scan_id):
    scan = Scan.query.get_or_404(scan_id)

    img_path = os.path.join(current_app.config['UPLOAD_FOLDER'], scan.filename)

    # 1. Preprocess Image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # 2. Prediction Settings
    model_names = ['vgg16', 'resnet50', 'densenet121']
    display_classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

    detailed_results = {}

    # Kotha Variable kachithanga DenseNet priority theeskovali
    densenet_prediction = None
    densenet_confidence = 0.0

    for name in model_names:
        model = get_model(name)
        if model:
            pred = model(img_array, training=False).numpy()
            class_idx = np.argmax(pred)
            confidence = round(float(np.max(pred)) * 100, 2)
            prediction_label = display_classes[class_idx]

            detailed_results[name] = {
                'prediction': prediction_label,
                'confidence': confidence
            }

            # Ikkada DenseNet values ni special ga store chesthunnam
            if name == 'densenet121':
                densenet_prediction = prediction_label
                densenet_confidence = confidence
        else:
            detailed_results[name] = {'prediction': 'Error', 'confidence': 0.0}

    # FIX: Dashboard inka Final Diagnosis ki kachithanga DenseNet theeskuntundi
    if densenet_prediction and (scan.tumor_type == "Processing..." or scan.confidence == 0.0):
        scan.tumor_type = densenet_prediction
        scan.confidence = densenet_confidence
        db.session.commit()

    # 3. Historical Metrics Loading
    metrics_path = os.path.join(current_app.root_path, 'static', 'metrics_data.json')
    historical_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            historical_metrics = json.load(f)

    return render_template('main/result.html',
                           scan=scan,
                           results=detailed_results,
                           metrics=historical_metrics)


@main_bp.route('/metrics')
@login_required
def metrics():
    metrics_path = os.path.join(current_app.root_path, 'static', 'metrics_data.json')
    metrics_data = None

    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
        except Exception as e:
            print(f"Error reading JSON: {e}")

    return render_template('main/metrics.html', metrics=metrics_data)


@main_bp.route('/report/<int:scan_id>')
@login_required
def generate_report(scan_id):
    scan = Scan.query.get_or_404(scan_id)
    ist_time = scan.upload_date + timedelta(hours=5, minutes=30)

    return render_template('main/report_pdf.html',
                           scan=scan,
                           ist_time=ist_time,
                           doctor_name=current_user.username,
                           doctor_email=current_user.email)


@main_bp.route('/delete_scan/<int:scan_id>', methods=['POST'])
@login_required
def delete_scan(scan_id):
    scan = Scan.query.get_or_404(scan_id)

    if scan.doctor_id != current_user.id:
        flash("Unauthorized access.", "danger")
        return redirect(url_for('main.dashboard'))

    entered_password = request.form.get('password')

    # Note: Using .password as per your previous database fix
    if check_password_hash(current_user.password, entered_password):
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], scan.filename)
        if os.path.exists(file_path):
            os.remove(file_path)

        db.session.delete(scan)
        db.session.commit()

        flash("Report deleted successfully.", "success")
    else:
        flash("Incorrect password! Deletion failed.", "danger")

    return redirect(url_for('main.dashboard'))
