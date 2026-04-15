import cv2
import os
import numpy as np
from flask import current_app


def generate_preprocessing_preview(image_path, filename):
    """
    Creates a 'medical view' of the scan:
    1. Grayscale
    2. High Contrast (CLAHE)
    3. Heatmap (Tumor Highlight)
    """
    img = cv2.imread(image_path)

    # 1. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) - Medical standard
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    # 3. Apply Heatmap for visualization
    heatmap = cv2.applyColorMap(contrast, cv2.COLORMAP_JET)

    # Save processed file
    save_path = os.path.join(current_app.config['PROCESSED_FOLDER'], 'proc_' + filename)
    cv2.imwrite(save_path, heatmap)

    return 'proc_' + filename