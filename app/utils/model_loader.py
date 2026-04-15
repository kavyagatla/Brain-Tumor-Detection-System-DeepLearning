import os
import numpy as np


# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

class BrainTumorModel:
    model = None

    @classmethod
    def load(cls):
        # NOTE: Uncomment this block when you have your .h5 file
        # if cls.model is None:
        #     model_path = 'path/to/your/model.h5'
        #     cls.model = load_model(model_path)
        pass

    @classmethod
    def predict(cls, file_path):
        """
        Returns prediction. Currently returns MOCK data for testing UI.
        """
        # --- REAL AI LOGIC (Uncomment later) ---
        # img = image.load_img(file_path, target_size=(150, 150))
        # x = image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        # images = np.vstack([x])
        # classes = cls.model.predict(images, batch_size=10)
        # ... logic to convert class to string ...

        # --- MOCK DATA (For UI Development) ---
        import random
        types = ['Meningioma', 'Glioma', 'Pituitary Tumor', 'No Tumor']
        return {
            'label': random.choice(types),
            'confidence': round(random.uniform(85.0, 99.9), 2)
        }