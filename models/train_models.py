import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from app.model_factory import build_model

# --- CONFIGURATION ---
# UPDATE THIS PATH to point to your actual Training dataset folder
DATASET_PATH = 'dataset/Training'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 40


def train():
    # 1. Create Data Generators (Data Augmentation)
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2  # Use 20% of training data for validation
    )

    print(f"Loading data from: {DATASET_PATH}")

    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # 2. Train Each Model
    models_to_train = ['vgg16', 'resnet50', 'densenet121']

    if not os.path.exists('models'):
        os.makedirs('models')

    for model_name in models_to_train:
        print(f"\n========================================")
        print(f" TRAINING MODEL: {model_name.upper()}")
        print(f"========================================")

        model = build_model(model_name, num_classes=4)

        # Save only the best model based on validation accuracy
        checkpoint = ModelCheckpoint(
            f'models/{model_name}_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )

        # Stop early if not improving
        # Update the patience to 10
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=val_generator,
            callbacks=[checkpoint, early_stop]
        )
        print(f"Finished training {model_name}. Saved to models/{model_name}_best.keras")


if __name__ == '__main__':
    train()
