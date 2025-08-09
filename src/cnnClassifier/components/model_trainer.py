# Prerequisite Reminder: For best results and to avoid compatibility warnings,
# ensure your Keras version is compatible with MLflow.
# In your terminal, run: pip install "keras<=3.10.0"

import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
tf.config.run_functions_eagerly(True)
from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path
import mlflow
import mlflow.keras

# Enables eager execution for better debugging and compatibility with MLflow hooks.
tf.config.run_functions_eagerly(True)


class Training:
    """
    This class handles the model training process, including data preparation,
    MLflow integration, training, and saving the final model.
    """
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        """Loads the pre-built and updated base model from the previous stage."""
        print("Loading base model...")
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
        print("Base model loaded successfully.")

    def train_valid_generator(self):
        """Creates training and validation data generators with optional augmentation."""
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode="categorical" # Ensure this is set for classification
        )

        # Validation data generator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # Training data generator
        if self.config.params_is_augmentation:
            print("Data augmentation is enabled.")
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            print("Data augmentation is disabled.")
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Saves the model to the specified path."""
        model.save(path)

    def train(self):
        """
        The main training loop. This function compiles the model, integrates with MLflow,
        and runs the training process.
        """
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
        
        # --- MLflow Setup ---
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
        mlflow.set_experiment("Chest_Cancer_Classification")
        
        with mlflow.start_run():
            # Enable autologging for parameters, metrics, and the model
            mlflow.keras.autolog()

            # --- CRITICAL FIX ---
            # Re-compile the model after loading it. This creates a new optimizer instance
            # and prevents the "Unknown variable" error by linking it to the loaded model.
            print("Re-compiling the model for training...")
            self.model.compile(
                optimizer="adam",  # NOTE: Ensure this matches your params.yaml
                loss="categorical_crossentropy", # NOTE: Ensure this matches your params.yaml
                metrics=["accuracy"]
            )
            print("Model re-compiled successfully.")

            # --- Model Training ---
            print("Starting model training...")
            self.model.fit(
                self.train_generator,
                epochs=self.config.params_epochs,
                steps_per_epoch=self.steps_per_epoch,
                validation_steps=self.validation_steps,
                validation_data=self.valid_generator
            )
            print("Model training completed.")

            # --- Save Final Model ---
            # The model is already saved by autolog(), but this also saves a local copy
            # to the path specified in your configuration for easy access.
            print(f"Saving final model locally to: {self.config.trained_model_path}")
            self.save_model(
                path=self.config.trained_model_path,
                model=self.model
            )
            print("Model saved successfully.")