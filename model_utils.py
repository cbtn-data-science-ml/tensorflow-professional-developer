import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def create_model_from_kaggle_tfhub(model_url, num_classes=10):
    """
    Creates a Keras model using a TensorFlow Hub model URL from Kaggle.

    Args:
        model_url (str): Kaggle TensorFlow Hub model URL.
        num_classes (int): Number of neurons in the output layer, equal to the number of classes.

    Returns:
        tf.keras.models.Sequential: An uncompiled Keras Sequential model.
    """
    feature_extractor_layer = hub.KerasLayer(model_url, trainable=False, input_shape=(224, 224, 3))
    model = tf.keras.Sequential([
        feature_extractor_layer,
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def plot_loss_and_accuracy(history):
    """
    Plots the loss and accuracy curves for training and testing.

    Args:
        history: History returned by the fit method.
    """
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Custom callback stops training when accuracy exceeds n% threshold
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.2):
            print("\nAccuracy is at 20%, stopping training.")
            self.model.stop_training = True

# Instantiate the callback
callbacks = MyCallback()


# Initialize EarlyStopping callback
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    mode='min'
)

# Initialize ModelCheckpoint callback
model_checkpoint_callback = ModelCheckpoint(
    filepath='./best_model',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)
