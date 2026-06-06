import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np

def verify_model():
    print("Loading model...")
    try:
        model = tf.keras.models.load_model('model.h5')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Loading MNIST test data...")
    (_, _), (x_test, y_test) = mnist.load_data()
    
    # Preprocess
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_test_cat = tf.keras.utils.to_categorical(y_test)
    
    print("Evaluating model on test set...")
    loss, accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
    
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Test on a few individual samples
    print("\nIndividual Predictions:")
    predictions = model.predict(x_test[:5])
    for i in range(5):
        pred_digit = np.argmax(predictions[i])
        true_digit = y_test[i]
        result = "CORRECT" if pred_digit == true_digit else "WRONG"
        print(f"Sample {i}: Pred={pred_digit}, True={true_digit} -> {result}")

if __name__ == "__main__":
    verify_model()
