from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0  # normalize if needed

    # Create a sequence of 10 identical frames
    sequence = np.stack([img_array] * 10, axis=0)  # Shape: (10, 64, 64, 3)
    sequence = np.expand_dims(sequence, axis=0)    # Shape: (1, 10, 64, 64, 3)

    return sequence
