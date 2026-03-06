import cv2
import numpy as np

def preprocess_slice(slice_25d):

    # Resize
    resized = cv2.resize(slice_25d, (224, 224))

    # Normalize
    normalized = resized / np.max(resized)

    return normalized


if __name__ == "__main__":

    dummy = np.random.rand(240,240,3)

    processed = preprocess_slice(dummy)

    print("Processed shape:", processed.shape)