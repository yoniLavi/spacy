import os
import numpy as np
import cv2
from utils.util import util
from tensorflow.keras.models import load_model


class CustomClassifier:
    def __init__(self, path):
        self.model = load_model(os.path.join(os.path.dirname(os.path.realpath(__file__)), "mobile_net2.h5"))

    def predict_prob(self, images=None, size=224):
        """
        Predict the probabilities of 10 classes per example
        :param images: a list of images
        :return: predicted probabilities of the images
        """
        if images is not None and len(images) > 0 :

            batch_size = len(images)
            batch = [util.transform_image(images[i], size) for i in range(batch_size)]
            batch = np.array(batch).reshape(batch_size, size, size, 3)*(1.0/255)
            probs = self.model.predict(batch, batch_size=batch_size)
            return probs

        return None

# object initialization
classifier2 = CustomClassifier('.')