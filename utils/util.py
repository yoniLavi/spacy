import numpy as np
import cv2

class Transformer:
    def __init__(self):
        pass

    def transform_frame(self, frame, size):
        """
        Transform a video frame to a ready image for the object detector
        """
        # resize the frame for width and aspect_ratio
        aspect_ratio = frame.shape[1] / frame.shape[0]
        shape = (size, int(size / aspect_ratio))
        img_resized = cv2.resize(frame, shape)

        # pad the resized image
        top_pad = int((img_resized.shape[1] - img_resized.shape[0]) / 2)
        bottom_pad = size - img_resized.shape[0] - top_pad
        left_pad = right_pad = 0
        r, g, b = cv2.split(img_resized)
        r = np.pad(r, [(top_pad, bottom_pad), (left_pad, right_pad)], constant_values=0)
        g = np.pad(g, [(top_pad, bottom_pad), (left_pad, right_pad)], constant_values=0)
        b = np.pad(b, [(top_pad, bottom_pad), (left_pad, right_pad)], constant_values=0)
        img = cv2.merge((b, g, r))

        return img

    def transform_image(self, img, size):
        """
        Transform an input image and return a ready image for the classifier
        """
        aspect_ratio = img.shape[1] / img.shape[0]

        if aspect_ratio > 1:
            shape = (size, int(size / aspect_ratio))
        elif aspect_ratio < 1:
            shape = (int(size * aspect_ratio), size)
        else:
            shape = (size, size)

        resized = cv2.resize(img, shape)

        if resized.shape[1] > resized.shape[0]:
            top_pad = int(np.floor((resized.shape[1] - resized.shape[0]) / 2))
            bottom_pad = int(np.ceil((resized.shape[1] - resized.shape[0]) / 2))
            left_pad = right_pad = 0
        elif resized.shape[0] > resized.shape[1]:
            top_pad = bottom_pad = 0
            left_pad = int(np.floor((resized.shape[0] - resized.shape[1]) / 2))
            right_pad = int(np.ceil((resized.shape[0] - resized.shape[1]) / 2))
        else:
            top_pad = bottom_pad = left_pad = right_pad = 0

        b, g, r = cv2.split(resized)
        r = np.pad(r, [(top_pad, bottom_pad), (left_pad, right_pad)], constant_values=0)
        g = np.pad(g, [(top_pad, bottom_pad), (left_pad, right_pad)], constant_values=0)
        b = np.pad(b, [(top_pad, bottom_pad), (left_pad, right_pad)], constant_values=0)
        img = cv2.merge((r, g, b))

        return img

# object initialization
util = Transformer()
