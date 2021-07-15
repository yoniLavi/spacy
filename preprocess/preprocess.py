import os
import cv2
import numpy as np
import classifier

class Localize:

    def __init__(self, device):
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model.pkl")

    def process(self, frame):
        """
        Todo:
        preprocess frame to find out the location of UFO's. 
        sort them from top to bottom based on their location,
        crop the UFO image from original frame and pass it to classifier.
        submit the predicted numbers list.
        """

        ################## Please write your code here ###################
 

        ##################################################################
        """
        Example: (how to use classifier)

        Note:-
        1. instead of reading cropped image from directory
        you need to find the locations of UFO's on the frame, crop them
        and sort them from top to bottom,
        pass them through classifier and return predicted numbers

        2. This example below is just for demonstration purpose,
        you can delete it when you write your own code above.
        """
        crop1 = cv2.imread("classifier/numbers/1/1_1.jpg")
        crop2 = cv2.imread("classifier/numbers/2/2_1.jpg")
        crop3 = cv2.imread("classifier/numbers/3/3_1.jpg")
        crop4 = cv2.imread("classifier/numbers/4/4_1.jpg")
		
        myanswer = classifier.predict(images=[crop1, crop2, crop3, crop4])
        return myanswer

localize = Localize('cpu')