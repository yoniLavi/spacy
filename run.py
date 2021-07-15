import cv2
import spacy
from preprocess import localize

while True:
    """
    read frame and corresponding numbers
    the read() method returns:
    1. frame with UFO's having number on it
    2. list of numbers sorted from top to bottom by position.
    """
    frame, numbers = spacy.read(width=900)

    ##### please write your code inside 'preprocess.py' ####
    # try to break your code into different parts using class methods
    # keep it clean and readable, add comments where required
    myanswer = localize.process(frame)
    # Expected output:
    # myanswer = [n1,n2,n3,n4]

    """
    Note:- the classifier is trained on 5000 images using SVM.
    It may happen, that your cropped image get classified as wrong.
    you are welcome to collect your own training data and train your own classifier.
    """

    # submit your answer
    status = spacy.submit(myanswer=myanswer, numbers=numbers)
    frame = spacy.write(frame, status)

    print(status, end="\r")

    # show frame, press q to quit
    spacy.show(frame)
    spacy.sleep(0.3)