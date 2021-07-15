import time
import cv2
from spacy.spacy import spacy
read = spacy.read
submit = spacy.submit
accuracy = spacy.accuracy
resize = spacy.resize
write = spacy.write
sleep = lambda x: time.sleep(x)

def show(frame):
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
        raise KeyboardInterrupt