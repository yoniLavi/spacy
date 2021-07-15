import os
import glob
import cv2
import numpy as np
from sklearn import svm
import pickle

class svmClassifier:
    def __init__(self, path):
        self.dataset_root = path
        self.classifier = svm.SVC(kernel='linear')
        self.model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model.pkl")
        self.X_test = None
        self.Y_test = None
        self.trained = False
        self.width = 170
        self.height = 115

    def preprocess(self, image):
        if len(image.shape) != 2:
            # convert to grayscale if not already
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if list(image.shape) != [self.height, self.width]:
            # resize image if not already
            image = cv2.resize(image, (self.width, self.height))

        # return 1d array
        return image.flatten()

    def load(self, path, limit=None):
        images, labels = [], []
        image_list = glob.glob(path+"/*/*")
        np.random.shuffle(image_list)

        for i, p in enumerate(image_list):
            images.append(self.preprocess(cv2.imread(p,0)))
            labels.append(os.path.basename(os.path.dirname(p)))
            if limit:
                if i > limit:
                    break
        
        return np.array(images), np.array(labels)

    def split(self, dataset, test_size=0.34):
        # train test split
        n = int(len(dataset[0])*(1-test_size))
        X_train, Y_train = dataset[0][:n], dataset[1][:n]
        X_test, Y_test = dataset[0][n:], dataset[1][n:]
        
        return (X_train, Y_train), (X_test, Y_test)

    def train(self, test_size=0.34, limit=None):
        # dataset preparation
        dataset = self.load(self.dataset_root, limit)
        (X_train, Y_train), (self.X_test, self.Y_test) = self.split(dataset, test_size)

        # training the model
        self.classifier.fit(X_train, Y_train)
        self.trained = True

        # saving model to disk
        with open(self.model_path, "wb") as f:
            f.write(pickle.dumps(self.classifier))
        
        return self.classifier
    
    def predict(self, image=None, images=None, model_path=None):
        if not self.trained:
            model_path = self.model_path if model_path == None else model_path
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    self.classifier = pickle.loads(f.read())
                    self.trained = True
            else:
                print("Trained model: {} not found".format(model_path))

        if image is not None:
            # predicting on single image
            images = [self.preprocess(image)]
        elif images is not None:
            # predicting on batch of images
            images = [self.preprocess(x) for x in images]

        return [eval(x) for x in self.classifier.predict(images)]

# object initialization
classifier = svmClassifier("numbers")

if __name__ == "__main__":
    print("Training model...")
    classifier.train()
    img = cv2.imread("numbers/1/1_1.jpg")
    print(classifier.predict(img))