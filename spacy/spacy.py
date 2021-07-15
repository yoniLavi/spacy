from os.path import dirname, realpath, join
import cv2
import numpy as np

class Spacy:
    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        self.abspath = dirname(realpath(__file__))
        self.background = cv2.imread(join(self.abspath, "background.jpg"))
        self.background = cv2.resize(self.background, (self.width, self.height))
        self.frame = self.draw()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.read_width = self.width

        self.correct = 0
        self. wrong = 0
        self.level = 1

    def resize(self, image, width):
        h, w = image.shape[:2]
        r = width / float(w)
        dim = (width, int(h * r))
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    @property
    def canvas(self):
        return self.background.copy()

    @property
    def randomDarkColor(self):
        b,g,r = tuple(np.random.randint(200, size=3))
        return int(b), int(g), int(r)

    @property
    def randomLightColor(self):
        b,g,r = tuple(np.random.randint(200, 255, size=3)) 
        return int(b), int(g), int(r)

    def ufo(self, canvas, center, radius, number=None):
        axes = (radius, radius//2)
        cv2.ellipse(canvas, (center[0], center[1]-radius//3), (radius//2, radius//2), 0,  0,  360, self.randomDarkColor, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, center, axes, 0, 0, 360, self.randomDarkColor, -1, cv2.LINE_AA)
        if number is not None:
            cv2.putText(canvas, str(number), (center[0]-15, center[1]+15), self.font, 1.5, self.randomLightColor, 3, cv2.LINE_AA)
        return canvas
    
    def draw(self):
        while True:
            canvas = self.canvas
            radius = 80
            numbers = np.arange(10)
            cx = np.arange(radius*2, self.width-(radius*2), radius*3)
            cy = np.arange(radius*2, self.height-(radius*2), radius*3)
            np.random.shuffle(numbers)
            np.random.shuffle(cx)
            np.random.shuffle(cy)

            positions = {}
            for i, (n, x, y) in enumerate(zip(numbers, cx[:10], cy[:10])):
                if i == 0 and self.level == 2:
                    # blank ufo
                    self.ufo(canvas, (x,y), radius)
                else:
                    # ufo with number
                    self.ufo(canvas, (x,y), radius, n)
                    positions.update({y:n})

            positions = dict(sorted(positions.items()))
            
            yield canvas, list(positions.values())

    def read(self, width=1920):
        self.read_width = width
        frame, numbers = next(self.frame)
        return self.resize(frame, width), numbers

    def write(self, frame, text, scale=1.2, thickness=2):
        frame = cv2.resize(frame, (self.width, self.height))
        return self.resize(cv2.putText(frame, str(text), (10, self.height-25), self.font, scale, (20, 232, 255), thickness, cv2.LINE_AA), width=self.read_width)

    def submit(self, myanswer, numbers):
        if list(myanswer) == list(numbers):
            self.correct += 1
        else:
            self.wrong += 1

        status = "numbers (top to bottom): {} | ".format(numbers)
        status += "myanswer: {} | ".format(myanswer)
        status += "accuracy: {} %".format(spacy.accuracy())
        return status

    def accuracy(self):
        try:
            return (self.correct/(self.correct+self.wrong))*100
        except ZeroDivisionError:
            return 0

spacy = Spacy()