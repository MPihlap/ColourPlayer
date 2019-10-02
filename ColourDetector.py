import cv2
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist

class ColorFrequencyParser:
    def __init__(self):
        # Maps frequencies to RGB colour values
        self.color_frequency_map = OrderedDict( \
        {
            "c" : (40, 255, 0),
            "c#": (0, 255, 232),
            "d" : (0, 124, 255),
            "d#": (5, 0, 255),
            "e" : (69, 0, 234),
            "f" : (87, 0, 158),
            "f#": (116, 0, 0),
            "g" : (179, 0, 0),
            "g#": (238, 0, 0),
            "a" : (255, 99, 0),
            "a#": (255, 236, 0),
            "b" : (153, 255, 0)
        })
        # self.color_frequency_map = OrderedDict( \
        # {
        #     "red" : (255, 0, 0),
        #     "green": (0, 255, 0),
        #     "yellow": (255, 255, 0),
        #     "blue" : (0, 0, 255)
        # })
        
        self.lab = np.zeros((len(self.color_frequency_map), 1, 3), dtype="uint8")
        self.colorNames = []

        for (i, (name, rgb)) in enumerate(self.color_frequency_map.items()):
            self.lab[i] = rgb
            self.colorNames.append(name)
        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)
    def getFrequency(self, image, contour):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(image, mask=mask)[:3]
        # print("mean: " + str(mean))
        minDist = (np.inf, None)
        for (i, row) in enumerate(self.lab):
            d = dist.euclidean(row[0], mean)
            if d < minDist[0]:
                minDist = (d, i)
        # print(minDist)
        # print(self.lab[minDist[1]][0])
        if minDist[1] != None:
            return self.colorNames[minDist[1]]
        return 
    # Return BGR color value of color corresponding to note
    def getColor(self, note):
        return np.array(list(reversed(cfparser.color_frequency_map.get(note))), dtype=float)/255

    def showPalette(self):
        increment = 0
        for key in self.color_frequency_map:
            newimg = np.zeros((100, 100, 3))
            for i in range(newimg.shape[0]):
                for j in range(newimg.shape[1]):
                    newimg[i][j] = np.array(list(reversed(cfparser.color_frequency_map.get(key))), dtype=float)/255
            print(key, newimg[0][0])
            cv2.imshow(key, newimg)
            cv2.moveWindow(key, increment, 100)
            increment += 200



filenames = ["solid_pink.png", "solid_grey.png", "solid_blue.png", "solid_brown.png"]
cfparser = ColorFrequencyParser()

# note to self - np.fill() vms
newimg = np.zeros((100, 100, 3))
for i in range(newimg.shape[0]):
    for j in range(newimg.shape[1]):
        #newimg[i][j] = [0, 0, 116]
        newimg[i][j] = cfparser.getColor("f#")
print("f#", newimg[0][0])
cv2.imshow("f#", newimg)    
cv2.moveWindow("f#", 100, 100)

newimg2 = np.zeros((100, 100, 3))
for i in range(newimg2.shape[0]):
    for j in range(newimg2.shape[1]):
        newimg2[i][j] = cfparser.getColor("g#")
print("g#", newimg2[0][0])
cv2.imshow("g#", newimg2)
cv2.moveWindow("g#", 300, 100)
cv2.waitKey(0)

cfparser.showPalette()
cv2.waitKey(0)
cv2.destroyAllWindows()

for f in filenames:
    image = cv2.imread("images/"+f)
    print(image.shape)
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 1, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    labimage = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    note = cfparser.getFrequency(labimage, contours[0])
    print(f, note)

    cv2.imshow("Original", image)

    newimg = np.zeros((100, 100, 3))
    for i in range(newimg.shape[0]):
        for j in range(newimg.shape[1]):
            newimg[i][j] = cfparser.getColor(note)
    print(newimg[0][0])
    cv2.imshow("Closest match", newimg)
    cv2.moveWindow("Closest match", 0, 100)
    cv2.waitKey(0)






cv2.destroyAllWindows()