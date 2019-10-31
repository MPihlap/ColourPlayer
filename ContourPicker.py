import cv2
import numpy as np
import copy

class ContourPicker:
    def __init__(self, image_source):
        self.image = cv2.imread(image_source)
        self.corners = []
        self.contours = [[]]
        self.contour_index = 0
        self.images = []
    def getBlurredImages(self):
        return [np.full(img.shape, cv2.mean(img)[:3])/255 for img in self.images]
    def getImages(self):
        return self.images
    def dist(self, p1, p2):
        return ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
    def showImages(self):
        for i in self.images:
            cv2.imshow("image", i)
    def addOrRemovePoint(self, x, y):
        for i, point in enumerate(self.corners):
            if self.dist((x, y), point) < 10:
                print("Removing point: ", point)
                self.corners.pop(i)
                return
        self.corners.append((x, y))
        self.contours[self.contour_index] = self.corners
        return 

    def mouseClickHandler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.addOrRemovePoint(x, y)

    def drawCorners(self, image, corners):
        for corner in corners:
            cv2.circle(image, corner, 10, (255, 255, 255))
            cv2.circle(image, corner, 8, (0, 0, 0))

    def saveContours(self):
        for i in range(len(self.contours)):
            if self.contours[i] == []:
                continue
            rect = cv2.boundingRect(np.array(self.contours[i]))
            cropped = self.image[rect[1]:rect[1] + rect[3] , rect[0]:rect[0] + rect[2], :]
            self.images.append(cropped)

    def run(self):
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.mouseClickHandler)
        cropped = None
        while True:
            newimg = copy.deepcopy(self.image)
            self.drawCorners(newimg, self.corners)
            if len(self.contours[self.contour_index]) > 0:
                rect = cv2.boundingRect(np.array(self.contours[self.contour_index]))
                cv2.rectangle(newimg, (rect[0], rect[1]),(rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), thickness=2)
            cv2.imshow("image", newimg)

            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                return
            elif key == ord('x') and len(self.contours[0]) > 0:
                self.saveContours()
                return 
            elif key == ord('n'):
                self.contour_index += 1
                self.contours.append([])
                self.corners = []
