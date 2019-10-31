import cv2
import numpy as np
import copy

image = cv2.imread("images/"+"tartu-maarja-small-orig.png")
corners = []
contours = [[]]
n_clicks = 0 

def dist(p1, p2):
    return ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5

def addOrRemovePoint(x, y):
    for i, point in enumerate(corners):
        if dist((x, y), point) < 10:
            print("Removing point: ", point)
            corners.pop(i)
            return
    corners.append((x, y))
    contours[0] = corners
    return 

def mouseClickHandler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        print("Click")
        addOrRemovePoint(x, y)

def drawCorners(image, corners):
    for corner in corners:
        cv2.circle(image, corner, 10, (255, 255, 255))
        cv2.circle(image, corner, 8, (0, 0, 0))

cv2.namedWindow("image")
cv2.setMouseCallback("image", mouseClickHandler)
cropped = None
while True:
    newimg = copy.deepcopy(image)
    drawCorners(newimg, corners)
    if len(contours[0]) > 0:
        rect = cv2.boundingRect(np.array(contours))
        cv2.rectangle(newimg, (rect[0], rect[1]),(rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), thickness=2)
    cv2.imshow("image", newimg)

    key = cv2.waitKey(20) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('x') and len(contours[0]) > 0:
        rect = cv2.boundingRect(np.array(contours))
        cropped = image[rect[1]:rect[1] + rect[3] , rect[0]:rect[0] + rect[2], :]
        cv2.destroyAllWindows()
        cv2.imshow("image", cropped)
        cv2.waitKey(0)
        break

# mean_color = tuple([int(i) for i in cv2.mean(cropped)[:3]])
cropped = np.full(cropped.shape, cv2.mean(cropped)[:3])/255

cv2.imshow("image", cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()