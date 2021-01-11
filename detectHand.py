import cv2
import matplotlib.pyplot as plt
import numpy as np
# PROJECT: Detect Contour

#Stack image function
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


# read the image
img = cv2.imread("Resources/hand.jpg")
img=cv2.resize(img,(640,480))


# convert to RGB
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
# convert to grayscale
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create a binary thresholded image
# _ (underscore) variable in Python mean a throwaway variable
_, imgBinary = cv2.threshold(imgGray, 225, 255, cv2.THRESH_BINARY_INV)
                    #The second argument is the threshold value which is used to classify the pixel values.
                    # The third argument is the maximum value which is assigned to pixel values exceeding the threshold
                    # So we can play with the second value to get the best of the output

                    #threshold for fissure is 140
                    #threshold for crack3 is 220

# show it
imgStack1=stackImages(1,([img,imgBinary]))
#cv2.imshow("Image Stack 1",imgStack1)

plt.imshow(imgBinary, cmap="gray")
plt.show()


# find the contours from the thresholded image
contours, hierarchy = cv2.findContours(imgBinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

imgBlack=np.ones((480,640,3),np.uint8)
#imgBlack[:]=255
imgBlack.fill(255)
imgWhite=imgBlack


# draw all contours
imgContour = cv2.drawContours(imgWhite, contours, -1, (0, 255, 0), 2)
# plt.imshow(img)
# plt.show()
imgStack=stackImages(1,([img,imgContour]))
cv2.imshow("Image Stack",imgStack)

cv2.waitKey(0)