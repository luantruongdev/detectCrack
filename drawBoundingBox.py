import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# This program will draw Bounding Box and Contour around a DETECTED CRACK


# create get Contour function
def drawBoundingBox(imgOri, imgCanny):
    #imgOriginal is just for beautiful purpose,
    #The important parameter here is imgCanny
    imgCopy=imgOri.copy()

    #Apply findContours function of cv2 on imgCanny
    # cv2.RETR_EXTERNAL is retrieval mode, it retrieves the extreme outer contour, we can use different mode here
    #cv2.CHAIN_APPROX_NONE is where we can use different method of openCV
    #to request for compressed values or it will reduce the point for you
    #in this case, we're gonna get all the contour, so we use CHAIN_APPROX_NONE
    # Once we have our contour, they're stocked in the contours variable
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # We 're gonna loop through each of our contour, for each contour
    # we're gonna find the edge of it (we find the area of each contour)

    for cnt in contours:
        area=cv2.contourArea(cnt)
        print("Area:",area)

        #we draw the contour on the copy of original Image
        #cv2.drawContours(imgContour,cnt,-1,(255,0,0),3) #-1 cuz we want to draw all contour
        cv2.drawContours(imgCopy, cnt, -1, (0, 0, 255), 2)

        #next we find the Perimeter (Arc Length)
        # Perimeter helps us approximate the corner of our shape
        peri=cv2.arcLength(cnt,True)
        print("Curve Length:",peri)

        # #Next we want to know how many corner point we have by using Contour Approximation
        # so 3 = 3 corner = triangle, 4 = 4 corner = rectangle
        # 0.02*peri is epsilon, play on this value to get desired output:
        approx=cv2.approxPolyDP(cnt,0.02*peri,True)
        print("Approximation:",len(approx))
        objCorner=len(approx)

        ##Next, we create a bounding box around the detected object.
        ##There are 2 types of Bounding Box: Straight Bounding Box and Rotated Bounding Box

        ##Straight Bounding Box, it doesn't consider the rotation of the object. So area of the
        ## bounding rectangle won't be minimum. It is found by the function cv2.boundingRect().
        ## Let (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.

        # x, y, w, h=cv2.boundingRect(approx)
        # cv2.rectangle(imgCopy,(x,y),(x+w,y+h),(0,255,0),2)

        # Rotated Rectangle considers the rotation of the object and it draws Bounding Box
        # with minimum area with function cv2.minAreaRect()
        # It returns a Box2D structure which contains following details:
        # ( center (x,y), (width, height), angle of rotation ).
        # But to draw this rectangle, we need 4 corners of the rectangle.
        # It is obtained by the function cv2.boxPoints()

        rect = cv2.minAreaRect(approx)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(imgCopy, [box], 0, (0, 255, 0), 2)

        # #get angle from created Bounding Box
        # angle=rect[-1]
        #
        # # from https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
        # # the `cv2.minAreaRect` function returns values in the
        # # range [-90, 0); as the rectangle rotates clockwise the
        # # returned angle trends to 0 -- in this special case we
        # # need to add 90 degrees to the angle
        # if angle < -45:
        #     angle = -(90 + angle)
        #
        # # otherwise, just take the inverse of the angle to make
        # # it positive
        # else:
        #     angle = -angle
        # print("Angle determined: ",angle,"degree\n")

    return imgCopy

path= "Resources/crack1.jpg"
imgOri = cv2.imread(path)

# convert to grayscale
imgGray=cv2.cvtColor(imgOri,cv2.COLOR_BGR2GRAY)

#use image threshold
_,imgThresholded = cv2.threshold(imgGray,150,255,cv2.THRESH_BINARY_INV)

imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)

#Apply Canny edge detection
#The output will be a Black background with the Object in White
threshold=50
imgCanny=cv2.Canny(imgBlur,threshold,2*threshold)

imgResult=drawBoundingBox(imgOri,imgCanny)

#savePath='/Users/tnluan/PycharmProjects/detectFissure/Resources/imgContour2.jpg'
#cv2.imwrite(savePath,imgResult)

cv2.imshow("Result",imgResult)
cv2.waitKey(0)

cv2.destroyAllWindows()

# fg, ax = plt.subplots(1, 2, figsize=(15, 10))  # 1 row, 2 columns, figsize is size of figure
# for i, image in enumerate([imgOri, imgResult]):
#     ax[i].imshow(image)
#     if i == 0:
#         ax[i].set_title('Origin Image')
#     else:
#         ax[i].set_title('Result Image')
#
# plt.show()






