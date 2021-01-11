import numpy as np
import cv2
import math
import matplotlib.pyplot as plt




# Shape of image is accessed by img.shape.
# It returns a tuple of number of rows, columns and channels (if image is color)
# Total number of pixels is accessed by img.size
# Image datatype is obtained by img.dtype

# check of image is Grayscale or RGB
# if Original is not grayscale, convert it to gray scale
def checkGray(img):
    testGray=False
    def isGray(img):
        if len(img.shape) < 3: return True
        if img.shape[2]  == 1: return True
        b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
        if (b==g).all() and (b==r).all(): return True
        return False

    testGray = isGray(img)
    if testGray == True:
        print("The input image is Grayscale? -> ", testGray)
        imgGray=img
    else:
        print("Thu input image is Grayscale? ->",testGray)
        print("I converted it to Grayscale for you!")
        imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return imgGray


path = "Resources/imgContour.jpg"
imgOri = cv2.imread(path)
imgGray = cv2.cvtColor(imgOri,cv2.COLOR_BGR2GRAY)
imgGray = checkGray(imgGray)
imgCanny = cv2.Canny(imgGray, 50, 100, apertureSize=3)
cv2.imshow("img Canny", imgCanny)


# imgRGB=cv2.cvtColor(imgCanny,cv2.COLOR_GRAY2BGR)
# cv2.imshow("img RGB",imgRGB)

# Draw Houghline using Standard Hough Line Transformation
def drawHoughlines(imgOri, imgCanny):
    imgCopy = imgOri.copy()
    # linesP = cv2.HoughLinesP(imgCanny, 1, np.pi / 180, 50, None, 50, 10)

    # Find lines in image using Standard Hough Transform
    # Second and third parameters are \rho and \theta accuracies respectively.
    # Fourth argument is the threshold, which means minimum vote it should get for it to be considered as a line.
    # Remember, number of votes depend upon number of points on the line.
    # So it represents the minimum length of line that should be detected.

    # Standard Hough Transformation gives you as result a vector of couples (rho,theta)
    lines = cv2.HoughLines(imgCanny, 1, np.pi / 180, 100)  # Find lines in image
    # print("roh, theta: ",lines[0])
    # print("Dim of lines is: ",np.shape(lines))
    # print("Dim of lines is: ", lines.ndim)

    if lines is not None:
        # Once we have hough lines. For any one line we draw Houghline since we know the end points
        # (x1, y1) and (x2, y2) of that line.
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(imgCopy, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)  # draw line, AA is anti-alias

            # #or:
            # pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            # pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            # cv2.line(imgCopy, pt1, pt2, (0,255,255), 3,cv2.LINE_AA)
    return imgCopy


# Draw Hough Line using Probabilistic Hough Transformation
def drawHoughlinesP(imgOri, imgCanny):
    imgCopy = imgOri.copy()
    # linesP = cv2.HoughLinesP(imgCanny, 1, np.pi / 180, 50, None, 50, 10)

    # Find lines in image using Probabilistic Hough Line Transform
    # A more efficient implementation of the Hough Line Transform.
    # It gives as output the extremes of the detected lines (x0,y0,x1,y1)

    ## Find lines in image using P Hough Transformation
    linesP = cv2.HoughLinesP(imgCanny, 1, np.pi / 180.0, 100, None, minLineLength=100, maxLineGap=5)
    print("roh, theta: ", linesP[0])

    if linesP is not None:
        for x1, y1, x2, y2 in linesP[0]:
            # Drawing lines
            cv2.line(imgCopy, (x1, y1), (x2, y2), (255, 0, 0), 3)
    return imgCopy


# calculate tilt angle using Standard Hough Transform
def calculateAngle(imgCanny):
    lines = cv2.HoughLines(imgCanny, 1, np.pi / 180, 100, None, 0, 0)
    print("lines[0] is: ", lines[0])
    print("Shape of lines is: ", np.shape(lines))
    print("Dim of lines is:", lines.ndim)
    radianAngle = lines[0][0][1]
    angle = (radianAngle * 180) / np.pi
    print("Tilt Angle found by Standard Hough Transform: ", radianAngle)
    return angle


# calculate tilt angle using Hough P
def calculateAngleP(imgCanny):
    # Probabilistic Houghline transformation
    linesP = cv2.HoughLinesP(imgCanny, 1, np.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
    print("roh, theta: ", linesP[0])

    angles = []
    if linesP is not None:
        # Once we have hough lines. For any one line we can calculate angle of that line since we know the end points
        # (x1, y1) and (x2, y2) of that line.
        for x1, y1, x2, y2 in linesP[0]:
            # calculating tilt angle
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)

    median_angle = np.median(angles)
    print('Median tilt angle found by Probabilistic Hough Transform: ', median_angle)
    return median_angle


imgResult = drawHoughlines(imgOri, imgCanny)
cv2.imshow("Hough line", imgResult)

angle = calculateAngle(imgCanny)

# remember to test img_rotated = ndimage.rotate(img_before, median_angle)

# rotate image with the determined angle by method: Affine Transformation
rows, cols, _ = imgResult.shape

# Create the transformation matrix
center_img = ((cols - 1) / 2.0, (rows - 1) / 2.0)
rot_mat = cv2.getRotationMatrix2D(center_img, angle, 1)

# Pass it to warpAffine function (dst = destination?)
warp_rotate_dst = cv2.warpAffine(imgResult, rot_mat, (cols, rows))

# Display the concatenated image
fg, ax = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 2 columns
for i, image in enumerate([imgOri, imgResult, warp_rotate_dst]):
    ax[i].imshow(image)
    if i == 0:
        ax[i].set_title("Origin Image")
    elif i == 1:
        ax[i].set_title(f"determined tilt angle: {angle}")
    else:
        ax[i].set_title("Rotated image")
plt.show()

# angles = []

# # Draw the lines
# if linesP is not None:
#     for i in range(0, len(linesP)):
#         l = linesP[i][0]
#         cv2.line(imgRGB, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv2.LINE_AA)

# cv2.imshow("Detected lines", imgRGB)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
