#   ------------------- Assignment_Two ------------------
#       Student Name:		Conor Kiernan
#       Student Number:		13512343
#
#	    Please note that I did this assignnment on windows
#	    and through the PyCharm IDE. I couldn't use the
#	    standard cv2 distribution due to an error with
#	    Windows 10. Instead I used a wrapper for cv2 that
#	    should function exactly the same. I'll convert it
#	    back to the standard library before submitting, but
#	    If you see cv2'wrap instead of cv2 anywhere it means
#	    I missed one. Other than that it should be fine.
#   -----------------------------------------------------

import numpy as np
import cv2wrap
import glob
import sys

#   -------------------------------- Section_Zero ------------------------------
#   		    	Set up parameters and criteria for other sections.
#   	    		Necessary for code to work.
#	    			Some code here is taken from the code given on Moodle.
#   ----------------------------------------------------------------------------

#   Termination criteria.
criteria = (cv2wrap.TERM_CRITERIA_EPS + cv2wrap.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#   Dimensions of the checkerboard
WIDTH = 6
HEIGHT = 9

#   Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0).
objp = np.zeros((WIDTH * HEIGHT, 3), np.float32)
objp[:, :2] = np.mgrid[0:HEIGHT, 0:WIDTH].T.reshape(-1, 2)

#   Scale points up to scale image up (comes into play later)
objp *= 50

#   Prepare image array.
images = glob.glob('left*.jpg')

#   Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space.
imgpoints = [] # 2d points in image plane.

#   Read in Image to be overlayed onto checkerboard and resize it accordingly.
warphole2 = cv2wrap.imread("projim.jpg")
warphole = cv2wrap.resize(warphole2,((HEIGHT-1)*50,(WIDTH-1)*50))

#   Read in a blank image to remove background artifacts when image is drawn.
canvas2 = cv2wrap.imread("canvas.jpg")
canvas = cv2wrap.resize(canvas2,((HEIGHT-1)*50,(WIDTH-1)*50))

#   Define function needed for Section 2.
#   Still my own code, just copied and pasted from Assignment One.
#   It has been edited slightly to make it easier to work with in this scenario
def calibrateCamera3D(data1,data2):
    x = data1.shape[0]
    y = x*2
    A = np.zeros((y,12))
    s = 0
    for i in range(0, y):
        if (i % 2) == 0:
            A[i][0] = data1[s][0]
            A[i][1] = data1[s][1]
            A[i][2] = data1[s][2]
            A[i][3] = 1
            A[i][8] = (-1 * data2[s][0])*data1[s][0]
            A[i][9] = (-1 * data2[s][0])*data1[s][1]
            A[i][10] = (-1 * data2[s][0])*data1[s][2]
            A[i][11] = (-1 * data2[s][0])
        else:
            A[i][4] = data1[s][0]
            A[i][5] = data1[s][1]
            A[i][6] = data1[s][2]
            A[i][7] = 1
            A[i][8] = (-1 * data2[s][1]) * data1[s][0]
            A[i][9] = (-1 * data2[s][1]) * data1[s][1]
            A[i][10] = (-1 * data2[s][1]) * data1[s][2]
            A[i][11] = (-1 * data2[s][1])
            s += 1

    At = A.transpose()

    Aproduct = np.dot(At, A)

    eigenvalues, eigenvectors = np.linalg.eig(Aproduct)

    smallestvalue = sys.float_info.max
    smallestindex = 0
    for s in range(0, 9):
        if eigenvalues[s] < smallestvalue:
            smallestindex = s

    camarray = eigenvectors.transpose()[smallestindex]
    final = np.zeros((3, 4))
    count = 0
    for s in range(0, 3):
        for c in range(0, 4):
            final[s][c] = camarray[count]
            count +=1
    return final	
#   End definition


#   -------------------------------- Section_One --------------------------------
#       Take array of pictures from webcam and use them to calibrate the camera
#       Take relevant data out to undistort image from Section Two
#   -----------------------------------------------------------------------------
#   Loop through image array
for fname in images:
    #   Select current image
    img = cv2wrap.imread(fname)

    #   Sets the image to grayscale
    gray = cv2wrap.cvtColor(img,cv2wrap.COLOR_BGR2GRAY)

    #   Find the chess board corners
    ret, corners = cv2wrap.findChessboardCorners(gray, (9,6),None)

    #   If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv2wrap.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        #   Draw and display the corners
        img = cv2wrap.drawChessboardCorners(img, (9,6), corners2,ret)
        cv2wrap.imshow('Image Array',img)
        cv2wrap.waitKey(500)

#   Clear windows
cv2wrap.destroyAllWindows()

#   Obtain necessary data to undistort camera image
ret, mtx, dist, rvecs, tvecs = cv2wrap.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
h, w = img.shape[:2]
newcameramtx, roi = cv2wrap.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

#   Load image to be undistorted
img = cv2wrap.imread("Precalib.jpg")

#   Undistort the image
dst = cv2wrap.undistort(img, mtx, dist, None, newcameramtx)

#   Crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]

#   Write the calibrated image to a jpg file
cv2wrap.imwrite("calibresult.jpg", dst)

#   Redraw corners onto the undistorted image before showing it
gray = cv2wrap.cvtColor(dst,cv2wrap.COLOR_BGR2GRAY)
ret, corners = cv2wrap.findChessboardCorners(gray, (9, 6), None)
corners2 = cv2wrap.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
dst = cv2wrap.drawChessboardCorners(dst, (9, 6), corners2, ret)

#   Show the undistorted image with corners drawn
cv2wrap.imshow('Undistorted Image with Corners Drawn', dst)
cv2wrap.waitKey(5000)

#   Clear windows
cv2wrap.destroyAllWindows()


#   ------------------------------ Section_Two ------------------------------
#       Use data from previous section to undistort the camera feed and map
#       an image to the Checkerboard.
#   -------------------------------------------------------------------------
#   Initialise video capture
cap = cv2wrap.VideoCapture(0)

#   Loop to perform video capture and live image manipulation
while True:
    #   Capture a frame
    ret, img = cap.read()

    #   Set shape parameters
    h, w = img.shape[:2]

    #   Initialise camera matrix
    newcameramtx, roi = cv2wrap.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    #   Undistort the current frame
    uframe = cv2wrap.undistort(img, mtx, dist, None, newcameramtx)

    #   Our operations on the frame come here
    #   Start by setting the image to grayscale
    gray = cv2wrap.cvtColor(uframe, cv2wrap.COLOR_BGR2GRAY)

    #   Find the chess board corners
    ret, corners = cv2wrap.findChessboardCorners(gray, (HEIGHT, WIDTH), None)

    #   If found, add object points, image points (after refining them), then perform image manipulation
    if ret:
        #   Refine corners
        cv2wrap.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        #   Reshape corners array for future use
        corners = np.reshape(corners, (54, 2))

        #   Get homogenous matrix
        p = calibrateCamera3D(objp, corners)

        #   Delete empty row due to Z == 0
        p = np.delete(p, 2, 1)

        #   Warp the image stored in warphole using the parameters generated thus far
        warpimg = cv2wrap.warpPerspective(warphole, p, (w, h))

        #   Convert canvas from white to black to better remove background artifact from image before overlaying
        clearcanvas = cv2wrap.bitwise_not(cv2wrap.warpPerspective(canvas, p, (w, h)))

        #   Combine the frame image and the cleared canvas to ready it for the warped image
        uframe = cv2wrap.bitwise_and(uframe, clearcanvas)

        #   layer The warped image onto the frame image in the correct position
        uframe += warpimg

    cv2wrap.imshow('Camera', uframe)

    # If q is pressed, break the loop, otherwise keep looping
    if cv2wrap.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2wrap.destroyAllWindows()