import numpy as np
import cv2wrap
import glob

# termination criteria
criteria = (cv2wrap.TERM_CRITERIA_EPS + cv2wrap.TERM_CRITERIA_MAX_ITER, 30, 0.001)

WIDTH = 6
HEIGHT = 9

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((WIDTH * HEIGHT, 3), np.float32)
objp[:, :2] = np.mgrid[0:HEIGHT, 0:WIDTH].T.reshape(-1, 2)

cap = cv2wrap.VideoCapture(0)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('left*.jpg')

for fname in images:
    img = cv2wrap.imread(fname)
    gray = cv2wrap.cvtColor(img,cv2wrap.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2wrap.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2wrap.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2wrap.drawChessboardCorners(img, (9,6), corners2,ret)
        #cv2wrap.imshow('img',img)
        #cv2wrap.waitKey(500)

ret, mtx, dist, rvecs, tvecs = cv2wrap.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
img = cv2wrap.imread('left06.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv2wrap.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
print ret
#print objpoints
#print imgpoints
print mtx
print dist
#print newcameramtx
#undistort
dst = cv2wrap.undistort(img, mtx, dist, None, newcameramtx)
#print dst
# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2wrap.imwrite('calibresult.png', dst)


gray = cv2wrap.cvtColor(dst,cv2wrap.COLOR_BGR2GRAY)
ret, corners = cv2wrap.findChessboardCorners(gray, (9,6),None)
corners2 = cv2wrap.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
dst = cv2wrap.drawChessboardCorners(dst, (9,6), corners2,ret)
#cv2wrap.imshow('img',dst)
#cv2wrap.waitKey(5000)

#print dst
with file("distortionCoefficients.txt", 'w') as outfile:
    for slice_2d in dst:
        np.savetxt(outfile, slice_2d,"%d")

#calculate average error
mean_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2wrap.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    #print _[0]
    error = cv2wrap.norm(imgpoints[i], imgpoints2, cv2wrap.NORM_L2) / len(imgpoints2)
    mean_error += error

print "total error: ", mean_error / len(objpoints)
cv2wrap.destroyAllWindows()