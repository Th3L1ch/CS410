'''Visualises the data file for cs410 camera calibration assignment
To run: %run LoadCalibData.py
Conor Kiernan - 13512343
'''
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import axes3d

data = np.loadtxt('data.txt')

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(data[:,0], data[:,1], data[:,2],'k.')

#fig = plt.figure()
#ax = fig.gca()
#ax.plot(data[:,3], data[:,4],'r.')

plt.show()

def calibrateCamera3D(data):
    x = data.shape[0]
    y = x*2
    A = np.zeros((y,12))
    s = 0
    for i in range(0, y):
        if (i % 2) == 0:
            A[i][0] = data[s][0]
            A[i][1] = data[s][1]
            A[i][2] = data[s][2]
            A[i][3] = 1
            A[i][8] = (-1 * data[s][3])*data[s][0]
            A[i][9] = (-1 * data[s][3])*data[s][1]
            A[i][10] = (-1 * data[s][3])*data[s][2]
            A[i][11] = (-1 * data[s][3])
        else:
            A[i][4] = data[s][0]
            A[i][5] = data[s][1]
            A[i][6] = data[s][2]
            A[i][7] = 1
            A[i][8] = (-1 * data[s][4]) * data[s][0]
            A[i][9] = (-1 * data[s][4]) * data[s][1]
            A[i][10] = (-1 * data[s][4]) * data[s][2]
            A[i][11] = (-1 * data[s][4])
            s += 1

    At = A.transpose()

    Aproduct = np.dot(At, A)

    eigenvalues, eigenvectors = np.linalg.eig(Aproduct)

    smallestvalue = sys.float_info.max
    smallestindex = 0
    for s in range(0, 12):
        if eigenvalues[s] < smallestvalue:
            smallestindex = s

    camarray = eigenvectors.transpose()[smallestindex]

    final = np.zeros((3, 4))
    count = 0
    for s in range(0, 3):
        for c in range(0, 4):
            final[s][c] = camarray[count]
            count +=1
    print final
    return final

def visualiseCameraCalibration3D(data, P):

    p = np.zeros((data.shape[0], 4))
    for s in range(0,data.shape[0]):
        p[s][0] = data[s][0]
        p[s][1] = data[s][1]
        p[s][2] = data[s][2]
        p[s][3] = 1

    p = p.transpose()

    tx = np.dot(P, p)

    xy = np.zeros((data.shape[0],2))
    for s in range(0,data.shape[0]):
        xy[s][0] = (tx[0][s]/tx[2][s])
        xy[s][1] = (tx[1][s]/tx[2][s])
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(data[:, 3], data[:, 4], 'r.', xy[:,0], xy[:,1], 'b.')
    plt.show()

    return


def evaluateCameraCalibration(data, P):
    p = np.zeros((data.shape[0], 4))
    for s in range(0,data.shape[0]):
        p[s][0] = data[s][0]
        p[s][1] = data[s][1]
        p[s][2] = data[s][2]
        p[s][3] = 1

    p = p.transpose()

    tx = np.dot(P, p)

    xy = np.zeros((data.shape[0],2))
    for s in range(0,data.shape[0]):
        xy[s][0] = (tx[0][s]/tx[2][s])
        xy[s][1] = (tx[1][s]/tx[2][s])
        #print s
        #print 'xy array:\t\tx={},\t\t\ty={}'.format(xy[s][0], xy[s][1])
        #print 'data array:\t\tx={},\t\t\ty={}'.format(data[s][3], data[s][4])
        #xs = xy[s][0] - data[s][3]
        # ys = xy[s][1] - data[s][4]
        #print '\t delx = {}\t\t dely = {}'.format(xs, ys)
        #xs2 = xs**2
        #ys2 = ys**2
        #print '\t delx2 = {}\t\t dely2 = {}'.format(xs2, ys2)
        #xs2ys2 = xs2+ys2
        #print '\t delx2dely2 = {}'.format(xs2ys2)
        #dst = np.sqrt(xs2ys2)
        #print '\t dst = {}'.format(dst)

    f = np.zeros((data.shape[0],1))
    for s in range(0,data.shape[0]):
         x2 = (xy[s][0]-data[s][3])*(xy[s][0]-data[s][3])
         y2 = (xy[s][1]-data[s][4])*(xy[s][1]-data[s][4])
         x2y2= x2+y2
         dist = np.sqrt(x2y2)
         f[s][0] = dist

    print'Mean:\n{}'.format(np.mean(f))
    print'Variance:\n{}'.format(np.var(f))
    print'Min:\n{}'.format(np.min(f))
    print'Max:\n{}'.format(np.max(f))
    return

P = calibrateCamera3D(data)
visualiseCameraCalibration3D(data, P)
evaluateCameraCalibration(data, P)
