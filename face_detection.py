# Basic Import Stuffs

import numpy as np
import scipy.ndimage as ndimage
from imageio import imread, imwrite
from skimage import feature
import cv2
import os
import pickle as pkl


def find_center(img):
    height = img.shape[0]
    width = img.shape[1]

    width_idx = np.array(list(range(1, width+1))).reshape(1, -1)
    width_idx = np.repeat(width_idx, height, axis=0)

    height_idx = np.array(list(range(1, height+1))).reshape(-1, 1)
    height_idx = np.repeat(height_idx, width, axis=1)

    x = np.sum(img*width_idx)/np.sum(img*width_idx != 0)-1
    y = np.sum(img*height_idx)/np.sum(img*height_idx != 0)-1
    if np.isnan(x) or np.isnan(y):
        return x, y

    return int(x), int(y)

# ## Method 1: canny detector
# Using canny detector to locate the contour of the face, upon which a rough face center could be computed


def canny(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #img = img/255.0
    img = img*1.0
    cannyIm = feature.canny(img, 20)
    return find_center(cannyIm)


# ## Method 2: Extension of sobel masks
# Utilize three our designed extended sobel masks (inlcuding the original sobel mask) to detect egde, symetrical structure, diagonal structure and rotational symmetrical structures (i.e., eyes)
class Process:
    def __init__(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img*1.0
        soble_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        soble_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        soble_symmetric_x = np.array([[-1, -2, -1], [2, 4, 2], [-1, -2, -1]])
        soble_symmetric_y = np.array([[-1, 2, -1], [-2, 4, -2], [-1, 2, -1]])
        soble_diagnoal_1 = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
        soble_diagnoal_2 = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]])
        soble_center = np.array([[-3, -1, 0, -1, -3], [-1, 0, 3, 0, -1],
                                 [0, 3, 8, 3, 0], [-1, 0, 3, 0, -1], [-3, -1, 0, -1, -3]])

        gxIm = ndimage.correlate(img, soble_x, mode='nearest')
        gyIm = ndimage.correlate(img, soble_y, mode='nearest')
        gxImS = ndimage.correlate(img, soble_symmetric_x, mode='nearest')
        gyImS = ndimage.correlate(img, soble_symmetric_y, mode='nearest')
        g1ImD = ndimage.correlate(img, soble_diagnoal_1, mode='nearest')
        g2ImD = ndimage.correlate(img, soble_diagnoal_2, mode='nearest')

        self.gIm = np.maximum(np.abs(gxIm), np.abs(gyIm))
        self.gImS = np.maximum(np.abs(gxImS), np.abs(gyImS))
        self.gImD = np.maximum(np.abs(g1ImD), np.abs(g2ImD))
        self.gImC = ndimage.correlate(img, soble_center, mode='nearest')

        n = 0.1
        self.T = self.gIm.reshape(-1)[np.argsort(self.gIm.reshape(-1))
                                      ][-int(self.gIm.shape[0]*self.gIm.shape[1]*n)]
        self.TS = self.gImS.reshape(-1)[np.argsort(self.gImS.reshape(-1))
                                        ][-int(self.gImS.shape[0]*self.gImS.shape[1]*n)]
        self.TD = self.gImD.reshape(-1)[np.argsort(self.gImD.reshape(-1))
                                        ][-int(self.gImD.shape[0]*self.gImD.shape[1]*n)]
        self.TC = self.gImC.reshape(-1)[np.argsort(self.gImC.reshape(-1))
                                        ][-int(self.gImC.shape[0]*self.gImC.shape[1]*(n*0.1))]


def sobel(img):
    p = Process(img)
    temp = p.gIm > p.T
    return find_center(temp)


def circular(img):
    # A new "circular" (rotational symmetry) mask, designed for detecting both eyes. Then, the center will be the mid-point of two eyes
    p = Process(img)
    temp = p.gImC > p.TC
    return find_center(temp)


def ensemble(img, openning=False):
    # Combine all four masks together
    # After obtaining the resultant matrix, (optionally) apply opening (dilution followed by errosion) and then take the average
    p = Process(img)
    temp = np.logical_and(p.gImC > p.TC, np.logical_and(
        p.gIm > p.T, np.logical_and(p.gImS > p.TS, p.gImD > p.TD)))
    if openning:
        temp = ndimage.binary_opening(temp)
    return find_center(temp)


def __computeFaceStats(*ims):
    ims = np.array(ims)
    mean = np.mean(ims, axis=0)
    std = np.std(ims, axis=0)
    return mean, np.sum(std)


__mean_face, __std_face = 0, 0
if not os.path.exists("./model/stats_model.pkl"):
    __ims = []
    __imspath = "./training/"
    dirs = os.listdir(__imspath)
    num_keep = 1000
    for i, name in enumerate(dirs):
        if num_keep != 0:
            if i >= num_keep:
                break
        impath = f"{__imspath}{name}"
        im = cv2.imread(impath)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        __ims.append(im[:, :, 1])
    __mean_face, __std_face = __computeFaceStats(*__ims)
    with open("./model/stats_model.pkl", 'wb') as f:
        pkl.dump((__mean_face, __std_face), f)
else:
    with open("./model/stats_model.pkl", 'rb') as f:
        __mean_face, __std_face = pkl.load(f)


def __find_face(img):
    mean = __mean_face
    std = __std_face
    stats = -1*np.ones(img.shape)
    stats = stats[:-mean.shape[0], :-mean.shape[1]]
    detail = 12
    for y in range(0, img.shape[0]-mean.shape[0], int(mean.shape[0]/detail)):
        for x in range(0, img.shape[1]-mean.shape[1], int(mean.shape[1]/detail)):
            sub = img[y:y+mean.shape[0], x:x+mean.shape[1]]
            stats[y, x] = (abs(sub - mean)/(std)).sum()
    # stats[stats == 0] = np.max(stats)
    try:
        min_y, min_x = np.where(stats == np.min(stats[stats > 0]))
        min_y = min_y[0]
        min_x = min_x[0]
    except:
        min_y = int(img.shape[0]/2-mean.shape[0]/2)
        min_x = int(img.shape[1]/2-mean.shape[1]/2)
    return min_y, min_x, min_y+mean.shape[0], min_x+mean.shape[1]


def mean_face(img):
    center = (0, 0)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    y, x, h, w = __find_face(img_hsv[:, :, 1])

    x_mean = int((2*x+w)/2)
    y_mean = int((2*y+h)/2)
    center = (x_mean, y_mean)

    return center


def open_cv(img):
    center = (0, 0)

    face_cascade = cv2.CascadeClassifier(
        "training/haarcascade_frontalface_default.xml")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, 1.1, 4)
    for (x, y, w, h) in faces[:1]:
        x_mean = int((2*x+w)/2)
        y_mean = int((2*y+h)/2)
        center = (x_mean, y_mean)

    return center


if __name__ == "__main__":
    v = cv2.VideoCapture(0)
    _, original = v.read()
    v.release()
    # Execution
    # Method 1
    center = canny(original)
    # Method 2
    center = sobel(original)
    center = circular(original)
    center = ensemble(original)
    center = ensemble(original, True)
