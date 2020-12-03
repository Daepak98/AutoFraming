"""
The script provides definitions for various face detectors that are used in the `VideoCamera` class.
"""

# Basic Import Stuffs
import numpy as np
import scipy.ndimage as ndimage
from imageio import imread, imwrite
from skimage import feature
import cv2
import os
import pickle as pkl


def find_center(img: np.ndarray) -> tuple:
    """
    Given a black and white image, find the center of all the white pixels in the image.
    - `img`: A numpy array representation of a black and white image.

    Returns a tuple of the (x, y) (or (column, row)) location of the center of the white pixels.
    """
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
    """
    Finds the center of the face by using the Canny Edge detector to 
    find the rough outline of a face.
    - `img`:  A numpy array representation of a grayscale image

    Returns a tuple of the (x, y) (or (column, row)) location of the computed center of the face
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #img = img/255.0
    img = img*1.0
    cannyIm = feature.canny(img, 20)
    return find_center(cannyIm)


# ## Method 2: Extension of sobel masks
# Utilize three our designed extended sobel masks (inlcuding the original sobel mask) to detect egde, symetrical structure, diagonal structure and rotational symmetrical structures (i.e., eyes)
class Process:
    """
    The Process class is used to streamline the computation of several reused masks in this script.
    - `img`: A numpy array representation of an image

    Once the object is constructed, several convolved images are produced, based on `img`:
    - `gIm` - The result of convolving using Sobel X,Y masks
    - `gImS` - The result of convolving using Symmetric X,Y masks
    - `gImD` - The result of convolving using Diagonal masks
    - `gImC` - The result of convolving using a Rotaional Symmetry mask

    There are also several thresholds that are generated as well, each geared towards a different resultant image:
    - `T` - A threshold for `gIm`
    - `TS` - A threshold for `gImS`
    - `TD` - A threshold for `gImD`
    - `TC` - A threshold for `gImC`

    The aforementioned resultant images and thresholds are attributes of the constructed object,
    """

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
    """
    Uses Sobel X,Y masks to find the center of a face by estimating facial structure using strong edges.
    - `img`:  A numpy array representation of a grayscale image

    Returns a tuple of the (x, y) (or (column, row)) location of the computed center of the face
    """
    p = Process(img)
    temp = p.gIm > p.T
    return find_center(temp)


def circular(img):
    """
    Using a rotational symmetry mask, find the center of a face by finding the location of the eyes.
    - `img`:  A numpy array representation of a grayscale image

    Returns a tuple of the (x, y) (or (column, row)) location of the computed center of the face
    """
    # A new "circular" (rotational symmetry) mask, designed for detecting both eyes. Then, the center will be the mid-point of two eyes
    p = Process(img)
    temp = p.gImC > p.TC
    return find_center(temp)


def ensemble(img, openning=False):
    """
    Using an ensemble of Sobel masks, Symmetry masks, Diagonal masks, and Rotational Symmetry masks, 
    find face-like features and find the center of them
    - `img`:  A numpy array representation of a grayscale image
    - `openning`: A `True`/`False` value that determines whether or not to process further by using Binary Opening

    Returns a tuple of the (x, y) (or (column, row)) location of the computed center of the face
    """
    # Combine all four masks together
    # After obtaining the resultant matrix, (optionally) apply opening (dilution followed by errosion) and then take the average
    p = Process(img)
    temp = np.logical_and(p.gImC > p.TC, np.logical_and(
        p.gIm > p.T, np.logical_and(p.gImS > p.TS, p.gImD > p.TD)))
    if openning:
        temp = ndimage.binary_opening(temp)
    return find_center(temp)


def __computeFaceStats(*ims) -> tuple:
    """
    Given a set of 2D face images, generate an average face and standard deviation face.
    - `ims`: 1 or more 2D images to compute mean and standard deviation on

    Returns a tuple of numpy arrays of the form, (mean, std)
    """
    ims = np.array(ims)
    mean = np.mean(ims, axis=0)
    std = np.std(ims, axis=0)
    return mean, std


"""
Generates or retrieves the mean and standard deviation face.
"""
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


def __find_face(img) -> tuple:
    """
    Given an image and using the pre-existing mean and standard deviation models, 
    find the closest thing to a face in the image
    - `img`: A 2D image to find a face in.

    Returns the 2 (x, y) (or (column, row)) points for the bounding box around the suspected face. 
    - In the form (x1, y1, x2, y2)
    """
    mean = __mean_face
    std = __std_face
    stats = -1*np.ones(img.shape)
    stats = stats[:-mean.shape[0], :-mean.shape[1]]
    detail = 12
    for y in range(0, img.shape[0]-mean.shape[0], int(mean.shape[0]/detail)):
        for x in range(0, img.shape[1]-mean.shape[1], int(mean.shape[1]/detail)):
            sub = img[y:y+mean.shape[0], x:x+mean.shape[1]]
            stats[y, x] = (np.abs(sub - mean)/(std)).sum()
    # stats[stats == 0] = np.max(stats)
    try:
        min_y, min_x = np.where(stats == np.min(stats[stats > -1]))
        min_y = min_y[0]
        min_x = min_x[0]
    except:
        min_y = int(img.shape[0]/2-mean.shape[0]/2)
        min_x = int(img.shape[1]/2-mean.shape[1]/2)
    return min_y, min_x, min_y+mean.shape[0], min_x+mean.shape[1]


def mean_face(img):
    """
    Using the Mean and Standard Deviation face models, compute the center of what's most likely a face in the image.
    - `img`: The RGB image to find a face in.

    Returns the (x, y) (or (column, row)) location of the center of the face.
    """
    center = (0, 0)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    y, x, h, w = __find_face(img_hsv[:, :, 1])

    x_mean = int((x+w)/2)
    y_mean = int((y+h)/2)
    center = (x_mean, y_mean)

    return center


def open_cv(img):
    """
    Using OpenCV's Haar Feature Cascade Classifier, find the center of the first face found in the image
    - `img`: The 3D image to find the face in.

    Returns the (x, y) (or (column, row)) location of the center of the suspected face.
    """
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
    """
    A tester to test the different methods independently
    """
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
