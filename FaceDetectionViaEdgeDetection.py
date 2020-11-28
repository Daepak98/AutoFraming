#Basic Import Stuffs
import numpy as np
import math
from imageio import imread, imwrite
from skimage import filters, color, feature
from matplotlib import cm 
import matplotlib as mpl
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage


def find_center(img):
    height = img.shape[0]
    width = img.shape[1]    

    width_idx = np.array(list(range(1,width+1))).reshape(1,-1)
    width_idx = np.repeat(width_idx,height,axis=0)

    height_idx = np.array(list(range(1,height+1))).reshape(-1,1)
    height_idx = np.repeat(height_idx,width,axis=1)
    
    x = np.sum(img*width_idx)/np.sum(img*width_idx!=0)-1
    y = np.sum(img*height_idx)/np.sum(img*height_idx!=0)-1
    
    return x,y

mpl.rcParams['figure.dpi'] = 100

# Image loading
original = color.rgb2gray(imread('input/paruru3.jpg'))

# Show image (optionally)
plt.imshow(original, cmap='gray')
plt.title('Original (Gray)')
plt.axis('off')


# ## Method 1: canny detector
# Using canny detector to locate the contour of the face, upon which a rough face center could be computed
def canny(img):
    cannyIm = feature.canny(original,8)
    plt.figure()
    plt.axis('off')
    plt.imshow(cannyIm, cmap=cm.gray)
    return find_center(cannyIm)



# ## Method 2: Extension of sobel masks
# Utilize three our designed extended sobel masks (inlcuding the original sobel mask) to detect egde, symetrical structure, diagonal structure and rotational symmetrical structures (i.e., eyes)
class Process:
    def __init__(self, img):
        soble_x=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        soble_y=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        soble_symmetric_x = np.array([[-1,-2,-1],[2,4,2],[-1,-2,-1]])
        soble_symmetric_y = np.array([[-1,2,-1],[-2,4,-2],[-1,2,-1]])
        soble_diagnoal_1 = np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
        soble_diagnoal_2 = np.array([[1,1,0],[1,0,-1],[0,-1,-1]])
        soble_center = np.array([[-3,-1,0,-1,-3],[-1,0,3,0,-1],[0,3,8,3,0],[-1,0,3,0,-1],[-3,-1,0,-1,-3]])

        gxIm = ndimage.correlate(img, soble_x, mode='nearest')
        gyIm = ndimage.correlate(img, soble_y, mode='nearest')
        gxImS = ndimage.correlate(img, soble_symmetric_x, mode='nearest')
        gyImS = ndimage.correlate(img, soble_symmetric_y, mode='nearest')
        g1ImD = ndimage.correlate(img, soble_diagnoal_1, mode='nearest')
        g2ImD = ndimage.correlate(img, soble_diagnoal_2, mode='nearest')

        self.gIm = np.maximum(np.abs(gxIm),np.abs(gyIm))
        self.gImS = np.maximum(np.abs(gxImS),np.abs(gyImS))
        self.gImD = np.maximum(np.abs(g1ImD),np.abs(g2ImD))
        self.gImC = ndimage.correlate(img, soble_center, mode='nearest')

        n=0.01
        self.T = gIm.reshape(-1)[np.argsort(gIm.reshape(-1))][-int(gIm.shape[0]*gIm.shape[1]*n)]
        self.TS = gImS.reshape(-1)[np.argsort(gImS.reshape(-1))][-int(gImS.shape[0]*gImS.shape[1]*n)]
        self.TD = gImD.reshape(-1)[np.argsort(gImD.reshape(-1))][-int(gImD.shape[0]*gImD.shape[1]*n)]
        self.TC = gImC.reshape(-1)[np.argsort(gImC.reshape(-1))][-int(gImC.shape[0]*gImC.shape[1]*(n*0.1))]

def sobel(img):
    p = Process(img)
    temp = p.gIm>p.T
    plt.figure()
    plt.axis('off')
    plt.imshow(temp, cmap=cm.gray)
    return find_center(temp)

def circular(img):
# A new "circular" (rotational symmetry) mask, designed for detecting both eyes. Then, the center will be the mid-point of two eyes
    p = Process(img)
    temp = p.gImC>p.TC
    plt.figure()
    plt.axis('off')
    plt.imshow(temp, cmap=cm.gray)
    return find_center(temp)

def ensemble(img, openning=False):
# Combine all four masks together
# After obtaining the resultant matrix, (optionally) apply opening (dilution followed by errosion) and then take the average
    p = Process(img)
    temp=np.logical_and(p.gImC>p.TC,np.logical_and(p.gIm>p.T, np.logical_and(p.gImS>p.TS,p.gImD>p.TD)))
    if openning:
        temp=ndimage.binary_opening(temp)
    plt.figure()
    plt.axis('off')
    plt.imshow(temp, cmap=cm.gray)
    return find_center(temp)


## Execution
# Method 1
center = canny(original)
# Method 2
center = sobel(original)
center = circular(original)
center = ensemble(original)
center = ensemble(original, True)
