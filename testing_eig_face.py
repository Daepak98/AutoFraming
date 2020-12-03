# %% Imports
import cv2
import numpy as np
import os
import pickle as pkl
import matplotlib.pyplot as plt
import random

from numpy.core.fromnumeric import mean
# %% Moving photos from nested folder to `training`
# import os
# from os import path
# import shutil
# source = "./training/"
# dest = "./training/"

# for fpath in os.listdir(source):
#     imdir = f"{source}{fpath}/"
#     if path.isdir(imdir):
#         for impath in os.listdir(imdir):
#             # print(f"{imdir}{impath}", path.isfile(f"{imdir}{impath}"))
#             shutil.move(f"{imdir}{impath}", f"{dest}{impath}")
#     if len(os.listdir(imdir)) == 0:
#         os.rmdir(imdir)

# %% Train a model


def computeFaceStats(*ims):
    ims = np.array(ims)
    mean = np.mean(ims, axis=0)
    std = np.std(ims, axis=0)
    return mean, np.sum(std)


def computeFaceCov(ims, mean):
    mean_shifted = np.transpose(ims - mean)
    cov = np.transpose(mean_shifted)@mean_shifted
    return cov


# def getTopMEigVectors(cov):
#     u, v = np.linalg.eig(cov)
#     retain_num = 0
#     max_indices = np.argsort(u)[::-1]
#     cutoff = 0.9
#     total = np.sum(u)
#     while retain_num < len(max_indices):
#         subset = u[max_indices][:retain_num]
#         if np.sum(subset)/total >= cutoff:
#             return subset, v[:, max_indices[:retain_num]]
#         else:
#             retain_num += 1
# %%
# mean_face = computeFaceStats()
# # face_cov = computeFaceCov(ims, mean_face)
# # topVectors = getTopMEigVectors(face_cov)

# %% Run a Video Stream


# def testMean(img):
#     cy, cx = tuple(i/2 for i in img.shape)
#     p1 = tuple(int(i) for i in (cx-125, cy-125))
#     p2 = tuple(int(i) for i in (cx+125, cy+125))
#     flat = img[p1[1]:p2[1], p1[0]:p2[0]].flatten()
#     # img_cov = computeFaceCov(np.array(flat), mean_face)
#     img[:250, :250] = mean_face.reshape(orig_shape) # cv2.resize(mean_face, tuple(int(i*1) for i in mean_face.reshape(orig_shape).shape[::-1]))
#     frame = cv2.rectangle(img, p1, p2, (255, 0, 0), 2)
#     diff = np.sum(np.abs(flat - mean_face))
#     frame = cv2.putText(frame,
#                         f"{diff}",
#                         (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
#     return frame, diff


# video_stream = cv2.VideoCapture(0)
# diffs = []
# while True and len(diffs) < 600:
#     ret, frame = video_stream.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame, diff = testMean(frame)
#     diffs.append(int(diff))
#     cv2.imshow(f"{frame.shape}", frame)
#     if cv2.waitKey(1) == 27:
#         break  # esc to quit
# cv2.destroyAllWindows()
# diffs = np.array(diffs)

ims = []
mean_face = []
imspath = "./training/"
dirs = os.listdir(imspath)
# random.shuffle(dirs)
num_keep = 1000
for i, name in enumerate(dirs):
    if num_keep != 0:
        if i >= num_keep:
            break
    impath = f"{imspath}{name}"
    im = cv2.imread(impath)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    ims.append(im[:, :, 1])
mean_face, std_face = computeFaceStats(*ims)

# %%
plt.imshow(mean_face, cmap=plt.cm.gray)

# %%


def test(img, mean, std):
    stats = -1*np.ones(img.shape)
    stats = stats[:-mean.shape[0], :-mean.shape[1]]
    for y in range(0, img.shape[0]-mean.shape[0], int(mean.shape[0]/12)):
        for x in range(0, img.shape[1]-mean.shape[1], int(mean.shape[1]/12)):
            sub = img[y:y+mean.shape[0], x:x+mean.shape[1]]
            stats[y, x] = (np.abs(sub - mean)/std).sum()
    # stats[stats == 0] = np.max(stats)
# try:
    min_y, min_x = np.where(stats == np.min(stats[stats > -1]))
    min_y = min_y[0]
    min_x = min_x[0]
# except:
#     min_y = int(img.shape[0]/2-mean.shape[0]/2)
#     min_x = int(img.shape[1]/2-mean.shape[1]/2)
    return min_y, min_x, min_y+mean.shape[0], min_x+mean.shape[1]


video_stream = cv2.VideoCapture(0)
while True:
    ret, frame = video_stream.read()
    frame = cv2.flip(frame, 1)
    process_this = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    min_y, min_x, min_h, min_w = test(
        process_this[:, :, 1], mean_face, std_face)
    center = ((min_x+min_w)/2, (min_y+min_h)/2)
    center = tuple(int(i) for i in center)
    frame[min_y:min_h, min_x:min_w, 1] = mean_face
    frame = cv2.rectangle(frame, (min_x, min_y),
                          (min_w, min_h), (0, 0, 0), 2)
    frame = cv2.circle(frame, center, radius=20, color=(255,0,0), thickness=-1)
    
    # cv2.imshow("HSV", process_this[:,:,1])
    cv2.imshow("Testing", frame)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()
video_stream.release()
