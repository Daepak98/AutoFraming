# %% Imports
import cv2
import numpy as np
import os
import pickle as pkl

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
def computeMeanFace():
    ims = []
    mean = []
    orig_shape = 0
    imspath = "./training/"
    dirs = os.listdir(imspath)
    num_ims = 10
    dirs = dirs[:num_ims]
    for i in dirs:
        impath = f"{imspath}{i}"
        im = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
        orig_shape = im.shape
        im = im.flatten()
        if len(ims) == 0:
            ims = im
        else:
            ims = np.vstack((ims, im))
        # print(impath)
    mean = np.mean(ims, axis=0)
    return ims, mean, orig_shape

def computeFaceCov(ims, mean):
    mean_shifted = np.transpose(ims - mean)
    np.cov(mean_shifted)
    pass

# %%

ims, mean_face, orig_shape = computeMeanFace()
face_cov = computeFaceCov(ims, mean_face)
# %% Run a Video Stream
video_stream = cv2.VideoCapture(0)
while True:
    ret, frame = video_stream.read()
    cv2.imshow('my webcam', frame)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()
