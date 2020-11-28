from math import ceil

import cv2
import numpy as np
from numpy.lib.function_base import diff
import random
import os

class VideoCamera(object):
    def __init__(self, debug=False, mirror=False):
        ims = self.__get_training()
        self.mean_face, self.std_face = self.__computeFaceStats(*ims)
        self.deebug = debug
        self.mirrored = mirror
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def __get_training(self):
        ims = []
        imspath = "./training/"
        dirs = os.listdir(imspath)
        random.shuffle(dirs)
        num_keep = 1000
        for i, name in enumerate(dirs):
            if num_keep != 0:
                if i >= num_keep:
                    break
            impath = f"{imspath}{name}"
            im = cv2.imread(impath)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
            ims.append(im[:, :, 1])
        return ims

    def __computeFaceStats(self, *ims):
        ims = np.array(ims)
        mean = np.mean(ims, axis=0)
        std = np.std(ims, axis=0)
        return mean, np.sum(std)

    def __find_face(self, img):
        mean = self.mean_face
        std = self.std_face
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

    def __find_face_center(self, img):
        center = (0, 0)

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        y, x, h, w = self.__find_face(img_gray[:, :, 1])

        x_mean = int((2*x+w)/2)
        y_mean = int((2*y+h)/2)
        center = (x_mean, y_mean)

        return center

    def __transform_image(self, img, move_this, padding=0):
        bounds = img.shape[:-1]
        img_center = tuple(int(i/2) for i in bounds[::-1])
        diff_vector = tuple(img_center[i] - move_this[i]
                            for i in range(len(img_center)))

        d = diff_vector[0]
        temp_view = img[:, :bounds[1]-d
                        ] if d >= 0 else img[:, -d:]
        d = diff_vector[1]
        temp_view = temp_view[:bounds[0]-d
                              ] if d >= 0 else temp_view[-d:]

        transform = temp_view.copy()
        # diff_vector = tuple(abs(i) for i in diff_vector)
        # pad_h = int(diff_vector[1]/2)+padding
        # pad_w = int(diff_vector[0]/2)+padding
        # transform = np.vstack(
        #     (np.zeros((pad_h, transform.shape[1], 3)), transform))
        # transform = np.vstack(
        #     (transform, np.zeros((pad_h, transform.shape[1], 3))))
        # transform = np.hstack(
        #     (np.zeros((transform.shape[0], pad_w, 3)), transform))
        # transform = np.hstack(
        #     (transform, np.zeros((transform.shape[0], pad_w, 3))))
        return transform

    def get_processed(self):
        _, frame = self.video.read()

        # Performance Phase
        center = self.__find_face_center(frame)
        if self.deebug:
            frame = cv2.circle(frame, center, 20, (255, 0, 0), -1)
        transformed = self.__transform_image(frame, center, 20)

        final = cv2.flip(transformed, 1) if self.mirrored else transformed
        return final

    def get_frame(self):
        final = self.get_processed()
        _, png = cv2.imencode('.png', final)

        return png.tobytes()


# This method is used to test the `VideoCamera` class
# independently
if __name__ == "__main__":
    video_stream = VideoCamera(debug=True, mirror=True)
    while True:
        frame = video_stream.get_processed()
        cv2.imshow("Testing Images", frame.astype('float64')/frame.max())
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()
