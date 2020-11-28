from math import ceil

import cv2
import numpy as np
from numpy.lib.function_base import diff
import random
import os

from face_detection import canny, sobel, circular, ensemble, mean_face, open_cv


class VideoCamera(object):
    OPEN_CV = 0
    CANNY = 1
    SOBEL = 2
    CICRULAR = 3
    ENSEMBLE = 4
    ENSEMBLE_CLOSING = 5
    MEAN_FACE = 6
    TEST_ALL = -1

    __methods = {
        OPEN_CV: open_cv,
        CANNY: canny,
        SOBEL: sobel,
        CICRULAR: circular,
        ENSEMBLE: ensemble,
        ENSEMBLE_CLOSING: ensemble,
        MEAN_FACE: mean_face
    }

    __plain_names = {
        OPEN_CV: "Open CV Cascade Classifier",
        CANNY: "Canny Edge Detection",
        SOBEL: "Sobel Edge Detection",
        CICRULAR: "Circular Mask Edge Detection",
        ENSEMBLE: "Ensemble Edge Detection",
        ENSEMBLE_CLOSING: "Ensemble Edge Detection w/Binary Closing",
        MEAN_FACE: "Mean Face Dectection"
    }

    def __init__(self, method=TEST_ALL, debug=False, mirror=False):
        ims = self.__get_training()
        self.__mean_face, self.__std_face = self.__computeFaceStats(*ims)
        self.deebug = debug
        self.mirrored = mirror
        self.method = method
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
        mean = self.__mean_face
        std = self.__std_face
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
        temp_view = img[:, :bounds[1]-d] if d >= 0 else img[:, -d:]
        d = diff_vector[1]
        temp_view = temp_view[:bounds[0]-d+2*padding
                              ] if d >= 0 else temp_view[-d+2*padding:]

        transform = temp_view.copy()
        diff_vector = tuple(abs(i) for i in diff_vector)
        pad_h = int(diff_vector[1]/2)
        pad_w = int(diff_vector[0]/2)
        transform = np.vstack(
            (np.zeros((pad_h, transform.shape[1], 3)), transform))
        transform = np.vstack(
            (transform, np.zeros((pad_h, transform.shape[1], 3))))
        transform = np.hstack(
            (np.zeros((transform.shape[0], pad_w, 3)), transform))
        transform = np.hstack(
            (transform, np.zeros((transform.shape[0], pad_w, 3))))
        return transform

    def get_processed(self):
        _, frame = self.video.read()
        final = 0

        text_padding = 30
        if self.method == self.TEST_ALL:
            openCVCenter = 0
            colors = [(0, 0, 0),      # Black
                      (255, 0, 0),    # Blue
                      (0, 255, 0),    # Green
                      (0, 0, 255),    # Red
                      (255, 0, 255),  # Magenta
                      (255, 255, 0),  # Cyan
                      (0, 255, 255)]   # Yellow
            marked = frame.copy()
            for i, (key, method) in enumerate(self.__methods.items()):
                center = method(
                    frame) if key != self.ENSEMBLE_CLOSING else self.__methods[self.ENSEMBLE_CLOSING](frame, True)
                if key == self.OPEN_CV:
                    openCVCenter = center
                if True in np.isnan(center):
                    marked = cv2.putText(marked, f"{self.__plain_names[key]}: Cannot Compute",
                                         (0, (i+1)*text_padding), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=colors[i])
                else:
                    marked = cv2.circle(marked, center, 20, colors[i], -1)
                    marked = cv2.putText(marked, f"{self.__plain_names[key]}: {np.linalg.norm(np.array(openCVCenter)-np.array(center))}",
                                         (0, (i+1)*text_padding), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=colors[i])
            final = marked
        else:
            if self.method != self.ENSEMBLE_CLOSING:
                method = self.__methods[self.method]
                center = method(frame)
            else:
                method = self.__methods[self.ENSEMBLE_CLOSING]
                center = method(frame, True)
            if True in np.isnan(center):
                final = cv2.putText(frame, f"{self.__plain_names[self.method]}: Cannot Compute",
                                    (0, text_padding), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255))
            else:
                transformed = self.__transform_image(frame, center, 20)
                final = transformed

        final = cv2.flip(final, 1) if self.mirrored else final
        return final

    def get_encoded_frame(self):
        final = self.get_processed()
        _, png = cv2.imencode('.png', final)

        return png.tobytes()


# This method is used to test the `VideoCamera` class
# independently
if __name__ == "__main__":
    video_stream = VideoCamera(
        VideoCamera.TEST_ALL, debug=True, mirror=False)
    while True:
        frame = video_stream.get_processed()
        cv2.imshow("Testing Images", frame.astype('float64')/frame.max())
        if cv2.waitKey(1) == 27:
            # cv2.destroyAllWindows()
            break  # esc to quit
    cv2.destroyAllWindows()
