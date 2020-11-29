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

    plain_names = {
        OPEN_CV: "Open CV Cascade Classifier",
        CANNY: "Canny Edge Detection",
        SOBEL: "Sobel Edge Detection",
        CICRULAR: "Circular Mask Edge Detection",
        ENSEMBLE: "Ensemble Edge Detection",
        ENSEMBLE_CLOSING: "Ensemble Edge Detection w/Binary Closing",
        MEAN_FACE: "Mean Face Dectection"
    }

    def __init__(self, method=TEST_ALL, debug=False, mirror=False):
        self.deebug = debug
        self.mirrored = mirror
        self.method = method
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

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
                    marked = cv2.putText(marked, f"{self.plain_names[key]}: Cannot Compute",
                                         (0, (i+1)*text_padding), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=colors[i])
                else:
                    marked = cv2.circle(marked, center, 20, colors[i], -1)
                    marked = cv2.putText(marked, f"{self.plain_names[key]}: {np.linalg.norm(np.array(openCVCenter)-np.array(center))}",
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
                final = cv2.putText(frame, f"{self.plain_names[self.method]}: Cannot Compute",
                                    (0, text_padding), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255))
            else:
                if self.deebug:
                    frame = cv2.circle(
                        frame, center, 20, (0, 0, 0), -1)
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
    # video_stream = VideoCamera(
    #     VideoCamera.TEST_ALL, debug=True, mirror=False)
    for i in range(7):
        video_stream = VideoCamera(i, debug=True, mirror=False)
        while True:
            frame = video_stream.get_processed()
            cv2.imshow(f"Testing {VideoCamera.plain_names[i]}",
                           frame.astype('float64')/frame.max())
            if cv2.waitKey(1) == 27:
                    # cv2.destroyAllWindows()
                break  # esc to quit
        cv2.destroyAllWindows()
