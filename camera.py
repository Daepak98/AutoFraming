"""
This script defines the `VideoCamera` class, as well as provides a `__main__` method to test the class independently
"""

import cv2
import numpy as np
from numpy.lib.function_base import diff

from face_detection import canny, sobel, circular, ensemble, mean_face, open_cv


class VideoCamera(object):
    """
    The VideoCamera class provides an interface to the user's video camera.

    Given a method to use, the class can read frames and adjust the position of the user's face to be in the center of the frame.

    The class give a set of constants that can be used to specify the method of face centering. 
    A further description of the methods can be found in `face_detection.py`.
    - `OPEN_CV` - Uses OpenCV's face detector
    - `CANNY` - Uses Canny Edge detector
    - `SOBEL` - Uses Sobel Mask
    - `CIRCULAR` - Uses Rotational Symmetry Masks
    - `ENSEMBLE` - Uses a combination of Circular, Sobel, and Diagonal Masks
    - `ENSEMBLE_OPENING` - Same as `ENSEMBLE`, but there is an extra step of Binary Closing performed
    - `MEAN_FACE` - Uses a Mean Face detector
    - `TEST_ALL` - Uses all the aforementioned detectors, and shows stats for how well they compare to OpenCV

    The class also provides a mapper (dictionary) from the constants to human-readable names in `plain_names`

    Aside from the constructor, there are two (2) accessible methods for use:
    - get_processed() - Returns a numpy array of the processed frame (processed using one of the aforementioned methods)
    - get_encoded_frame() - Returns a PNG formatted bytestream of the processed frame
    """

    OPEN_CV = 0
    CANNY = 1
    SOBEL = 2
    CICRULAR = 3
    ENSEMBLE = 4
    ENSEMBLE_OPENING = 5
    MEAN_FACE = 6
    TEST_ALL = -1

    __methods = {
        OPEN_CV: open_cv,
        CANNY: canny,
        SOBEL: sobel,
        CICRULAR: circular,
        ENSEMBLE: ensemble,
        ENSEMBLE_OPENING: lambda i: ensemble(i,  True),
        MEAN_FACE: mean_face
    }

    plain_names = {
        TEST_ALL: "All Detectors",
        OPEN_CV: "Open CV Cascade Classifier",
        CANNY: "Canny Edge Detection",
        SOBEL: "Sobel Edge Detection",
        CICRULAR: "Circular Mask Edge Detection",
        ENSEMBLE: "Ensemble Edge Detection",
        ENSEMBLE_OPENING: "Ensemble Edge Detection w/Binary Closing",
        MEAN_FACE: "Mean Face Dectection"
    }

    def __init__(self, method=OPEN_CV, debug=False, mirror=False, file=None):
        """
        The constructor as 4 options:
        - `method` - A enum, picked from the aforementioned constants
            - Default: `TEST_ALL`
        - `debug` - A True/False value that determines whether a tracking dot is drawn on the frame
            - Default: `False` - Tracking dot is not drawn on frame
        - `mirror` - A True/False value that determines if the frame is flipped after processing
            - Default: `False` - Frame is not flipped
        - `file` - A filepath to a video file
            - Default: None
        """
        self.deebug = debug
        self.mirrored = mirror
        self.method = method
        if file:
            self.file = file
        self.video = cv2.VideoCapture(
            0) if not file else cv2.VideoCapture(file)

    def __del__(self):
        """Releases access to the camera upon object deletion"""
        self.video.release()

    def __transform_image(self, img: np.ndarray, move_this: tuple, padding: int = 0) -> np.ndarray:
        """
        Takes an image, an x, y (or (column, row)) point, and a padding and crops the image so that `move_this`
        is the center of the new image.
        - `img`: The image to transform.
        - `move_this`: A tuple in (x, y) or (column, row) format. All values must be `int`s
        - `padding`: An integer value detailing how much padding around the initial cropping should be kept
        """
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

    def get_processed(self) -> np.ndarray:
        """
        Gets the next frame in the video stream (either file or camera) and processes it.

        Returns a numpy array representation of the image
        """
        ret, frame = self.video.read()
        if not ret:
            return None
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
                center = method(frame)

                if key == self.OPEN_CV:
                    openCVCenter = center
                if True in np.isnan(center):
                    marked = cv2.putText(marked, f"{self.plain_names[key]}: Cannot Compute",
                                         (0, (i+1)*text_padding), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=colors[i])
                else:
                    l2_norm = np.linalg.norm(
                        np.array(openCVCenter)-np.array(center))
                    marked = cv2.circle(marked, center, 20, colors[i], -1)
                    marked = cv2.putText(marked, f"{self.plain_names[key]}: {l2_norm}",
                                         (0, (i+1)*text_padding), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=colors[i])
            final = marked
        else:
            method = self.__methods[self.method]
            center = method(frame)

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
        """
        Takes the next frame in the video feed and processes it.

        Returns a PNG byte representation of the image.
        """
        final = self.get_processed()
        _, png = cv2.imencode('.png', final)

        return png.tobytes()


def process_livestream(method):
    """
    Given a method, run and display the processed version of computer's webcam feed.

    See constants documentation in `VideoCamera` class for valid values of `method`    
    """
    video_stream = VideoCamera(
        method, debug=True, mirror=False)
    while True:
        frame = video_stream.get_processed()
        cv2.imshow(f"Testing {VideoCamera.plain_names[video_stream.method]}",
                   frame.astype('float64')/frame.max())
        if cv2.waitKey(1) == 27:
            # cv2.destroyAllWindows()
            break  # esc to quit
    cv2.destroyAllWindows()


def process_to_file(video_path):
    """
    Given a path to a video file, process each frame of the video and 
    save it in the output folder of the current directory.

    Requires `video_path` to be a valid video.
    """
    i = VideoCamera.TEST_ALL
    video_stream = VideoCamera(
        i, debug=True, mirror=False,
        file=video_path)
    width = int(video_stream.video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_stream.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(video_stream.video.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(
        f"./output/Test_{VideoCamera.plain_names[i]}.avi",
        fourcc, 20.0, (width, height))
    for j in range(length):
        frame = video_stream.get_processed()
        if type(frame) == type(None):
            break
        video_writer.write(frame)
    cv2.destroyAllWindows()
    video_writer.release()
    print(f"Saved: Test_{VideoCamera.plain_names[i]}.avi")


# This method is used to test the `VideoCamera` class
# independently from the Flask setup in video_feed.py
if __name__ == "__main__":
    # process_livestream(VideoCamera.TEST_ALL) # Uncomment this line to test the VideoCamera class on a live webcam
    # process_to_file("./input/vid_test.avi") # Uncomment this line to test the VideoCamera class on a video file
    pass
