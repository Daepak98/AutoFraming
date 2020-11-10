from math import ceil

import cv2

deebug = True
mirrored = True

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.training_frames = [] # These vars will be for when 
        self.trained = False      # we have to create our own
        self.num_train = 60       # classifier/detector

    def __del__(self):
        self.video.release()   

    def __find_face_center(self, img):
        center = (0,0)

        face_cascade = cv2.CascadeClassifier("training/haarcascade_frontalface_default.xml")

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_gray, 1.1, 4)
        for (x,y,w,h) in faces[:1]:
            x_mean = int((2*x+w)/2)
            y_mean = int((2*y+h)/2)
            center = (x_mean, y_mean)

        return center

    def __transform_image(self, img, move_here):
        return img.copy()

    def __crop(self, img):
        return img.copy()

    def get_processed(self):
        ret, frame = self.video.read()

        # Performance Phase        
        center = self.__find_face_center(frame)
        if deebug:
            frame = cv2.circle(frame, center, 20, (255,0,0), -1)
        transformed = self.__transform_image(frame, center)
        cropped = self.__crop(transformed)

        final = cv2.flip(transformed, 1) if mirrored else frame
        return final

    def get_frame(self):
        final = self.get_processed()
        ret, png = cv2.imencode('.png', final)

        return png.tobytes()

# This method is used to test the `VideoCamera` class
# independently 
if __name__ == "__main__": 
    video_stream = VideoCamera()
    while True:
        frame = video_stream.get_processed()
        cv2.imshow('my webcam', frame)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()