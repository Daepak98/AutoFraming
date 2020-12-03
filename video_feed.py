"""
This script serves as an entry point for the application. It serves a video feed using the VideoCamera class to an HTML page.
The application is based on Flask.

To run, use `python <path_to_video_feed.py>` and open the browser at `http://127.0.0.1:5000`
"""

from flask import Flask, render_template, Response
from camera import VideoCamera

app = Flask(__name__)

video_stream = VideoCamera(
    method=VideoCamera.MEAN_FACE, mirror=False, debug=False)


@app.route('/')
def index():
    """Returns an HTTPS reponse with the index.html page"""
    return render_template('index.html')


def gen(camera):
    """Returns a content frame with the computer's attached video camera feed"""
    while True:
        frame = camera.get_encoded_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    """Returns an HTTPS reponse for /video_feed"""
    return Response(gen(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True, port="5000")
