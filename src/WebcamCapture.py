import cv2


class WebcamCapture():

    def __init__(self):
        self.camera = cv2.VideoCapture(0)

    def __del__(self):
        self.camera.release()

    def get_feed(self):
        check, frame = self.camera.read()
        return frame
