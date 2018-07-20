import os
import cv2
from ProcessImage import ProcessImage


class StoreImage():
    BASE_PATH = os.path.dirname(__file__) + '/person/'

    def __init__(self):
        self.image_processor = ProcessImage()

    def user_exists(self, name):
        if os.path.exists(self.BASE_PATH + name):
            return True

        os.mkdir(self.BASE_PATH + name)
        return False

    def save_image(self, frame, faces, name, counter):
        path = os.path.join(self.BASE_PATH + name + '/')

        frame = self.image_processor.normalize_image(frame, faces)
        cv2.imwrite(path + str(counter) + '.jpg', frame[0])
        return
