import cv2


class ProcessImage():

    def _cut_face(self, image, faces):
        cutted_face = []

        for x, y, w, h in faces:
            width_remove = int(0.2 * w / 2)
            cutted_face.append(image[y:y + h,
                                     x + width_remove:x + w - width_remove])

        return cutted_face

    def _normalize_intensity(self, images):
        images_norm = []

        for image in images:

            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            images_norm.append(cv2.equalizeHist(image))

        return images_norm

    def _resize_image(self, images, size=(100, 100)):
        images_norm = []

        for image in images:
            if image.shape < size:
                image_norm = cv2.resize(image, size,
                                        interpolation=cv2.INTER_AREA)
            else:
                image_norm = cv2.resize(image, size,
                                        interpolation=cv2.INTER_CUBIC)

            images_norm.append(image_norm)

        return images_norm

    def normalize_image(self, frame, faces):
        frame = self._cut_face(frame, faces)
        frame = self._normalize_intensity(frame)
        frame = self._resize_image(frame)
        return frame
