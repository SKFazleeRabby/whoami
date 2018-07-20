import cv2
from WebcamCapture import WebcamCapture
from FaceDetection import FaceDetection
from ProcessImage import ProcessImage
from utils import collect_dataset


images, labels, label_dic = collect_dataset()
rec_lbph = cv2.face.LBPHFaceRecognizer_create()
rec_lbph.train(images, labels)


camera = WebcamCapture()
face_detector = FaceDetection()
image_processor = ProcessImage()

collector = cv2.face.StandardCollector_create()

while True:
    frame = camera.get_feed()
    faces_area = face_detector.detect_face(frame)

    if len(faces_area):
        faces = image_processor.normalize_image(frame, faces_area)
        face = faces[0]
        rec_lbph.predict_collect(face, collector)
        conf = collector.getMinDist()
        pred = collector.getMinLabel()
        threshold = 140
        print('LBPH Faces -> Prediction: ' + label_dic[pred].capitalize() +
              ' Confidence: ' + str(round(conf)))

        for x, y, width, height in faces_area:
            frame = cv2.rectangle(frame, (x, y), (x + width, y + height),
                                  (0, 0, 255), 3)

            if conf < threshold:
                frame = cv2.putText(frame, label_dic[pred].capitalize(),
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255),
                                    1, cv2.LINE_AA)
            else:
                frame = cv2.putText(frame, "Unknown",
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255),
                                    1, cv2.LINE_AA)

    cv2.imshow('Face', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

del camera
del face_detector
del image_processor
