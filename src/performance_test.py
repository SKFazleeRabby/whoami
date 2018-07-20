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

images, labels, label_dic = collect_dataset()

rec_lbph = cv2.face.LBPHFaceRecognizer_create()
rec_lbph.train(images, labels)

rec_eigen = cv2.face.EigenFaceRecognizer_create()
rec_eigen.train(images, labels)

rec_fisher = cv2.face.FisherFaceRecognizer_create()
rec_fisher.train(images, labels)

collector = cv2.face.StandardCollector_create()

frame = camera.get_feed()
faces_area = face_detector.detect_face(frame)

if len(faces_area):
    faces = image_processor.normalize_image(frame, faces_area)
    face = faces[0]

    rec_eigen.predict_collect(face, collector)
    conf = collector.getMinDist()
    pred = collector.getMinLabel()
    print('Eigen Faces -> Prediction: ' + label_dic[pred].capitalize() +
          ' Confidence: ' + str(round(conf)))

    rec_fisher.predict_collect(face, collector)
    conf = collector.getMinDist()
    pred = collector.getMinLabel()
    print('Fisher Faces -> Prediction: ' + label_dic[pred].capitalize() +
          ' Confidence: ' + str(round(conf)))

    rec_lbph.predict_collect(face, collector)
    conf = collector.getMinDist()
    pred = collector.getMinLabel()
    print('LBPH Faces -> Prediction: ' + label_dic[pred].capitalize() +
          ' Confidence: ' + str(round(conf)))

    while True:
        cv2.imshow('Face', face)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
