import os
import cv2
import numpy as np


def collect_dataset():
    images = []
    labels = []
    label_dic = {}

    BASE_PATH = os.path.dirname(__file__) + '/'
    persons = [person for person in os.listdir(BASE_PATH + 'person/')]

    for i, person in enumerate(persons):
        label_dic[i] = person
        for image in sorted(os.listdir(BASE_PATH + 'person/' + person)):
            images.append(cv2.imread(BASE_PATH + 'person/' + person + '/' +
                                     image, 0))
            labels.append(i)

    return (images, np.array(labels), label_dic)
