"""
    Script para preprocesar el dataset. Porque es costoso para el entrenamiento 
    tener que estar constantemente detectando la cara y sin embargo, no ocupan tanto 
    guardados en memoria. Se aplica un vez y se olvida.

    Se genera un directorio con el mismo nombre que el original pero con un 'refactor'
    incluido. 

    Procesado:
        1. Extraccion de las caras de las imagenes
"""
import cv2 as cv2
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import face_recognition
import numpy as np
import os

def detect_face(image, padding=0):
    # extract location
    face_locations = face_recognition.face_locations(image)
    
    # crop
    for (top, right, bottom, left) in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top -= padding
        right += padding
        bottom += padding
        left -= padding

    # return cropped face
    cropped_image = image[top:bottom, left:right,:]
    
    return cropped_image

def parse_file_name(filepath):
    w = filepath.split("/")
    path =  w[1] + "/" + w[2] + "/" + w[3]
    name = w[4]
    return path, name

black_list = ['fake.007295.jpg','007292.jpg', '007324.jpg', '007759.jpg', '007786.jpg', '007792.jpg']

if __name__ == '__main__':
    padding = 0
    path = "Task_1"
    refactor_path = "refactor_check_task_1" # name of target directory

    image_loader = ImageFolder(path)

    i = 0
    for (im, tag), path in zip(image_loader, image_loader.imgs):
        print(path[0])
        path, name = parse_file_name(path[0])

        if name not in black_list:
            im = np.array(im)
            face = detect_face(im)

            if not os.path.exists(refactor_path + "/"+ path):
                    os.makedirs(refactor_path + "/"+ path)

            im_path = refactor_path + "/" + path + "/" + name
            #print(im_path)
            #save_image(im, im_path)
            face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB) 

            cv2.imwrite(im_path, face)

