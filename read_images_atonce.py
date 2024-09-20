import cv2
import os
import sys

def read_images_from_folder(folder_path):
    images = []

    for filename in os.listdir(folder_path):
        print(filename)
        file_path = os.path.join(folder_path, filename)
        img = cv2.imread(file_path)

        if img is not None:
            images.append(img)
        else:
            logging.error("Failed to load image: %s", file_path)
    
    return images

read_images_from_folder(sys.argv[1])