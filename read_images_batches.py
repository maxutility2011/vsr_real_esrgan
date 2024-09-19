import cv2
import os
import sys

def read_images_from_folder(folder_path):
    images = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        img = cv2.imread(file_path)

        if img is not None:
            images.append(img)
        else:
            logging.error("Failed to load image: %s", file_path)
    
    return images

for dirpath, dirnames, filenames in os.walk(sys.argv[1], onerror=lambda e: print(e)):
    if dirpath == sys.argv[1]:
        continue

    input_images = read_images_from_folder(dirpath)