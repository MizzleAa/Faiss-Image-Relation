import numpy as np
import cv2
import os

def crop_largest_object(image_path, output_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)
    if w > h:
        y = max(0, y - (w - h) // 2)
        h = w
    else:
        x = max(0, x - (h - w) // 2)
        w = h

    cropped = image[max(0, y):min(y + h, image.shape[0]), max(0, x):min(x + w, image.shape[1])]

    cv2.imwrite(output_path, cropped)

def crop_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)

        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            output_path = os.path.join(output_folder, file_name)
            crop_largest_object(file_path, output_path)
            print(f"Processed {file_path} and saved to {output_path}")

if __name__ == '__main__':
    loop = ["knife","gun","container"]
    for l in loop:
        input_folder = f"./dataset/base/{l}"
        output_folder = f"./dataset/train/{l}"
        crop_images_in_folder(input_folder, output_folder)