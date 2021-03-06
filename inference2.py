from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import argparse
import csv

dic = {1: 8, 2: 5, 3: 4, 5: 9, 6: 1, 8: 7, 9: 6, 10: 3, 12: 2, 13: 0}
operators = [0, 4, 7, 11]


def calculate(array):
    cnt = 0

    numbers = []
    to_do = 0

    for i in array:
        if i in operators:
            cnt += 1
            to_do = i
        else:
            numbers.append(dic[i])

    if (cnt != 1): return 0
    try:
        if (to_do == 0): return numbers[0] / numbers[1]
        if (to_do == 4): return numbers[0] - numbers[1]
        if (to_do == 7): return numbers[0] + numbers[1]
        if (to_do == 11): return numbers[0] * numbers[1]
    except:
        return 0


parser = argparse.ArgumentParser()
parser.add_argument("folder", type=str)
args = parser.parse_args()

folder_path = args.folder  # image folder

# folder_path = "test/"
model_path = 'models/recognizer/'  # model folder

img_width, img_height = 384, 128
M, N = 128, 128

model = load_model(model_path)  # load the trained model

images = []
files = os.listdir(folder_path)
for img in files:
    img = os.path.join(folder_path, img)
    img = image.load_img(img, target_size=(img_height, img_width), color_mode='grayscale')
    img = image.img_to_array(img)
    img /= 255.
    tiles = [img[x:x + M, y:y + N] for x in range(0, img.shape[0], M) for y in range(0, img.shape[1], N)]

    img = np.expand_dims(tiles[0], axis=0)
    images.append(img)
    img = np.expand_dims(tiles[1], axis=0)
    images.append(img)
    img = np.expand_dims(tiles[2], axis=0)
    images.append(img)

images = np.vstack(images)
classes = np.argmax(model.predict(images, batch_size=16), axis=-1)
k = list(classes)
answers = [["Image_Name", "Value"]]
for i in range(len(files)):
    p = k[3 * i: 3 * i + 3]
    ans = 0
    try:
        ans = calculate(p)
    except:
        ans = 0
    answers.append([files[i], int(ans)])

with open("team_name_2.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(answers)
