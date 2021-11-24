import torch
import torch.nn as nn
import torchvision.transforms as tfs
from PIL import Image
import numpy as np
import pandas as pd
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_bounding_boxes
import os
from pathlib import Path

img_size = 224
transform = tfs.Compose([
    tfs.Resize(size=(img_size, img_size)),
    tfs.ToTensor()])

images = pd.read_csv("data/images.txt", sep=" ", header=None)
images.columns = ["id", "name"]

image_class_labels = pd.read_csv("data/image_class_labels.txt", sep=" ", header=None)
image_class_labels.columns = ["id", "class_label"]

bounding_boxes = pd.read_csv("data/bounding_boxes.txt", sep=" ", header=None)
bounding_boxes.columns = ["id", "x", "y", "w", "h"]

train_test_split = pd.read_csv("data/train_test_split.txt", sep=" ", header=None)
train_test_split.columns = ["id", "train"]

train_images = (train_test_split.loc[train_test_split["train"] == 1])["id"]


def crop(img_name):
    index = np.where(images["name"] == img_name)

    bounding_box = bounding_boxes.loc[index]
    x = bounding_box["x"]
    y = bounding_box["y"]
    w = bounding_box["w"]
    h = bounding_box["h"]
    bbox = (x, y, x + w, y + h)

    input_img = Image.open("data/images/" + img_name)
    input_img = input_img.crop(bbox)
    return input_img


def crop_images(dir_name):
    dir_path = os.path.join("data/images", dir_name)
    directory = os.fsencode(dir_path)
    Path("data/train_cropped/" + dir_name).mkdir(parents=True, exist_ok=True)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            file_path = dir_name + "/" + filename
            id = images.loc[images["name"] == file_path]["id"]
            id = id.tolist()[0]
            is_train = train_test_split.loc[train_test_split["id"] == id]["train"].tolist()
            if is_train[0] == 1:
                img = crop(file_path)
                img.save("data/train_cropped/" + file_path)
        else:
            continue


def visualize_real_prototype(model, img_name, class_number, prototype_number):
    input_img = Image.open("data/train_cropped/" + img_name)

    input_tensor = transform(input_img)
    input_tensor = input_tensor.unsqueeze(0)
    distances = model.prototype_distances(input_tensor)
    activations = model.distance_2_similarity(distances)

    input = activations[0][class_number * 10 + prototype_number].unsqueeze(0).unsqueeze(0)

    m = nn.Upsample(size=(img_size, img_size), mode='nearest')
    output = m(input)

    q = torch.quantile(output, 0.95)
    print(q)

    mask = torch.where(output > q, 1, 0).squeeze(0)
    boxes = masks_to_boxes(mask)
    input_tensor *= 255
    input_tensor = input_tensor.to(torch.uint8)

    drawn_boxes = draw_bounding_boxes(input_tensor.squeeze(0), boxes, colors="yellow")

    pilimg = tfs.ToPILImage()(drawn_boxes)
    return pilimg
