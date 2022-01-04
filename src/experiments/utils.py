import torch
import torch.nn as nn
import torchvision.transforms as tfs
import PIL
from PIL import Image
import numpy as np
import pandas as pd
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_bounding_boxes
import os
from pathlib import Path
import cv2


def crop(img_name: str) -> PIL.Image.Image:
    """
    Crops image using its bounding box.

    Parameters:
        img_name: valid image name from images.txt file

    Returns:
        cropped image
    """

    images = pd.read_csv("data/images.txt", sep=" ", header=None)
    images.columns = ["img_id", "name"]

    bounding_boxes = pd.read_csv("data/bounding_boxes.txt", sep=" ", header=None)
    bounding_boxes.columns = ["img_id", "x", "y", "w", "h"]

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


def crop_images(dir_name: str) -> None:
    """
    Crops all images from given directory using their bounding boxes.

    Parameters:
        dir_name: directory from data/images/
    Returns:
        None
    """

    images = pd.read_csv("data/images.txt", sep=" ", header=None)
    images.columns = ["img_id", "name"]

    image_class_labels = pd.read_csv("data/image_class_labels.txt", sep=" ", header=None)
    image_class_labels.columns = ["img_id", "class_label"]

    train_test_split = pd.read_csv("data/train_test_split.txt", sep=" ", header=None)
    train_test_split.columns = ["img_id", "train"]

    dir_path = os.path.join("data/images", dir_name)
    dir = os.fsencode(dir_path)
    Path("data/train_cropped/" + dir_name).mkdir(parents=True, exist_ok=True)

    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            file_path = os.path.join(dir_name, filename).replace("\\", "/")
            img_id = images.loc[images["name"] == file_path]["img_id"].tolist()[0]
            is_train = train_test_split.loc[train_test_split["img_id"] == img_id]["train"].tolist()
            if is_train[0] == 1:
                img = crop(file_path)
                img.save("data/train_cropped/" + file_path)


def visualize_real_prototype(model: nn.Module, img_name: str, class_number: int, prototype_number: int) \
        -> PIL.Image.Image:
    """
    Visualizes "real" prototype using given training image.

    Parameters:
        model: model to use
        img_name: valid image name from data/train_cropped/ directory
        class_number: number of the image class
        prototype_number: number of the prototype to visualize (0 to 9)
    Returns:
        image with bounding box
    """
    device = next(model.parameters()).device
    img_size = 224
    transform = tfs.Compose([tfs.Resize(size=(img_size, img_size)), tfs.ToTensor()])

    input_img = Image.open("data/train_cropped/" + img_name)
    input_tensor = transform(input_img)
    input_tensor = input_tensor.unsqueeze(0)
    with torch.no_grad():
        distances = model.prototype_distances(input_tensor.to(device))
        activations = model.distance_2_similarity(distances).cpu()
    proto_per_class = model.num_prototypes // model.num_classes
    input = activations[0][class_number * proto_per_class + prototype_number].unsqueeze(0).unsqueeze(0)

    m = nn.Upsample(size=(img_size, img_size), mode='bicubic')
    output = m(input)

    q = torch.quantile(output, 0.95)
    mask = torch.where(output > q, 1, 0).squeeze(0)
    boxes = masks_to_boxes(mask)
    input_tensor *= 255
    input_tensor = input_tensor.to(torch.uint8)
    drawn_boxes = draw_bounding_boxes(input_tensor.squeeze(0), boxes, colors="yellow")
    pilimg = tfs.ToPILImage()(drawn_boxes)

    return pilimg


def heatmap(model: nn.Module, input_img: PIL.Image.Image, class_number: int, prototype_number: int, superimpose=False) \
        -> PIL.Image.Image:
    """
    Plots heatmap or heatmap over the original image.

    Parameters:
        model: model to use
        input_img: original image
        class_number: number of the image class
        prototype_number: number of the prototype to visualize (0 to 9)
        superimpose: False to plot only heatmap, True to plot heatmap over input image

    Returns:
        heatmap
    """

    device = next(model.parameters()).device
    img_size = 224
    transform = tfs.Compose([tfs.Resize(size=(img_size, img_size)), tfs.ToTensor()])

    input_tensor = transform(input_img)
    input_tensor = input_tensor.unsqueeze(0)
    with torch.no_grad():
        distances = model.prototype_distances(input_tensor.to(device))
        activations = model.distance_2_similarity(distances).cpu()
    proto_per_class = model.num_prototypes // model.num_classes
    input = activations[0][class_number * proto_per_class + prototype_number].unsqueeze(0).unsqueeze(0)

    m = nn.Upsample(size=(img_size, img_size), mode='bicubic')
    output = m(input)

    output = output.squeeze(0).squeeze(0).cpu().detach().numpy()
    rescaled_output = output - np.amin(output)
    rescaled_act_pattern = rescaled_output / np.amax(rescaled_output)
    heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_pattern), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]

    if superimpose:
        original_image = input_tensor.squeeze(0).cpu().detach().numpy()
        original_image = original_image.transpose(1, 2, 0)
        heatmap_over_original = 0.3 * heatmap + 0.7 * original_image
        pilimg = tfs.ToPILImage()(np.uint8(heatmap_over_original * 255))
    else:
        pilimg = tfs.ToPILImage()(np.uint8(heatmap * 255))

    return pilimg
