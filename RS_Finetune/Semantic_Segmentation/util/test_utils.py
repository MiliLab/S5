
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import os
from PIL import Image

DATASET_MAPPINGS = {
    "vaihingen": OrderedDict({
        'Impervious_surface': (255, 255, 255),
        'Building': (0, 0, 255),
        'Low_vegetation': (0, 255, 255),
        'Tree': (0, 255, 0),
        'Car': (255, 255, 0),
        'Others': (255, 0, 0),
    }),

    "potsdam": OrderedDict({
        'Impervious_surface': (255, 255, 255),
        'Building': (0, 0, 255),
        'Low_vegetation': (0, 255, 255),
        'Tree': (0, 255, 0),
        'Car': (255, 255, 0),
        'Others': (255, 0, 0),
    }),

    "openearthmap": OrderedDict({
        "unknown": [0, 0, 0],
        "Bareland": [128, 0, 0],
        "Grass": [0, 255, 36],
        "Pavement": [48, 148, 148],
        "Road": [255, 255, 255],
        "Tree": [34, 97, 38],
        "Water": [0, 69, 255],
        "Cropland": [75, 181, 73],
        "buildings": [222, 31, 7],
    }),

    "loveda": OrderedDict({
        'background': (255, 255, 255),
        'building': (255, 0, 0),
        'road': (255, 255, 0),
        'water': (0, 0, 255),
        'barren': (159, 129, 183),
        'forest': (0, 255, 0),
        'agriculture': (255, 195, 128),
    }),
}

def get_class_to_rgb(dataset):
    mapping = DATASET_MAPPINGS.get(dataset.lower())
    if mapping is None:
        raise ValueError(f"Dataset {dataset} not found in DATASET_MAPPINGS.")
    return {idx: value for idx, value in enumerate(mapping.values())}

def class_to_rgb_map(image, dataset):
    class_to_rgb = get_class_to_rgb(dataset)
    h, w = image.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, rgb in class_to_rgb.items():
        mask = (image == cls)
        rgb_image[mask] = rgb
    return rgb_image


def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from saved weights in multi-GPU training"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_key = k[len('module.'):]
        else:
            new_key = k
        new_state_dict[new_key] = v

    return new_state_dict

def net_process(model, image, cfg, flip=True):
    inputs = [image]

    if flip:
        inputs.append(image.flip(3))

    input = torch.cat(inputs, dim=0)
    with torch.no_grad():
        output = model(input, cfg['dataset'])

    output = F.softmax(output, dim=1)
    preds = []

    for i, inp in enumerate(inputs):
        out = output[i]
        if i == 1 and flip:
            out = torch.flip(out, dims=[2])
        preds.append(out)

    output = torch.stack(preds, dim=0).mean(0)
    output = output.unsqueeze(0)
    return output


def save_prediction(pred, id, dataset):

    pred_rgb_output_dir = os.path.join("outputs", dataset, "pred")
    pred_mask_output_dir = os.path.join("outputs", dataset, "mask")

    os.makedirs(pred_rgb_output_dir, exist_ok=True)
    os.makedirs(pred_mask_output_dir, exist_ok=True)

    image_name = os.path.splitext(os.path.basename(id[0]))[0]

    pred_np = pred.squeeze().cpu().numpy().astype(np.uint8)
    pred_mask_pil = Image.fromarray(pred_np)
    pred_mask_pil.save(os.path.join(pred_mask_output_dir, f"{image_name}.png"))

    pred_rgb = class_to_rgb_map(pred_np, dataset)
    pred_rgb_pil = Image.fromarray(pred_rgb.astype(np.uint8))
    pred_rgb_pil.save(os.path.join(pred_rgb_output_dir, f"{image_name}.png"))