import torch
from torchvision import transforms
import numpy as np
from captum.attr import visualization as viz
from captum.attr import Occlusion
import json
from typing import Union, List, Dict


def load_data_labels(input: Union[str, Dict[str, List[str]]]) -> Dict[str, List[str]]:
    """Reads labels from a file. Labels can also be given directly"""
    if isinstance(input, str):
        # Opening JSON file
        f = open(input)
        # returns JSON object as
        # a dictionary
        data = json.load(f)
    else:
        data = input
    return data


def process_image(image):
    center_crop = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    )

    normalize = transforms.Compose(
        [
            # Convert the image to a tensor with values between 0 and 1
            transforms.ToTensor(),
            # normalize to follow 0-centered imagenet pixel rgb distribution
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return normalize(center_crop(image)).unsqueeze(0)


def predict_classes(input_img, labels, model, total_preds: int = 5):
    # Find the score in terms of percentage by using torch.nn.functional.softmax function
    # which normalizes the output to range [0,1] and multiplying by 100
    out = model(input_img)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    # Find the index (tensor) corresponding to the maximum score in the out tensor.
    # Torch.max function can be used to find the information
    _, indices = torch.sort(out, descending=True)
    prediction = [
        (idx.item(), labels[str(idx.item())][1], percentage[idx].item())
        for idx in indices[0][:total_preds]
    ]

    return prediction


def create_attribution(target: int, model, input_img):
    occlusion = Occlusion(model)
    strides = (3, 9, 9)  # smaller = more fine-grained attribution but slower
    sliding_window_shapes = (
        3,
        45,
        45,
    )  # choose size enough to change object appearance
    baselines = 0  # values to occlude the image with. 0 corresponds to gray
    attribution = occlusion.attribute(
        input_img,
        strides=strides,
        target=target,
        sliding_window_shapes=sliding_window_shapes,
        baselines=baselines,
    )

    trans_attribution = np.transpose(
        attribution.squeeze().cpu().detach().numpy(), (1, 2, 0)
    )

    return trans_attribution


def get_prediction_name_from_predictions(predictions):
    name_of_prediction = predictions[0][1]
    name_of_prediction = name_of_prediction.capitalize().replace("_", " ")
    return name_of_prediction


def get_prediction_name_from_labels(labels, target):
    name_of_prediction = labels[target][1]
    name_of_prediction = name_of_prediction.capitalize().replace("_", " ")
    return name_of_prediction


def get_image_titles(vis_types, vis_signs, name_of_prediction):
    image_titles_list = []
    for i in range(len(vis_types)):
        if vis_signs[i] == "all":
            title = (
                vis_types[i].capitalize().replace("_", " ")
                + " for "
                + name_of_prediction
            )
        else:
            title = (
                vis_signs[i].capitalize().replace("_", " ")
                + " "
                + vis_types[i].capitalize().replace("_", " ")
                + " for "
                + name_of_prediction
            )
        image_titles_list.append(title)
    return image_titles_list


def create_explanation(
    attribution,
    image,
    name_of_prediction,
    vis_types=[["blended_heat_map", "original_image"]],
    vis_signs=[["all", "all"]],
):
    center_crop = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    )
    image_titles_list = get_image_titles(vis_types, vis_signs, name_of_prediction)
    _ = viz.visualize_image_attr_multiple(
        attribution,
        np.array(center_crop(image)),
        vis_types,
        vis_signs,
        image_titles_list,
        show_colorbar=True,
    )
