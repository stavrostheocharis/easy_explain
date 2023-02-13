from PIL import Image
from io import BytesIO
import requests
from typing import List
from easy_explain.functions import *


def run_easy_explain(
    model,
    image_url: str,
    total_preds: int,
    vis_types: List[List[str]] = [["blended_heat_map", "original_image"]],
    vis_signs: List[List[str]] = [["all", "all"]],
    labels_path: str = "imagenet_class_index.json",
    target=None,
):
    labels = load_data_labels(labels_path)
    # Get image from url
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    input_img = process_image(image)

    if target == None:
        prediction = predict_classes(input_img, labels, model, total_preds)
        target = prediction[0][0]
        prediction_name = get_prediction_name_from_predictions(prediction)
    else:
        prediction_name = get_prediction_name_from_labels(labels, str(target))

    trans_attribution = create_attribution(target, model, input_img)
    number_of_sets = len(vis_types)
    for i in range(number_of_sets):
        create_explanation(
            trans_attribution,
            image,
            prediction_name,
            vis_types=vis_types[i],
            vis_signs=vis_signs[i],
        )
