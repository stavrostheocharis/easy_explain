from PIL import Image
from io import BytesIO
import requests
from src.functions import *

def run_easy_explain(model, image_url, total_preds, vis_types=[["blended_heat_map", "original_image"]], vis_signs = [["all","all"]], labels_path: str =  'imagenet_class_index.json'):
    labels = load_data_labels(labels_path)
    # Get image from url
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    input_img = process_image(image)
    prediction = predict_classes(input_img, labels, model, total_preds)
    prediction_name = get_prediction_name(prediction)
    trans_attribution = create_attribution(prediction[0][0], model, input_img)
    number_of_sets = len(vis_types)
    for i in range(number_of_sets):
        create_explanation(trans_attribution, image, prediction_name, vis_types=vis_types[i], vis_signs = vis_signs[i])