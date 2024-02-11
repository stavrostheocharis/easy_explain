import requests
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms
import numpy as np
from captum.attr import visualization as viz
from captum.attr import Occlusion
import json
from typing import Union, List, Dict
from easy_explain.methods.xai_base import ExplainabilityMethod


class OcclusionExplain(ExplainabilityMethod):
    def generate_explanation(self, **kwargs):
        image_url = kwargs["image_url"]
        total_preds = kwargs.get("total_preds", 5)
        vis_types = kwargs.get("vis_types", [["blended_heat_map", "original_image"]])
        vis_signs = kwargs.get("vis_signs", [["all", "all"]])
        labels_path = kwargs.get("labels_path", "imagenet_class_index.json")
        target = kwargs.get("target")

        labels = self.load_data_labels(labels_path)
        image = self.get_image_from_url(image_url)
        input_img = self.process_image(image)

        if target is None:
            prediction = self.predict_classes(
                input_img, labels, self.model, total_preds
            )
            target = prediction[0][0]
            prediction_name = self.get_prediction_name_from_predictions(prediction)
        else:
            prediction_name = self.get_prediction_name_from_labels(labels, str(target))

        trans_attribution = self.create_attribution(target, input_img)
        self.create_explanation(
            trans_attribution, image, prediction_name, vis_types, vis_signs
        )

    def load_data_labels(
        self, input: Union[str, Dict[str, List[str]]]
    ) -> Dict[str, List[str]]:
        if isinstance(input, str):
            with open(input) as f:
                data = json.load(f)
        else:
            data = input
        return data

    def get_image_from_url(self, image_url):
        try:
            response = requests.get(image_url)
            response.raise_for_status()  # Will raise an HTTPError if unsuccessful status code
            image = Image.open(BytesIO(response.content))
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            image = None
        return image

    @staticmethod
    def process_image(image):
        center_crop = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ]
        )

        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return normalize(center_crop(image)).unsqueeze(0)

    def predict_classes(self, input_img, labels, model, total_preds: int = 5):
        out = model(input_img)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        _, indices = torch.sort(out, descending=True)
        prediction = [
            (idx.item(), labels[str(idx.item())][1], percentage[idx].item())
            for idx in indices[0][:total_preds]
        ]
        return prediction

    def create_attribution(self, target: int, input_img):
        occlusion = Occlusion(self.model)
        attribution = occlusion.attribute(
            input_img,
            strides=(3, 9, 9),
            target=target,
            sliding_window_shapes=(3, 45, 45),
            baselines=0,
        )
        trans_attribution = np.transpose(
            attribution.squeeze().cpu().detach().numpy(), (1, 2, 0)
        )
        return trans_attribution

    @staticmethod
    def get_prediction_name_from_predictions(predictions):
        name_of_prediction = predictions[0][1]
        return name_of_prediction.capitalize().replace("_", " ")

    @staticmethod
    def get_prediction_name_from_labels(labels, target):
        name_of_prediction = labels[target][1]
        return name_of_prediction.capitalize().replace("_", " ")

    def get_image_titles(
        self,
        vis_type: Union[List[str], List[List[str]]],
        vis_sign: Union[List[str], List[List[str]]],
        name_of_prediction: str,
    ) -> List[str]:
        """
        Generates titles for each visualization based on visualization types, signs, and the prediction name.
        Adjusted to handle both single and multiple sets of types and signs.
        """
        image_titles_list = []

        # Normalize input to always work with a list of lists for uniform processing
        if not isinstance(vis_type[0], list):
            vis_type = [vis_type]  # Wrap in a list to uniform the structure
            vis_sign = [vis_sign]  # Same for vis_sign

        for types, signs in zip(vis_type, vis_sign):
            for type_, sign in zip(types, signs):
                title = (
                    f"{type_.capitalize().replace('_', ' ')} for {name_of_prediction}"
                    if sign.lower() == "all"
                    else f"{sign.capitalize().replace('_', ' ')} {type_.capitalize().replace('_', ' ')} for {name_of_prediction}"
                )
                image_titles_list.append(title)

        return image_titles_list

    def create_explanation(
        self, attribution, image, name_of_prediction, vis_types, vis_signs
    ):
        # Prepare the image using the same transformations as the original function
        center_crop = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ]
        )
        processed_image = np.array(center_crop(image))

        for vis_type, vis_sign in zip(vis_types, vis_signs):
            image_titles_list = self.get_image_titles(
                vis_type, vis_sign, name_of_prediction
            )
            assert len(vis_type) == len(
                image_titles_list
            ), "Number of visualizations must match number of titles"

            # No need to flatten since we're processing one set at a time
            _ = viz.visualize_image_attr_multiple(
                attribution,
                processed_image,  # Use the processed image for visualization
                methods=vis_type,  # Directly use the current set of visualization types
                signs=vis_sign,  # Directly use the current set of signs
                titles=image_titles_list,  # Use the generated titles for the current set
                show_colorbar=True,
                use_pyplot=True,  # Ensure this is set to True to use matplotlib for plotting
            )
