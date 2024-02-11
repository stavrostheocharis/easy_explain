import torch
from torchcam.methods import SmoothGradCAMpp, LayerCAM
from torchcam.utils import overlay_mask
from torchvision import transforms
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any
import logging
from easy_explain.methods.xai_base import ExplainabilityMethod


class CAMExplain(ExplainabilityMethod):
    def __init__(self, model: torch.nn.Module):
        self.model = model
        logging.basicConfig(level=logging.INFO)

    def transform_image(
        self,
        img: torch.Tensor,
        trans_params: Dict[str, Dict[str, Any]],
    ) -> torch.Tensor:
        """
        Transforms an image using specified resizing and normalization parameters.

        Args:
            img (Image.Image): The image to transform.
            trans_params (Dict[str, Dict[str, Any]]): Parameters for resizing and normalization.

        Returns:
            torch.Tensor: The transformed image tensor.
        """
        try:
            resize_params = trans_params["Resize"]
            normalize_params = trans_params["Normalize"]
            input_tensor = transforms.functional.normalize(
                transforms.functional.resize(
                    img, (resize_params["h"], resize_params["w"])
                )
                / 255.0,
                normalize_params["mean"],
                normalize_params["std"],
            )
            return input_tensor

        except Exception as e:
            logging.error(f"Error transforming image: {e}")
            raise

    def get_multiple_layers_result(
        self,
        img: torch.Tensor,
        input_tensor: torch.Tensor,
        layers: List[str],
        alpha: float,
    ):
        """
        Visualizes CAMs for multiple layers and their fused result.

        Args:
            img (torch.Tensor): The original image tensor.
            input_tensor (torch.Tensor): The tensor to input to the model.
            layers (List[str]): List of layer names to visualize CAMs for.
            alpha (float): Alpha value for blending CAMs on the original image.
        """
        try:
            # Retrieve the CAM from several layers at the same time
            cam_extractor = LayerCAM(self.model, layers)
            # Preprocess your data and feed it to the model
            output = self.model(input_tensor.unsqueeze(0))
            # Retrieve the CAM by passing the class index and the model output
            cams = cam_extractor(output.squeeze(0).argmax().item(), output)
            logging.info("Successfully retrieved CAMs for multiple layers")

            cam_per_layer_list = []
            # Get the cam per target layer provided
            for cam in cams:
                cam_per_layer_list.append(cam.shape)

            logging.info(f"The cams per target layer are: {cam_per_layer_list}")

            # Raw CAM
            _, axes = plt.subplots(1, len(cam_extractor.target_names))
            for id, name, cam in zip(
                range(len(cam_extractor.target_names)), cam_extractor.target_names, cams
            ):
                axes[id].imshow(cam.squeeze(0).numpy())
                axes[id].axis("off")
                axes[id].set_title(name)
            plt.show()

            fused_cam = cam_extractor.fuse_cams(cams)
            # Plot the raw version
            plt.imshow(fused_cam.squeeze(0).numpy())
            plt.axis("off")
            plt.title(" + ".join(cam_extractor.target_names))
            plt.show()
            # Plot the overlayed version
            result = overlay_mask(
                transforms.functional.to_pil_image(img),
                transforms.functional.to_pil_image(fused_cam, mode="F"),
                alpha=alpha,
            )
            plt.imshow(result)
            plt.axis("off")
            plt.title(" + ".join(cam_extractor.target_names))
            plt.show()
            cam_extractor.remove_hooks()

        except Exception as e:
            logging.error(f"Error retrieving CAMs for multiple layers: {e}")
            raise

    def get_localisation_mask(self, input_tensor: torch.Tensor, img: torch.Tensor):
        """
        Generates and visualizes localization masks based on CAMs.

        Args:
            input_tensor (torch.Tensor): The tensor input to the model.
            img (torch.Tensor): The original image tensor.
        """
        try:
            # Retrieve CAM for differnet layers at the same time
            cam_extractor = LayerCAM(self.model)
            output = self.model(input_tensor.unsqueeze(0))
            cams = cam_extractor(output.squeeze(0).argmax().item(), output)

            # Transformations
            resized_cams = [
                transforms.functional.resize(
                    transforms.functional.to_pil_image(cam.squeeze(0)), img.shape[-2:]
                )
                for cam in cams
            ]
            segmaps = [
                transforms.functional.to_pil_image(
                    (
                        transforms.functional.resize(cam, img.shape[-2:]).squeeze(0)
                        >= 0.5
                    ).to(dtype=torch.float32)
                )
                for cam in cams
            ]

            # Plots
            for name, cam, seg in zip(
                cam_extractor.target_names, resized_cams, segmaps
            ):
                _, axes = plt.subplots(1, 2)
                axes[0].imshow(cam)
                axes[0].axis("off")
                axes[0].set_title(name)
                axes[1].imshow(seg)
                axes[1].axis("off")
                axes[1].set_title(name)
                plt.show()
            cam_extractor.remove_hooks()

        except Exception as e:
            logging.error(f"Error generating localization masks: {e}")
            raise

    def generate_explanation(
        self,
        img: torch.Tensor,
        input_tensor: torch.Tensor,
        target_layer: Optional[str] = None,
        localisation_mask: bool = True,
        multiple_layers: List[str] = [],
        alpha=0.5,
    ):
        """
        Extracts and visualizes CAMs for a target layer or multiple layers.

        Args:
            img (torch.Tensor): The original image tensor.
            input_tensor (torch.Tensor): The tensor input to the model.
            target_layer (Optional[str]): The target layer for CAM visualization.
            localisation_mask (bool): Whether to generate localization masks.
            multiple_layers (List[str]): Layers for multi-layer CAM visualization.
            alpha (float): Alpha value for blending CAMs on the original image.
        """
        try:
            cam_extractor = SmoothGradCAMpp(self.model, target_layer=target_layer)
            output = self.model(input_tensor.unsqueeze(0))
            # Get the CAM giving the class index and output
            cams = cam_extractor(output.squeeze(0).argmax().item(), output)

            cam_per_layer_list = []
            # Get the cam per target layer provided
            for cam in cams:
                cam_per_layer_list.append(cam.shape)

            logging.info(f"The cams per target layer are: {cam_per_layer_list}")

            # The raw CAM
            for name, cam in zip(cam_extractor.target_names, cams):
                plt.imshow(cam.squeeze(0).numpy())
                plt.axis("off")
                plt.title(name)
                plt.show()

            # Overlayed on the image
            for name, cam in zip(cam_extractor.target_names, cams):
                result = overlay_mask(
                    transforms.functional.to_pil_image(img),
                    transforms.functional.to_pil_image(cam.squeeze(0), mode="F"),
                    alpha=alpha,
                )
                plt.imshow(result)
                plt.axis("off")
                plt.title(name)
                plt.show()

            cam_extractor.remove_hooks()

            if localisation_mask:
                self.get_localisation_mask(input_tensor, img)

            if len(multiple_layers) > 0:
                self.get_multiple_layers_result(
                    img, input_tensor, multiple_layers, alpha
                )

        except Exception as e:
            logging.error(f"Error extracting CAM: {e}")
            raise
