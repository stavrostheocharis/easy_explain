import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from torchcam.methods import SmoothGradCAMpp, LayerCAM
from torchcam.utils import overlay_mask
from easy_explain.methods.xai_base import ExplainabilityMethod
from torchvision.transforms.functional import to_pil_image, resize
from typing import List, Optional, Union


class CAMExplain(ExplainabilityMethod):
    """A class to generate explanations using CAM techniques.

    Supports both LayerCAM and SmoothGradCAMpp for generating Class Activation Maps
    to visualize areas of interest in image inputs that influence model predictions.

    Attributes:
        model: The model to explain.
        target_layers: Optional; the layers for which to generate CAMs.
        cam_type: The type of CAM method to use ('LayerCAM' or 'SmoothGradCAMpp').
        alpha: Transparency level for the overlay.
    """

    def __init__(
        self,
        model,
        target_layers: Optional[Union[List[str], List[torch.nn.Module]]] = None,
        cam_type="LayerCAM",
        alpha=0.5,
    ):
        """Initializes the GradCamExplain with a model and CAM configuration."""
        super().__init__(model)
        self.alpha = alpha
        if cam_type == "LayerCAM":
            # Assuming LayerCAM can be initialized with target layer names
            self.cam_extractor = LayerCAM(model, target_layers)
        elif cam_type == "SmoothGradCAMpp":
            # Assuming SmoothGradCAMpp can also be initialized similarly
            self.cam_extractor = SmoothGradCAMpp(
                model, target_layers=target_layers if target_layers else [None]
            )
        else:
            raise ValueError(f"Unsupported CAM type: {cam_type}")

    def generate_explanation(
        self,
        img,
        input_tensor,
        localisation_mask=False,
        multiple_layers: Optional[List[str]] = None,
        **kwargs,
    ):
        output = self.model(input_tensor)
        class_idx = output.squeeze(0).argmax().item()
        cams = self.cam_extractor(class_idx, output)

        if multiple_layers:
            self._handle_multiple_layers(img, input_tensor, multiple_layers)
        else:
            self._visualize_cam(cams, img)

        if localisation_mask:
            self._get_localisation_mask(input_tensor, img)

    def _visualize_cam(self, cams, img):
        for cam in cams:
            result = overlay_mask(
                img,  # Use directly without conversion
                transforms.functional.to_pil_image(
                    cam, mode="F"
                ),  # Convert CAM tensor to PIL Image
                alpha=self.alpha,
            )
            plt.imshow(result)
            plt.axis("off")
            plt.show()

    def _handle_multiple_layers(self, img, input_tensor, layers: List[str]):
        cam_extractor = LayerCAM(self.model, layers)
        output = self.model(input_tensor)
        class_idx = output.squeeze(0).argmax().item()
        cams = cam_extractor(class_idx, output)

        # Visualization of CAMs for each layer
        _, axes = plt.subplots(1, len(cams))
        if len(cams) > 1:
            for ax, cam, layer in zip(axes, cams, cam_extractor.target_names):
                ax.imshow(cam.squeeze().cpu().numpy(), cmap="jet")
                ax.axis("off")
                ax.set_title(layer)
        else:
            axes.imshow(cams[0].squeeze().cpu().numpy(), cmap="jet")
            axes.axis("off")
            axes.set_title(cam_extractor.target_names[0])
        plt.show()

        # Visualization of fused CAM overlay on the original image
        fused_cam = cam_extractor.fuse_cams(cams)
        result = overlay_mask(
            img,  # Use directly without conversion
            transforms.functional.to_pil_image(
                cam, mode="F"
            ),  # Convert CAM tensor to PIL Image
            alpha=self.alpha,
        )
        plt.imshow(result)
        plt.axis("off")
        plt.title("Fused CAM")
        plt.show()

        # Ensure to remove hooks after processing to clean up
        cam_extractor.remove_hooks()

    def _get_localisation_mask(self, input_tensor, img):
        # Assuming LayerCAM is used for localisation mask generation, as per your snippet
        # If needed, this could be adjusted to support other CAM types
        output = self.model(input_tensor)
        class_idx = output.squeeze(0).argmax().item()
        cams = self.cam_extractor(class_idx, output)

        # Transformations and visualization
        for cam in cams:
            resized_cam = resize(to_pil_image(cam.squeeze(0)), img.size[::-1])
            segmap = to_pil_image(
                (resize(cam, img.size[::-1]).squeeze(0) >= 0.5).to(torch.float32)
            )

            _, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(resized_cam)
            axes[0].set_title("CAM")
            axes[0].axis("off")

            axes[1].imshow(segmap)
            axes[1].set_title("Segmentation Map")
            axes[1].axis("off")

            plt.show()

        # Cleanup to remove model hooks
        self.cam_extractor.remove_hooks()

    def remove_hooks(self):
        """Remove all the hooks set by the CAM extractor."""
        self.cam_extractor.remove_hooks()
