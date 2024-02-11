import pytest
from unittest.mock import patch
import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend
import matplotlib.pyplot as plt
from torchvision.models import resnet50, vgg19
from torchvision.io.image import read_image
from easy_explain import CAMExplain

resnet50_model = resnet50(weights="ResNet50_Weights.DEFAULT").eval()
vgg19_model = vgg19(weights="VGG19_Weights.DEFAULT").eval()
img = read_image("examples/data/nam-anh-QJbyG6O0ick-unsplash.jpg")
trans_params = {
    "ImageNet_transformation": {
        "Resize": {"h": 224, "w": 224},
        "Normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
}


@pytest.mark.parametrize("target_layer", [None, "features.29"])
def test_SmoothGradCAMpp_explainer(target_layer):
    explainer = CAMExplain(model=vgg19_model)
    input_tensor = explainer.transform_image(
        img, trans_params["ImageNet_transformation"]
    )

    with patch("matplotlib.pyplot.show") as mock_show:
        explainer.generate_explanation(img, input_tensor, target_layer=target_layer)
        # Close all matplotlib figures
        plt.close("all")
        # Assert plt.show() was called, indicating plots were generated
        assert mock_show.called


def test_LayerCAM_explainer_no_exceptions():
    explainer = CAMExplain(resnet50_model)
    input_tensor = explainer.transform_image(
        img, trans_params["ImageNet_transformation"]
    )

    # Explicitly documenting that no exceptions are expected
    try:
        explainer.generate_explanation(
            img, input_tensor, multiple_layers=["layer2", "layer3", "layer4"]
        )
    except Exception:
        pytest.fail("Unexpected exception was raised.")

    plt.close("all")


def test_LayerCAM_explainer_plots_closed():
    explainer = CAMExplain(resnet50_model)
    input_tensor = explainer.transform_image(
        img, trans_params["ImageNet_transformation"]
    )

    # Get the number of figures before the explanation generation
    initial_fig_count = len(plt.get_fignums())

    with patch("matplotlib.pyplot.show") as mock_show:
        explainer.generate_explanation(
            img, input_tensor, multiple_layers=["layer2", "layer3", "layer4"]
        )
        # Assert plt.show() was called, indicating plots were generated
        assert mock_show.called

    # Get the number of figures after the explanation generation
    final_fig_count = len(plt.get_fignums())

    # Assert that figures were created
    assert final_fig_count > initial_fig_count, "No new figures were created."

    plt.close("all")  # Close all figures

    # Assert that all figures are closed
    assert len(plt.get_fignums()) == 0, "Not all figures were closed."
