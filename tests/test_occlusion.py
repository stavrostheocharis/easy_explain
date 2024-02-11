import pytest
import torchvision
from unittest.mock import patch, MagicMock
from easy_explain.methods.occlusion import OcclusionExplain
import matplotlib.pyplot as plt


# Test that plots are created and then closed
def test_occlusion_generation():
    model = torchvision.models.resnet50(pretrained=True).eval()
    occlusion_explain = OcclusionExplain(model=model)

    image_url = "https://unsplash.com/photos/aGQMKvPiBN4/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8M3x8c3F1aXJyZWwlMjBtb25rZXl8ZW58MHx8fHwxNjc1NzczNTIy&force=true"
    total_preds = 5
    vis_types = [
        ["blended_heat_map", "original_image"],
        ["blended_heat_map", "original_image"],
        ["alpha_scaling", "original_image"],
    ]
    vis_signs = [["positive", "all"], ["negative", "all"], ["positive", "all"]]
    labels_path = "examples/occlusion/imagenet_class_index.json"
    target = 7

    # Patch plt.show to simulate plot being displayed and automatically closed
    with patch("matplotlib.pyplot.show") as mock_show:
        mock_show.side_effect = plt.close("all")

        # Execute the method under test
        occlusion_explain.generate_explanation(
            image_url=image_url,
            total_preds=total_preds,
            vis_types=vis_types,
            vis_signs=vis_signs,
            labels_path=labels_path,
            target=target,
        )

        # Assert plt.show() was called the expected number of times (once per plot)
        assert mock_show.call_count == len(vis_types)
