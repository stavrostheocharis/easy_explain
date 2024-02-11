import requests
from io import BytesIO
import torchvision
from unittest.mock import patch
from ultralytics import YOLO
from PIL import Image
import torch
import pytest
import matplotlib.pyplot as plt


from easy_explain import YOLOv8LRP


model = YOLO("ultralyticsplus/yolov8s")
response = requests.get(
    "https://images.unsplash.com/photo-1584649525122-8d6895492a5d?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
)
image = Image.open(BytesIO(response.content))

desired_size = (512, 640)
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(desired_size),
        torchvision.transforms.ToTensor(),
    ]
)
image = transform(image)


def test_lrp_yolov8_explanation():
    lrp = YOLOv8LRP(model, power=2, eps=1, device="cpu")
    explanation_lrp = lrp.explain(image, cls="traffic light", contrastive=False).cpu()

    assert explanation_lrp is not None, "Explanation should not be None"
    assert isinstance(
        explanation_lrp, torch.Tensor
    ), "Explanation should be a torch.Tensor"

    with patch("matplotlib.pyplot.show") as mock_show:
        lrp.plot_explanation(
            frame=image,
            explanation=explanation_lrp,
            contrastive=True,
            cmap="seismic",
            title='Explanation for Class "traffic light"',
        )
        assert mock_show.called, "Plot show method should be called"

    with patch("matplotlib.pyplot.show") as mock_show:
        lrp.plot_explanation(
            frame=image,
            explanation=explanation_lrp,
            contrastive=False,
            cmap="seismic",
            title='Explanation for Class "traffic light"',
        )
        assert mock_show.called, "Plot show method should be called"

    with patch("matplotlib.pyplot.show") as mock_show:
        lrp.plot_explanation(
            frame=image,
            explanation=explanation_lrp,
            contrastive=False,
            cmap="Reds",
            title='Explanation for Class "traffic light"',
        )
        assert mock_show.called, "Plot show method should be called"

    plt.close("all")
