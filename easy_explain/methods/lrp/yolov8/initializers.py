import torch
from .utils import LayerRelevance


class YOLOv8RelevanceInitializer(object):

    """
    Assign initial relevance for YOLOv5 model explanation

    Attributes
    ----------

    cls : int

        Index to the class of interest.

    conf_thres : float

        Threshold set for object detection confidence. All output tiles
        with a confidence score lower than this will be truncated to zero

    max_class_only : bool

        Zero all output activations from classes that are not the max.

    contrastive : bool

        Whether to implement relevance as contrastive or not.

    Methods
    -------

    set_class(cls=None) :
        Set class of interest

    set_prediction(prediction=None) :
        Set prediction

    __call__(prediction : list) :
        Set initial relevance based on prediction made by YOLOv5 model

    """

    def __init__(
        self,
        cls: int = None,
        conf: bool = False,
        max_class_only: bool = False,
        contrastive: bool = False,
    ):
        if contrastive:
            assert (
                cls is not None
            ), "Contrastive implementation of lrp requires target class specification"

        self.cls = cls
        self.conf = conf
        self.max_class_only = max_class_only
        self.contrastive = contrastive

        # prop_to has to do with the YOLOv5 head architecture, more specifically it defines
        # the module numbers that relevance originates from. If this chages the list bellow
        # must be manually changed.
        self.prop_to = [15, 18, 21]

    def __call__(self, cls_preds: list):
        """
        Initialize relevance based on the model's predictions.

        Args:
            cls_preds (list): List of class prediction tensors from the YOLO model.

        Returns:
            LayerRelevance: An object containing the initialized relevance scores.
        """
        initial_relevance = []
        for cls_pred in cls_preds:
            # Filter for the class of interest if specified
            if self.cls is not None:
                # For contrastive, keep dual relevance; otherwise, filter directly
                if self.contrastive:
                    dual_relevance = cls_pred.clone()
                    dual_relevance[
                        :, self.cls
                    ] = 0  # Zero out the class of interest in the dual tensor
                    cls_pred.zero_()  # Reset cls_pred to zero
                    cls_pred[:, self.cls] = dual_relevance[
                        :, self.cls
                    ]  # Assign class of interest relevance
                    cls_pred = torch.cat(
                        [cls_pred, dual_relevance], dim=0
                    )  # Combine for contrastive relevance
                else:
                    temp = cls_pred[
                        :, self.cls
                    ].clone()  # Temporarily store relevance for the class of interest
                    cls_pred.zero_()  # Zero out all activations
                    cls_pred[
                        :, self.cls
                    ] = temp  # Restore relevance for the class of interest

            # Apply max class only filtering if specified
            if self.max_class_only:
                max_vals, _ = cls_pred.max(dim=1, keepdim=True)
                cls_pred = torch.where(
                    cls_pred == max_vals, cls_pred, torch.zeros_like(cls_pred)
                )

            initial_relevance.append(cls_pred)

        # Note: The normalization step mentioned in the original code was not applied to 'norm'
        return LayerRelevance(
            relevance=[(-1, initial_relevance)], contrastive=self.contrastive
        )
