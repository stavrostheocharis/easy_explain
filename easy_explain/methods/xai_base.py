from abc import ABC, abstractmethod


class ExplainabilityMethod(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def generate_explanation(self, **kwargs):
        """
        Generate an explanation for a given input.

        Args:
            **kwargs: A flexible argument dictionary to accommodate specific needs of each explainability method.
        """
        pass
