import torch

class Clamp:
    """
    Clamps the tensor values between min and max.
    This works compatible with torchvision.transforms.Compose.
    """
    def __init__(self, min_: torch.Tensor | float | None = None, max_: torch.Tensor | float | None = None):
        """
        Args:
            min_ (float, optional): Lower bound. If None, no lower bound.
            max_ (float, optional): Upper bound. If None, no upper bound.
        """
        self.min_ = min_
        self.max_ = max_

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be clamped.
        Returns:
            Tensor: Clamped tensor.
        """
        return torch.clamp(tensor, min=self.min_, max=self.max_)

    def __repr__(self) -> str:
        # This makes the transform print nicely: "Clamp(min=0.0, max=1.0)"
        return f"{self.__class__.__name__}(min={self.min_}, max={self.max_})"