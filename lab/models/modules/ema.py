import torch
from torch import nn

class BiasCorrectedEMA(nn.Module):
    """
    Computes an Exponential Moving Average (EMA) with bias correction.
    Uses tensor.lerp_ for efficient in-place updates.
    """

    def __init__(self, shape: tuple | int, decay: float = 0.9):
        super().__init__()
        self.decay = decay

        # 1. Register the EMA state as a buffer (initialized to zeros)
        # We accept an int or a tuple for the shape.
        if isinstance(shape, int):
            shape = (shape,)
        self.register_buffer("ema_state", torch.zeros(shape))

        # 2. Register the step count as a buffer (initialized to zero)
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

    def forward(self, new_value: torch.Tensor) -> torch.Tensor:
        """
        Updates the EMA state with the new value and returns the bias-corrected mean.
        """
        # We use torch.no_grad() because we don't want the EMA buffer
        # to track gradients or become part of the backprop graph.
        with torch.no_grad():
            # Increment the call counter
            self.step_count += 1

            # Update the internal state efficiently using lerp_
            # Formula: ema_state = ema_state * decay + new_value * (1 - decay)
            self.ema_state.lerp_(new_value, weight=1.0 - self.decay)

            # Compute the bias correction factor: (1 - decay^step)
            # We use the tensor self.step_count directly
            bias_correction = 1.0 - (self.decay ** self.step_count)

            # Compute the corrected value
            corrected_ema = self.ema_state / bias_correction

        return corrected_ema

    def reset(self):
        self.ema_state = torch.zeros_like(self.ema_state)
        self.step_count = torch.tensor(0, dtype=torch.long)

    def get_current_mean(self) -> torch.Tensor:
        """
        Helper method to get the current corrected mean WITHOUT updating it.
        Useful for inference or logging.
        """
        if self.step_count == 0:
            return self.ema_state

        bias_correction = 1.0 - (self.decay ** self.step_count)
        return self.ema_state / bias_correction