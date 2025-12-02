import torch

class MyTransform:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a custom transformation: per-image standardization, 
        followed by contrast scaling and clamping.

        x: (1, 28, 28) float tensor in [0,1]
        Return: transformed tensor, same shape/dtype.
        Must be non-identity and deterministic.
        """
        # Calculate the mean and standard deviation across all pixels of the image
        mu = x.mean()
        epsilon = 1e-6
        sigma = torch.max(x.std(), torch.tensor(epsilon, dtype=x.dtype, device=x.device))
        
        x_standardized = (x - mu) / sigma

        # Increase the contrast/scale of the standardized image
        contrast_factor = 1.5 
        x_scaled = x_standardized * contrast_factor

        # Clamp the output to a reasonable range to stabilize training
        # This prevents extremely large values resulting from the scaling
        x_transformed = x_scaled.clamp(min=-3.0, max=3.0)
        
        # The result retains the shape (1, 28, 28) and the original dtype (float)
        return x_transformed