import torch
import torch.nn.functional as F


def simple_conv2d(input_matrix: torch.Tensor, kernel: torch.Tensor, padding: int, stride: int) -> torch.Tensor:
    """
    Perform a 2D convolution on a single-channel input using PyTorch's built-in conv2d.

    Args:
        input_matrix: 2D tensor of shape (H, W)
        kernel: 2D tensor of shape (kH, kW)
        padding: int, zero-padding on all sides
        stride: int, stride of the convolution

    Returns:
        2D tensor (H_out, W_out) - result of convolution
    """
    # Reshape input -> (N, C, H, W)
    input_tensor = input_matrix.unsqueeze(0).unsqueeze(0)  # shape (1, 1, H, W)

    # Reshape kernel -> (out_channels, in_channels, kH, kW)
    kernel_tensor = kernel.unsqueeze(0).unsqueeze(
        0)       # shape (1, 1, kH, kW)

    # Perform convolution
    output = F.conv2d(input_tensor, kernel_tensor,
                      stride=stride, padding=padding)

    # Remove batch & channel dims -> (H_out, W_out)
    return output.squeeze(0).squeeze(0)
