import torch
from torch.nn.functional import max_pool2d
from typing import Tuple


def prop_SPPF(inverter, mod: torch.nn.Module, relevance: torch.Tensor) -> torch.Tensor:
    """
    Propagate relevance through an SPPF module.

    Args:
        inverter: The relevance inverter object.
        mod: The SPPF module being processed.
        relevance: The relevance tensor from the subsequent layer.

    Returns:
        The propagated relevance tensor for the input of the SPPF module.
    """
    relevance = inverter(mod.cv2, relevance)
    msg = relevance.scatter(which=-1)
    ch = msg.size(1) // 4

    r3 = msg[:, 3 * ch : 4 * ch, ...]
    r2 = msg[:, 2 * ch : 3 * ch, ...] + r3
    r1 = msg[:, ch : 2 * ch, ...] + r2
    rx = msg[:, :ch, ...] + r1

    msg = inverter(mod.cv1, rx)
    relevance.gather([(-1, msg)])

    return relevance


def SPPF_fwd_hook(
    m: torch.nn.Module, in_tensor: Tuple[torch.Tensor], out_tensor: torch.Tensor
):
    """
    Forward hook for an SPPF module to capture max pooling indices.

    Args:
        m: The module the hook is attached to.
        in_tensor: The input tensors to the module.
        out_tensor: The output tensor from the module.
    """
    # Validate assumptions about module structure
    if not hasattr(m, "cv1") or not hasattr(m, "m"):
        raise AttributeError(
            "Module does not have expected attributes for SPPF_fwd_hook."
        )

    x = m.cv1(in_tensor[0])
    indices = []

    for _ in range(3):  # Assuming three max_pool2d operations
        x, idx = max_pool2d(
            x,
            kernel_size=m.m.kernel_size,
            stride=m.m.stride,
            padding=m.m.padding,
            dilation=m.m.dilation,
            return_indices=True,
            ceil_mode=m.m.ceil_mode,
        )
        indices.append(idx)

    # Save indices for later use in relevance propagation
    m.indices = indices


def Concat_fwd_hook(
    m: torch.nn.Module, in_tensors: torch.Tensor, out_tensor: torch.Tensor
):
    """
    Forward hook for capturing input and output shapes of a Concat module.
    """
    shapes = [in_tensor.shape[m.d] for in_tensor in in_tensors[0]]

    setattr(m, "in_shapes", shapes)
    setattr(m, "out_shape", out_tensor.shape)


def prop_Concat(*args):
    """
    Propagates relevance through a concatenation operation.
    """
    _, mod, relevance = args

    slices = relevance.scatter(-1).split(mod.in_shapes, dim=mod.d)
    relevance.gather([(to, msg) for to, msg in zip(mod.f, slices)])

    return relevance


def prop_Detect(*args):
    """
    Propagates relevance through a detection layer.
    """
    inverter, mod, relevance = args
    relevance_out = []

    _, scattered = relevance.scatter()[0]
    prop_to = [21, 18, 15][::-1]
    for i, rel in enumerate(scattered):
        relevance_out.append((prop_to[i], inverter(mod.cv3[i], rel)))
        inverter(mod.cv3[i], rel)

    relevance.gather(relevance_out)
    return relevance


def prop_Conv(
    inverter, module: torch.nn.Module, relevance: torch.Tensor
) -> torch.Tensor:
    """
    Propagates relevance through a convolutional layer.
    """
    # This function is already quite streamlined. Added a docstring for consistency.
    return inverter(module.conv, relevance)


def prop_C3(inverter, mod: torch.nn.Module, relevance: torch.Tensor) -> torch.Tensor:
    """
    Propagates relevance through C3 blocks in the network.

    Args:
        inverter: The relevance inverter object.
        mod: The module being processed.
        relevance: The incoming relevance tensor.

    Returns:
        The propagated relevance tensor.
    """
    # Scatter relevance for processing
    msg = relevance.scatter(which=-1)
    c_ = msg.size(1)
    msg_cv1, msg_cv2 = msg.split(c_ // 2, dim=1)

    # Process through C3 blocks
    for m1 in mod.m:
        msg_cv1 = inverter(m1, msg_cv1)

    # Combine and propagate relevance through cv1 and cv2
    msg = inverter(mod.cv1, msg_cv1) + inverter(mod.cv2, msg_cv2)
    relevance.gather([(-1, msg)])
    return relevance


def prop_Bottleneck(
    inverter, mod: torch.nn.Module, relevance: torch.Tensor
) -> torch.Tensor:
    """
    Propagates relevance through Bottleneck blocks.

    Args:
        inverter: The relevance inverter object.
        mod: The module being processed.
        relevance: The incoming relevance tensor.

    Returns:
        The propagated relevance tensor.
    """
    # Propagate relevance through cv1 and cv2
    relevance = inverter(mod.cv1, relevance)
    relevance = inverter(mod.cv2, relevance)
    return relevance


def prop_C2f(inverter, mod: torch.nn.Module, relevance: torch.Tensor) -> torch.Tensor:
    """
    Propagates relevance through C2f blocks.

    Args:
        inverter: The relevance inverter object.
        mod: The module being processed.
        relevance: The incoming relevance tensor.

    Returns:
        The propagated relevance tensor.
    """
    # Scatter relevance for processing
    msg = relevance.scatter(which=-1)
    msg_cv2 = inverter(mod.cv2, msg)

    # Chunk and propagate relevance through bottleneck blocks
    msg_m = list(msg_cv2.chunk(msg_cv2.size(1) // mod.c, dim=1))
    for i, m_block in enumerate(reversed(mod.m)):
        msg_m[-(i + 2)] += inverter(m_block, msg_m[-(i + 1)])

    # Combine and propagate through cv1
    msg_cv1 = torch.cat(msg_m[:2], dim=1)
    msg = inverter(mod.cv1, msg_cv1)
    relevance.gather([(-1, msg)])
    return relevance


def prop_DFL(inverter, mod: torch.nn.Module, relevance: torch.Tensor) -> torch.Tensor:
    """
    Propagates relevance through DFL blocks.

    Args:
        inverter: The relevance inverter object.
        mod: The module being processed.
        relevance: The incoming relevance tensor.

    Returns:
        The propagated relevance tensor.
    """
    # Assuming 'in_shape' is accessible and relevant here; otherwise, adjust accordingly
    a = mod.in_shape[-1]
    relevance = inverter(relevance.unsqueeze(0), mod.conv.weight.data).transpose(2, 1)
    relevance = torch.cat(
        [relevance[:, :, :, ai].flatten().unsqueeze(-1) for ai in range(a)], dim=-1
    ).unsqueeze(0)
    return relevance
