"""Validation and preprocessing of data."""

import numpy


def validate_input_shape(input_yx: tuple[int], expected_tile_size: tuple[int]):
    assert all((x % expected_tile_size == 0) for x in input_yx), (
        f"Invalid input shape {input_yx}. Last dims should be divisible by {expected_tile_size}"
    )


def channel_chunks_rigid3(pixels: numpy.ndarray) -> list[numpy.ndarray]:
    """Split an N-channel image into 3-channel chunks for a rigid 3-channel
    backbone (ImageNet-pretrained ViTs, ResNets, etc.).

    Drops the Z axis if present (takes the first slice, matching
    :func:`pad_channel_dim`'s convention). Accepts ``(N, C, Z, Y, X)`` or
    ``(N, C, Y, X)``; all returned chunks are 4-D ``(N, 3, Y, X)``.

    - ``C == 3``: single chunk, input as-is.
    - ``C  < 3``: single chunk, zero-padded to 3 channels.
    - ``C  > 3``: ``ceil(C / 3)`` chunks. Missing slots in the trailing chunk
      recycle leading channels (modular wrap):

      ====  ====================================
       C    chunk channel indices
      ====  ====================================
       4    ``[(0,1,2), (3,0,1)]``
       5    ``[(0,1,2), (3,4,0)]``
       6    ``[(0,1,2), (3,4,5)]``
       7    ``[(0,1,2), (3,4,5), (6,0,1)]``
      ====  ====================================

    Callers concatenate the per-chunk feature outputs along the feature axis
    to produce a ``(N, D · ceil(C/3))`` embedding.
    """
    if pixels.ndim == 5:
        pixels = pixels[:, :, 0]
    if pixels.ndim != 4:
        raise ValueError(
            f"Expected (N, C, Z, Y, X) or (N, C, Y, X), got shape {pixels.shape}"
        )
    n, c = pixels.shape[:2]
    if c == 3:
        return [pixels]
    if c < 3:
        padded = numpy.zeros((n, 3, *pixels.shape[2:]), dtype=pixels.dtype)
        padded[:, :c] = pixels
        return [padded]
    num_rounds = (c + 2) // 3  # ceil(c / 3)
    return [pixels[:, [(r * 3 + i) % c for i in range(3)]] for r in range(num_rounds)]


def pad_channel_dim(pixels: numpy.ndarray, expected_channels: int) -> numpy.ndarray:
    """Pads the channel dimension of a numpy array to a target size.

    This function is designed to work with image data where the channel
    dimension is not the last one. It makes two key assumptions based on the
    original implementation:
    1. The z-stack (third dimension, index 2) is removed by taking the
       first slice.
    2. The channel dimension to be padded is the second dimension (index 1).

    If the number of channels after slicing the z-stack is less than
    `expected_channels`, it is padded with zeros.

    Parameters
    ----------
    pixels : numpy.ndarray
        The input image data, expected to have at least 3 dimensions.
        For example, a shape like (H, W, Z, ...).
    expected_channels : int
        The desired number of channels for the output array.

    Returns
    -------
    numpy.ndarray
        The processed array with the z-stack removed and the channel
        dimension padded to `expected_channels`. If the array already has
        enough channels, it is returned as is after z-stack removal.

    """
    # Note: This logic is preserved from the original implementation.
    # It assumes a specific data layout where the z-stack is the 3rd dimension.
    if pixels.ndim > 2:
        pixels = pixels[:, :, 0]

    # None signals "no padding constraint" (e.g. channel-agnostic models).
    if expected_channels is None:
        return pixels

    input_channels = pixels.shape[1]
    to_pad = expected_channels - input_channels

    if to_pad > 0:
        padding_shape = list(pixels.shape)
        padding_shape[1] = to_pad
        padding = numpy.zeros(padding_shape, dtype=pixels.dtype)
        pixels = numpy.concatenate((pixels, padding), axis=1)

    return pixels
