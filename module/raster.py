"""Shared full-resolution prop masks and zero/nonzero boundaries."""
from pathlib import Path
import contourpy
import numpy as np

def read_binary_mask(path, shape):
    """Stream rows to a one-byte mask; retain every zero/nonzero transition."""
    mask = np.empty(shape, dtype=np.uint8)
    count = 0
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            if count >= shape[0]:
                raise ValueError('prop has more rows than configured')
            values = np.fromstring(line, sep=' ')
            if values.size != shape[1] or not np.isfinite(values).all():
                raise ValueError(f'Invalid prop row {count + 1}: expected {shape[1]} finite values')
            mask[count] = values != 0
            count += 1
    if count != shape[0]:
        raise ValueError(f'prop has {count} rows, expected {shape[0]}')
    return mask


def boundary_segments(mask, bounds, spacing, block_rows=256):
    """March at level 0.5 in overlapping strips without downsampling the grid."""
    segments = []
    x = float(bounds['xmin']) + np.arange(mask.shape[1]) * spacing
    for start in range(0, mask.shape[0] - 1, block_rows):
        stop = min(start + block_rows + 1, mask.shape[0])
        z = mask[start:stop]
        if z.min() == z.max():
            continue
        y = float(bounds['ymin']) + np.arange(start, stop) * spacing
        generator = contourpy.contour_generator(x=x, y=y, z=z, name='serial')
        segments.extend(generator.lines(0.5))
    return segments

