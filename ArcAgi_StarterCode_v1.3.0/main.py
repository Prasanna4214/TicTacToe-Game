import numpy as np
from scipy.ndimage import label


def hollow_nonzero_rectangle(matrix: np.ndarray) -> np.ndarray:
    result = matrix.copy()

    # Label connected non-zero regions
    labeled, num_features = label(matrix != 0)

    for region_id in range(1, num_features + 1):
        coords = np.argwhere(labeled == region_id)
        top, left = coords.min(axis=0)
        bottom, right = coords.max(axis=0)

        # Hollow interior for this rectangle
        result[top + 1:bottom, left + 1:right] = 0

    return result



# Example
mat = np.array(    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 6, 6, 6, 6, 0, 0, 0, 0],
        [0, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 6, 6, 6, 6, 0, 0, 0, 0],
        [0, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 6, 6, 6, 6, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

print("Original:\n", mat)
print("Hollowed:\n", hollow_nonzero_rectangle(mat))
