import numpy as np


def smart_scaling(seq, min_values, max_values):
    assert seq.ndim == 3, "Input sequence must be 3D: (batch, time, feature)"
    assert min_values.shape == max_values.shape == (seq.shape[0], 1, seq.shape[2]), \
        "Min and max values must have shape (batch, 1, feature)"

    batch, time, feature = seq.shape
    first_elements = seq[:, 0:1, :]  # Preserve the first elements
    assert np.all(first_elements < max_values) and np.all(first_elements > min_values), \
        f"The first elements must be within min and max values: \n" \
        f"first_elements:{first_elements}, \nmin_values:{min_values}, \nmax_values:{max_values}"

    # Calculate scaling factors for each batch
    seq_max_values = np.max(seq, axis=1, keepdims=True)  # Include the first element
    seq_min_values = np.min(seq, axis=1, keepdims=True)  # Include the first element

    # Apply scaling to the sequences that exceed the max values
    for i in range(batch):
        for j in range(feature):
            # 如果存在大于max的值，则把值介于(first,max)的数值进行scale
            tmp_seq = seq[i, :, j]
            first_value = first_elements[i, 0, j]
            max_value = max_values[i, 0, j]
            min_value = min_values[i, 0, j]
            seq_max_value = seq_max_values[i, 0, j]
            seq_min_value = seq_min_values[i, 0, j]
            if seq_max_value > max_value:
                scale = (max_value - first_value) / (seq_max_value - first_value)
                upper_mask = tmp_seq > first_value
                seq[i, upper_mask, j] = first_value + (seq[i, upper_mask, j] - first_value) * scale
            if seq_min_value < min_value:
                scale = (min_value - first_value) / (seq_min_value - first_value)
                lower_mask = tmp_seq < first_value
                seq[i, lower_mask, j] = first_value + (seq[i, lower_mask, j] - first_value) * scale
    return seq


# Example usage
# seq = np.random.rand(2, 3, 1)  # Example sequence with shape (batch, time, feature)
seq = np.array([[0.22, 0.53, 0.24], [0.55, 0.56, 0.77]]).reshape(2, 3, 1)
print(seq.shape)
min_values = np.array([[0.2], [0.5]]).reshape(2, 1, 1)
max_values = np.array([[0.4], [0.7]]).reshape(2, 1, 1)

print('seq', seq)
scaled_seq = smart_scaling(seq, min_values, max_values)
print('scaled_seq', scaled_seq)
