import cv2
import numpy as np


# Size of the pattern (projector resolution)
size = (720, 1280)  # Height, Width
phase_gap = 8  # Stripe height
num_shifts = 8  # Number of shifts (45° increments)

# Initialize the base pattern
base_pattern = np.zeros(size, dtype=np.uint8)

# Create the horizontal stripe pattern
index = size[0] // phase_gap  # Number of stripes
for i in range(index):
    if i % 2 == 0:
        base_pattern[phase_gap * i : phase_gap * (i + 1), :] = 255

# Generate shifted patterns
for shift_index in range(num_shifts):
    shift_amount = (phase_gap * shift_index) * 2 // num_shifts  # Calculate the shift for this step
    shifted_pattern = np.roll(base_pattern, shift_amount, axis=0)  # Shift vertically

    # Display the shifted pattern
    breakpoint()
    cv2.imwrite(f"45_shifted_8_patterns/gap_{phase_gap}/Pattern_Shift_{shift_index * 45}deg.png", shifted_pattern)
    # cv2.waitKey(0)

# cv2.destroyAllWindows()