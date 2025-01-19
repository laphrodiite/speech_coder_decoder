import numpy as np

# Hard coded quantization parameters described in GSM 06.10 standard
A = [20.0, 20.0, 20.0, 20.0, 13.637, 15.0, 8.334, 8.824]
B = [0.0, 0.0, -5.0, -5.0, 0.184, -0.5, -0.666, -2.235]
min_values = [-32, -32, -16, -16, -8, -8, -4, -4]
max_values = [31, 31, 15, 15, 7, 7, 3, 3]

def decode_LARc_to_reflection(LARc: np.ndarray) -> np.ndarray:
    """
        Decode quantized LARc values back into reflection coefficients.
    """
    LAR = (LARc - np.array(B)) / np.array(A)
    reflection_coefficients = []
    for lar in LAR:
        abs_lar = abs(lar)
        if abs_lar < 0.675:
            reflection_coefficients.append(lar)
        elif 0.675 <= abs_lar < 1.35:
            reflection_coefficients.append(np.sign(lar) * (0.5 * (abs_lar + 0.675)))
        else:  # abs_lar >= 1.35
            reflection_coefficients.append(np.sign(lar) * ((abs_lar + 6.375) / 8))
    return np.array(reflection_coefficients)