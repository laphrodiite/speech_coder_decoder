import numpy as np

from utils import A, B

def decode_LARc_to_reflection(LARc: np.ndarray) -> np.ndarray:
    """ Decode quantized LARc values back into reflection coefficients. """
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