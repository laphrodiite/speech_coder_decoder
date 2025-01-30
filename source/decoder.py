import numpy as np
from scipy.signal import lfilter
from source.hw_utils import reflection_coeff_to_polynomial_coeff
from source.utils import decode_LARc_to_reflection

def RPE_frame_st_decoder(LARc: np.ndarray, curr_frame_st_resd: np.ndarray) -> np.ndarray:
    """
        Decode the short-term residual and LAR coefficients
        to reconstruct the original frame signal.
    """
    r = decode_LARc_to_reflection(LARc)
    ak, _ = reflection_coeff_to_polynomial_coeff(r)

    # Short-Term Synthesis Filtering
    # Hs(z) = 1 / (1 - ∑ ak * z^(-k)) -> Convolution in the time domain
    s0 = lfilter([1], np.concatenate(([1], -ak[1:])), curr_frame_st_resd)
    return s0

