import numpy as np
from scipy.signal import lfilter
from hw_utils import reflection_coeff_to_polynomial_coeff
from utils import decode_LARc_to_reflection
from typing import List, Tuple

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

def RPE_frame_slt_decoder(LARc: np.ndarray,
                          Nc: List[int],
                          bc: List[int],
                          curr_frame_ex_full: np.ndarray,
                          prev_frame_st_resd: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:

    frame_length = 160
    subframe_length = 40
    # Initialize the array for the reconstructed short term residual d'(n)
    curr_frame_st_resd = np.zeros(frame_length)
    
    # Process each of the 4 subframes
    for subframe in range(4):
        # Calculate the start and end indices for the current subframe
        start = subframe * subframe_length
        end = start + subframe_length
        
        # Get long term prediction parameters for this subframe.
        # They are assumed to be dequantized.
        N = Nc[subframe]
        b = bc[subframe]
        
        # Process each sample in the subframe
        for n in range(start, end):
            # Compute the index for the delayed sample: n - N
            delayed_index = n - N
            
            # If the delayed index is within the current frame, use the reconstructed residual.
            # Otherwise, use the previous frame's residual.
            if delayed_index >= 0:
                d_prev = curr_frame_st_resd[delayed_index]
            else:
                # Adjust negative indices to fetch from the end portion of the previous frame.
                d_prev = prev_frame_st_resd[frame_length + delayed_index]
            
            # Long term prediction:
            # d'_pred(n) = b * d'(n - N)
            predicted = b * d_prev
            
            # Reconstruct the short term residual for the current sample:
            # d'(n) = e(n) + d'_pred(n)
            curr_frame_st_resd[n] = curr_frame_ex_full[n] + predicted
            
    # Now, use the short term synthesis decoder to reconstruct the final speech signal.
    s0 = RPE_frame_st_decoder(LARc, curr_frame_st_resd)
    
    return s0, curr_frame_st_resd

if __name__ == '__main__':
    # Create example (dummy) data for demonstration purposes.
    # In practice, these values are obtained from the transmission or previous decoding steps.
    LARc = np.random.randn(8)                # Example LAR coefficients
    Nc = [60, 65, 70, 75]                    # Example pitch delays for the 4 subframes (each between 40 and 120)
    bc = [0.7, 0.8, 0.85, 0.9]                 # Example long term prediction gains
    curr_frame_ex_full = np.random.randn(160)  # Example overall prediction error signal e(n)
    prev_frame_st_resd = np.random.randn(160)    # Example previous frame residual d'(n)
    
    # Run the long-term synthesis decoder for the current frame
    s0, curr_frame_st_resd = RPE_frame_slt_decoder(LARc, Nc, bc, curr_frame_ex_full, prev_frame_st_resd)
    
    # Output some of the results for verification
    print("Synthesized speech signal (first 10 samples):", s0[:10])
    print("Reconstructed short term residual (first 10 samples):", curr_frame_st_resd[:10])