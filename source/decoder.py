import numpy as np
from scipy.signal import lfilter
from hw_utils import reflection_coeff_to_polynomial_coeff
from utils import decode_LARc_to_reflection, beta
from typing import List, Tuple
import struct
from bitstring import BitArray

def RPE_frame_st_decoder(LARc: np.ndarray, curr_frame_st_resd: np.ndarray) -> np.ndarray:
    """
        Decode the short-term residual and LAR coefficients
        to reconstruct the original frame signal.
    """
    r = decode_LARc_to_reflection(LARc)
    ak, _ = reflection_coeff_to_polynomial_coeff(r)

    # Short-Term Synthesis Filtering
    # Hs(z) = 1 / (1 - ∑ ak * z^(-k)) -> Convolution in the time domain
    s_prime = lfilter([1], np.concatenate(([1], -ak[1:])), curr_frame_st_resd)
    
    # Postprocessing step included
    # IIR filter coefficients
    b = [1]  # Numerator (direct gain)
    a = [1, -beta]  # Denominator (feedback term)

    # Apply the filter
    s0 = lfilter(b, a, s_prime)
    return s0


def RPE_frame_slt_decoder(LARc: np.ndarray,
                          Nc: List[int],
                          bc: List[int],
                          curr_frame_ex_full: np.ndarray,
                          prev_frame_st_resd: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
    
    """
        It is assumed that Nc and bc are the dequantized parameters
        and the overall error signal curr_frame_ex_full is formed from the 
        parameters M', x'max and the 13 values of x'M(i)
    """

    frame_length = 160
    subframe_length = 40
    # Initialize the array for the reconstructed short term residual d'(n)
    curr_frame_st_resd = np.zeros(frame_length)
    
    # Step 2 of paragraph 2.2
    # Process each of the 4 subframes
    for subframe in range(4):
        # Calculate the start and end indices for the current subframe
        start = subframe * subframe_length
        end = start + subframe_length
        
        # Get long term prediction parameters for this subframe.
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
    
    # Steps 3-5 of paragraph 2.2
    # Now, use the short term synthesis decoder to reconstruct the final speech signal.
    s0 = RPE_frame_st_decoder(LARc, curr_frame_st_resd)
    
    return s0, curr_frame_st_resd


# --------------------------------- This is the thiiiiiiird paradoteo --------------------------------- #

def RPE_frame_decoder(frame_bit_stream: BitArray, prev_frame_resd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode the bitstream and reconstruct the frame.

    :param frame_bit_stream: The bitstream (260 bits) representing the frame.
    :param prev_frame_resd: The short-term residual of the previous frame (160 samples).
    :return: A tuple containing the reconstructed signal (160 samples) and the current frame's short-term residual.
    """
    # Step 1: Parse the bitstream
    index = 0

    # Decode LARc (Log-Area Ratios)
    LARc = [
        frame_bit_stream[index:index+6].int,  # LARc[1] - 6 bits (signed)
        frame_bit_stream[index+6:index+12].int,  # LARc[2] - 6 bits (signed)
        frame_bit_stream[index+12:index+17].int,  # LARc[3] - 5 bits (signed)
        frame_bit_stream[index+17:index+22].int,  # LARc[4] - 5 bits (signed)
        frame_bit_stream[index+22:index+26].int,  # LARc[5] - 4 bits (signed)
        frame_bit_stream[index+26:index+30].int,  # LARc[6] - 4 bits (signed)
        frame_bit_stream[index+30:index+33].int,  # LARc[7] - 3 bits (signed)
        frame_bit_stream[index+33:index+36].int,  # LARc[8] - 3 bits (signed)
    ]
    index += 36

    # Decode subframe parameters
    Nc, bc, curr_frame_ex_full = [], [], []
    for _ in range(4):  # 4 subframes
        # Nc - 7 bits (LTP lag, signed)
        Nc.append(frame_bit_stream[index:index+7].uint)
        index += 7
        # bc - 2 bits (LTP gain, signed)
        bc.append(frame_bit_stream[index:index+2].uint)
        index += 2
        # xMc (curr_frame_ex_full) - 40 samples per subframe, 3 bits per sample (signed)
        subframe_ex = []
        for _ in range(40):
            subframe_ex.append(frame_bit_stream[index:index+3].int)
            index += 3
        curr_frame_ex_full.extend(subframe_ex)

    # Convert curr_frame_ex_full to a numpy array
    curr_frame_ex_full = np.array(curr_frame_ex_full)

    # Step 2: Decode the frame using short-term and long-term prediction
    s0, curr_frame_st_resd = RPE_frame_slt_decoder(LARc, Nc, bc, curr_frame_ex_full, prev_frame_resd)

    return s0, curr_frame_st_resd