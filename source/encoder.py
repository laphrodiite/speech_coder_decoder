import numpy as np
from scipy.signal import lfilter
from bitstring import BitArray

from hw_utils import polynomial_coeff_to_reflection_coeff
from utils_encoder import *
from utils_decoder import decode_LARc_to_reflection


# --------------------------------- PART ONE --------------------------------- #
def RPE_frame_st_coder(s0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode the current frame using short-term prediction.

    :param s0: The input signal frame.
    :return: A tuple containing the encoded LARc and the short-term residual signal.
    """
    # Preprocess the input signal
    s = preprocess_signal(s0)

    # Calculate prediction coefficients using ACF
    prediction_coefficients = calculate_acf(s)

    # Convert to reflection coefficients
    reflection_coefficients = polynomial_coeff_to_reflection_coeff(prediction_coefficients)

    # Convert to Log-Area Ratios (LAR)
    LAR = convert_reflection_to_LAR(reflection_coefficients)

    # Quantize and encode LAR values
    LARc = quantize_LAR(LAR)

    # Decode LARc to modified reflection coefficients
    decoded_reflection_coefficients = decode_LARc_to_reflection(LARc)

    # Compute residual signal using modified coefficients
    fir_coefficients = np.concatenate(([1], -decoded_reflection_coefficients))
    curr_frame_st_resd = lfilter(fir_coefficients, [1], s)

    return LARc, curr_frame_st_resd


# --------------------------------- PART TWO --------------------------------- #
def RPE_frame_slt_coder(s0: np.ndarray, prev_frame_st_resd: np.ndarray) \
        -> tuple[np.ndarray, list, list, np.ndarray, np.ndarray]:
    """
    Encode the current frame using Regular Pulse Excitation (RPE) with long-term prediction (LTP).

    :param s0: The input signal frame.
    :param prev_frame_st_resd: The short-term residual of the previous frame.
    :return: A tuple containing:
             - LARc: The encoded Log-Area Ratios (LAR).
             - Nc: The optimal pitch delays for each subframe.
             - bc: The quantized gain factors for each subframe.
             - curr_frame_ex_full: The reconstructed excitation signal.
             - curr_frame_st_resd: The reconstructed short-term residual signal.
    """
    # Encode the current frame's short-term residual
    LARc, current_frame_st_resd = RPE_frame_st_coder(s0)

    # Concatenate previous and current frame residuals
    concat_frames_resd = np.concatenate((prev_frame_st_resd, current_frame_st_resd))

    # Find optimal pitch delay (N) and gain factor (b)
    N, b = find_optimal_pitch_gain_subframes(concat_frames_resd)

    # Quantize the factors
    Nc = N
    bc = quantize_b(b)

    # Reconstruct the excitation and short-term residual signals
    curr_frame_ex_full, curr_frame_st_resd = reconstruct_frame_residuals(concat_frames_resd, Nc, bc)

    return LARc, Nc, bc, curr_frame_ex_full, curr_frame_st_resd


# --------------------------------- PART THREE --------------------------------- #
def RPE_frame_coder(s0: np.ndarray, prev_frame_resd: np.ndarray) -> tuple[BitArray, np.ndarray]:
    """
    Encode the current frame and generate the bitstream.

    :param s0: The input signal frame (160 samples).
    :param prev_frame_resd: The short-term residual of the previous frame (160 samples).
    :return: A tuple containing the bitstream (260 bits) and the current frame's short-term residual.
    """
    # Encode the frame using short-term and long-term prediction
    LARc, Nc, bc, curr_frame_ex_full, curr_frame_st_resd = RPE_frame_slt_coder(s0, prev_frame_resd)

    # Compose the bitstream using BitArray
    bitstream = BitArray()

    # Encode LARc (Log-Area Ratios)
    bitstream.append(BitArray(int=LARc[0], length=6))  # LARc[1] - 6 bits (signed)
    bitstream.append(BitArray(int=LARc[1], length=6))  # LARc[2] - 6 bits (signed)
    bitstream.append(BitArray(int=LARc[2], length=5))  # LARc[3] - 5 bits (signed)
    bitstream.append(BitArray(int=LARc[3], length=5))  # LARc[4] - 5 bits (signed)
    bitstream.append(BitArray(int=LARc[4], length=4))  # LARc[5] - 4 bits (signed)
    bitstream.append(BitArray(int=LARc[5], length=4))  # LARc[6] - 4 bits (signed)
    bitstream.append(BitArray(int=LARc[6], length=3))  # LARc[7] - 3 bits (signed)
    bitstream.append(BitArray(int=LARc[7], length=3))  # LARc[8] - 3 bits (signed)

    # Encode subframe parameters
    for i in range(4):  # 4 subframes
        # Nc - 7 bits (LTP lag, unsigned)
        bitstream.append(BitArray(uint=Nc[i], length=7))
        # bc - 2 bits (LTP gain, unsigned)
        bitstream.append(BitArray(uint=bc[i], length=2))
        # xMc (curr_frame_ex_full) - 40 samples per subframe, 3 bits per sample (signed)
        subframe_ex = curr_frame_ex_full[i * 40 : (i + 1) * 40]
        for sample in subframe_ex:
            bitstream.append(BitArray(int=int(sample), length=3))  # Quantize to 3 bits (signed)

    # Validate bitstream length
    #assert len(bitstream) == 260, f"Bitstream length is {len(bitstream)}, expected 260"

    return bitstream, curr_frame_st_resd