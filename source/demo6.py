from encoder import RPE_frame_slt_coder, RPE_frame_st_coder
import numpy as np

def main():
    # Generate random arrays of 160 speech signal samples for two consecutive frames
    s0 = np.random.normal(0, 1, size=160)
    s1 = np.random.normal(0, 1, size=160)

    # Encode the first frame
    LARc0, prev_frame_st_resd = RPE_frame_st_coder(s0)

    # Encode the second frame using the residual from the first frame
    LARc1, Nc, bc, curr_frame_ex_full, curr_frame_st_resd = RPE_frame_slt_coder(s1, prev_frame_st_resd)

    # Expected behavior: LARc0 and LARc1 should have a shape of (8,), and the residuals should have the same shape as the input
    assert LARc0.shape == (8,), "Output LARc0 shape does not match expected shape (8,)."
    assert LARc1.shape == (8,), "Output LARc1 shape does not match expected shape (8,)."
    assert prev_frame_st_resd.shape == s0.shape, "Residual signal shape from first frame does not match input signal shape."
    assert curr_frame_st_resd.shape == s1.shape, "Residual signal shape from second frame does not match input signal shape."

    # Output results
    print("Input Signal s0 dimensions:", s0.shape)
    print("Input Signal s1 dimensions:", s1.shape)
    print("Quantized and Encoded LARc0 dimensions:", LARc0.shape)
    print("Quantized and Encoded LARc1 dimensions:", LARc1.shape)
    print("Residual Signal from first frame dimensions:", prev_frame_st_resd.shape)
    print("Residual Signal from second frame dimensions:", curr_frame_st_resd.shape)
    print("First 10 LARc0 values:", LARc0[:10])
    print("First 10 LARc1 values:", LARc1[:10])
    print("First 10 residual signal values from first frame:", prev_frame_st_resd[:10])
    print("First 10 residual signal values from second frame:", curr_frame_st_resd[:10])

if __name__ == "__main__":
    main()