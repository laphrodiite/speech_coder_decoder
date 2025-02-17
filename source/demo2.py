from encoder import RPE_frame_st_coder
import numpy as np

def main():
    # Example Inputs
    # Generate a random array of 160 speech signal samples for the current frame
    s0 = np.random.normal(0, 1, size=160)

    # Encode the signal
    LARc, curr_frame_st_resd = RPE_frame_st_coder(s0)

    # Expected behavior: LARc should have a shape of (8,) and the residual should have the same shape as the input
    assert LARc.shape == (8,), "Output LARc shape does not match expected shape (8,)."
    assert curr_frame_st_resd.shape == s0.shape, "Residual signal shape does not match input signal shape."

    # Output results
    print("Input Signal dimensions:", s0.shape)
    print("Quantized and Encoded LAR dimensions:", LARc.shape)
    print("Residual Signal dimensions:", curr_frame_st_resd.shape)
    print("First 10 LARc values:", LARc[:10])
    print("First 10 residual signal values:", curr_frame_st_resd[:10])

if __name__ == "__main__":
    main()
