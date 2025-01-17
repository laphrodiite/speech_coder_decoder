from source.encoder import *

def main():
    # Example Inputs
    # Generate a random array of 160 samples for the input signal (e.g., random noise as placeholder)
    s0 = np.random.normal(0, 1, size=160)

    # Encode the signal
    LARc, curr_frame_st_resd = RPE_frame_st_coder(s0)

    # Expected behavior: LARc should have a length of 8, and residual should have the same length as the input signal
    assert LARc.shape == (8,), "LARc output does not have the expected shape (8,)."
    assert curr_frame_st_resd.shape == s0.shape, "Residual signal shape does not match input signal shape."

    # Output results
    print("Input Signal (first 10 samples):", s0[:10])
    print("Quantized LARc:", LARc)
    print("Residual Signal (first 10 samples):", curr_frame_st_resd[:10])

if __name__ == "__main__":
    main()
