from source.decoder import *

def main():
    # Example Inputs
    # Generate a random array of 8 LARc values between -1.5 and 1.5 (example range based on ETSI GSM spec)
    LARc = np.random.uniform(-1.5, 1.5, size=8)

    # Generate a random array of 160 residual values for the current frame
    curr_frame_st_resd = np.random.normal(0, 1, size=160)

    # Decode the signal
    s0 = RPE_frame_st_decoder(LARc, curr_frame_st_resd)

    # Expected behavior: The decoded signal (s0) should have the same shape as the input residual
    assert s0.shape == curr_frame_st_resd.shape, "Output signal shape does not match input residual shape."

    # Output results
    print("Input LARc:", LARc.shape)
    print("Input Residual dimensions:", curr_frame_st_resd.shape)
    print("Reconstructed Signal dimensions:", s0.shape)
    print("MSE: ", np.mean((curr_frame_st_resd - s0)**2))

if __name__ == "__main__":
    main()
