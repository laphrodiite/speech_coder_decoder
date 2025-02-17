from encoder import RPE_frame_slt_coder
from decoder import RPE_frame_slt_decoder
import numpy as np

def main():
    # Generate a dummy input signal (160 samples)
    np.random.seed(42)  # For reproducibility
    s0 = np.random.randn(160)  # Example speech frame

    # Initialize previous frame residual (160 samples)
    prev_frame_st_resd = np.zeros(160)  # Assume no previous residual for the first frame

    # Encode the frame using RPE_frame_slt_coder
    print("Encoding the frame using RPE_frame_slt_coder...")
    LARc, Nc, bc, curr_frame_ex_full, curr_frame_st_resd = RPE_frame_slt_coder(s0, prev_frame_st_resd)
    print("LARc:", LARc)
    print("Nc:", Nc)
    print("bc:", bc)
    print("Current Frame ST Residual (first 10 samples):", curr_frame_st_resd[:10])

    # Decode the frame using RPE_frame_slt_decoder
    print("\nDecoding the frame using RPE_frame_slt_decoder...")
    decoded_s0, decoded_curr_frame_st_resd = RPE_frame_slt_decoder(LARc, Nc, bc,curr_frame_ex_full, prev_frame_st_resd)
    print("Reconstructed Speech Signal (first 10 samples):", decoded_s0[:10])
    print("Reconstructed ST Residual (first 10 samples):", decoded_curr_frame_st_resd[:10])

    # Compare the original and decoded signals
    print("\nComparison Results:")
    print("Original Signal (first 10 samples):", s0[:10])
    print("Decoded Signal (first 10 samples):", decoded_s0[:10])


if __name__ == '__main__':
    main()