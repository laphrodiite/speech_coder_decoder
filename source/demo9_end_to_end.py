import numpy as np
from encoder import RPE_frame_coder
from decoder import RPE_frame_decoder

# Generate a dummy input signal (160 samples)
np.random.seed(42)  # For reproducibility
s0 = np.random.randn(160)  # Example speech frame

# Initialize previous frame residual (160 samples)
prev_frame_resd = np.zeros(160)  # Assume no previous residual for the first frame

# Encode the frame
print("Encoding the frame...")
frame_bit_stream, curr_frame_resd = RPE_frame_coder(s0, prev_frame_resd)
print("Encoded Bitstream:", frame_bit_stream)
print("Current Frame Residual (first 10 samples):", curr_frame_resd[:10])

# Decode the frame
print("\nDecoding the frame...")
decoded_s0, decoded_curr_frame_resd = RPE_frame_decoder(frame_bit_stream, prev_frame_resd)
print("Reconstructed Speech Signal (first 10 samples):", decoded_s0[:10])
print("Reconstructed Short-Term Residual (first 10 samples):", decoded_curr_frame_resd[:10])

# Compare the original and decoded signals
print("\nComparison Results:")
print("Original Signal (first 10 samples):", s0[:10])
print("Decoded Signal (first 10 samples):", decoded_s0[:10])
print("Signal Match:", np.allclose(s0, decoded_s0, atol=1e-5))  # Check if signals match within a tolerance
