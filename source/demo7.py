from decoder import RPE_frame_slt_decoder, RPE_frame_st_decoder
import numpy as np

def main():
    # Example Inputs (simulating encoded data)
    LARc0 = np.random.randint(-32, 32, size=8)
    LARc1 = np.random.randint(-32, 32, size=8)
    prev_frame_st_resd = np.random.normal(0, 1, size=160)
    Nc = np.random.randint(0, 120, size=4)  # Random lag for RPE frame
    bc = np.random.randint(0, 4, size=4)    # Random gain codebook index
    curr_frame_ex_full = np.random.normal(0, 1, size=160)

    # Decode the first frame
    s0 = RPE_frame_st_decoder(LARc0, prev_frame_st_resd)

    # Decode the second frame
    s1, curr_frame_st_resd = RPE_frame_slt_decoder(LARc1, list(Nc), list(bc), curr_frame_ex_full, prev_frame_st_resd)

    # Expected behavior: s0 and s1 should both have the shape (160,)
    assert s0.shape == (160,), "Decoded signal s0 shape does not match expected shape (160,)."
    assert s1.shape == (160,), "Decoded signal s1 shape does not match expected shape (160,)."

    # Output results
    print("Decoded Signal s0 dimensions:", s0.shape)
    print("Decoded Signal s1 dimensions:", s1.shape)
    print("First 10 LARc0 values:", LARc0[:10])
    print("First 10 LARc1 values:", LARc1[:10])
    print("First 10 decoded signal values from first frame:", s0[:10])
    print("First 10 decoded signal values from second frame:", s1[:10])

if __name__ == "__main__":
    main()
