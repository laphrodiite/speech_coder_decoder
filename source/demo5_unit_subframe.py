import numpy as np
from utils_encoder import RPE_subframe_slt_lte

# This main checks the RPE_subframe_slt_lte function
def main():
    # Generate test signals
    np.random.seed(42)
    d = np.random.randn(40)  # Random current subframe
    prev_d = np.random.randn(120)  # Random previous subframes

    # Compute pitch period and gain factor
    N, b = RPE_subframe_slt_lte(d, prev_d)

    # Print the results
    print(f"Estimated pitch period (N): {N}")
    print(f"Estimated gain factor (b): {b:.4f}")

if __name__ == '__main__':
    main()
