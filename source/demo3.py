from encoder import RPE_frame_st_coder
from decoder import RPE_frame_st_decoder
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Generate random speech frames
    n = 4
    mses = []

    for i in range(n):
        # Initialize a signal with varying deviation
        s = np.random.normal(0, 10 ** i, size=160)

        # Encode and decode the signal
        LARc, curr_frame_st_resd = RPE_frame_st_coder(s)
        reconstructed_s = RPE_frame_st_decoder(LARc, curr_frame_st_resd)

        # Calculate Mean Squared Error (MSE) between original and reconstructed signal
        mse = np.mean((s - reconstructed_s) ** 2)
        print('DEVIATION: ', 10**i)
        print(f'\tOriginal range: ({np.min(s)}, {np.max(s)}')
        print(f'\tReconstructed range: ({np.min(reconstructed_s)}, {np.max(reconstructed_s)})')
        print(f'\tReconstruction MSE: {mse}')
        mses.append(mse)

    plt.plot(mses, marker='o')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.title('Short-term Reconstruction MSE for 1 frame for different deviations')
    plt.xlabel('Deviation as a power of 10')
    plt.ylabel('Reconstruction MSE (log scale)')
    plt.savefig('./short_term_mse_deviation.png')

if __name__ == '__main__':
    main()


