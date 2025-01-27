from source.utlis import load_wav, save_wav
import numpy as np
from source.decoder import RPE_frame_st_decoder
from source.encoder import RPE_frame_st_coder
import matplotlib.pyplot as plt

def encode_decode_pipeline(original_audio, frame_size = 160, verbose=False):
    # Split to 160 samples per frame
    n_frames = len(original_audio) // frame_size

    # Initialize audio and mse arrays
    reconstructed_audio, mse_per_frame = [], []

    # Normalize the audio by its range for better results
    audio_range = np.max(original_audio) - np.min(original_audio)
    original_audio = original_audio * 2 / audio_range

    # Loop through all the frames to apply the encoding and decoding
    for i in range(n_frames):
        frame = original_audio[i * frame_size:(i + 1) * frame_size]

        # Encode and decode the frame
        LARc, curr_frame_st_resd = RPE_frame_st_coder(frame)
        reconstructed_frame = RPE_frame_st_decoder(LARc, curr_frame_st_resd)

        mse_per_frame.append(np.mean((reconstructed_frame - frame) ** 2))
        if verbose: print(f"\tFrame MSE: {mse_per_frame[-1]}")

        # Append reconstructed frame
        reconstructed_audio.extend(reconstructed_frame)

    # De-normalize with the same parameter and clip to avoid outlier values
    reconstructed_audio = np.array(reconstructed_audio) * audio_range / 2
    return np.clip(reconstructed_audio, -audio_range, audio_range), mse_per_frame


def plot_waves(s0, reconstructed_s0):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(s0, label="s0", color="blue")
    ax1.set_title("Original Sound Wave")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)

    ax2.plot(reconstructed_s0, label="reconstructed s0", color="orange")
    ax2.set_title("Reconstructed Sound Wave")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True)

    fig.suptitle('RPE Frame Coder & Decoder Results')
    plt.savefig('../plots/1_waves.png')


def main():
    # Input and output file paths
    input_wav_path = "../ena_dio_tria.wav"
    output_wav_path = "../reconstructed.wav"

    # Load input WAV file
    s0, framerate = load_wav(input_wav_path)

    # Encode and decode using RPE
    reconstructed_s0, mses = encode_decode_pipeline(s0)

    # Save reconstructed audio to WAV file
    save_wav(output_wav_path, reconstructed_s0, framerate)

    # Calculate Mean Squared Error (MSE)
    mse = np.mean((s0[:len(reconstructed_s0)] - reconstructed_s0) ** 2)

    # Plot the waves in a common plot
    plot_waves(s0, reconstructed_s0)
    print(f"Reconstruction MSE: {mse}")
    #plt.figure()
    #plt.plot(mses)
    #plt.show()

if __name__ == '__main__':
    main()