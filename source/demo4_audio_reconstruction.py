from utils import load_wav, save_wav
import numpy as np
from decoder import RPE_frame_st_decoder
from encoder import RPE_frame_st_coder
from utils import plot_waves

def encode_decode_pipeline(original_audio, frame_size = 160, verbose=False):
    # Split to 160 samples per frame
    n_frames = len(original_audio) // frame_size

    # Initialize audio and mse arrays
    reconstructed_audio, mse_per_frame = [], []

    # Normalize the audio by its range for better results (assuming approximately symmetrical signal)
    audio_range = (np.max(original_audio) - np.min(original_audio)) / 2
    norm_param =  audio_range
    original_audio = original_audio / norm_param

    # Loop through all the frames to apply the encoding and decoding
    for i in range(n_frames):
        frame = original_audio[i * frame_size:(i + 1) * frame_size]
        assert len(frame) == frame_size, f'Frame should have {frame_size} frames, but it was {len(frame)} samples'

        # Encode and decode the frame
        LARc, curr_frame_st_resd = RPE_frame_st_coder(frame)
        reconstructed_frame = RPE_frame_st_decoder(LARc, curr_frame_st_resd)

        # Append the reconstructed frame and its MSE
        mse_per_frame.append(np.mean((reconstructed_frame - frame) ** 2))
        if verbose: print(f"\tFrame MSE: {mse_per_frame[-1]}")
        reconstructed_audio.extend(reconstructed_frame)

    # De-normalize with the same parameter and clip to avoid outlier values
    reconstructed_audio = np.array(reconstructed_audio) * norm_param
    return np.clip(reconstructed_audio, -audio_range, audio_range), mse_per_frame

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
    plot_waves(s0, reconstructed_s0, mses)
    print(f"Reconstruction MSE: {int(mse) / (10 ** (len(str(int(mse))) - 1)) : .3f} * 10 ^ {len(str(int(mse))) - 1}")

if __name__ == '__main__':
    main()