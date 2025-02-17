from utils import load_wav, save_wav, plot_waves
import numpy as np
from decoder import RPE_frame_slt_decoder
from encoder import RPE_frame_slt_coder, RPE_frame_st_coder

def encode_decode_pipeline(original_audio, frame_size=160, verbose=False):
    # Split to 160 samples per frame
    n_frames = len(original_audio) // frame_size

    # Initialize audio and mse arrays
    reconstructed_audio, mse_per_frame = [], []

    # Normalize the audio by its range for better results (assuming approximately symmetrical signal)
    audio_range = (np.max(original_audio) - np.min(original_audio)) / 2
    norm_param = audio_range
    original_audio = original_audio / norm_param

    # Encode the first frame separately
    first_frame = original_audio[:frame_size]
    LARc0, prev_frame_st_resd = RPE_frame_st_coder(first_frame)

    # Decode the first frame to get the reconstructed signal
    reconstructed_first_frame, _ = RPE_frame_slt_decoder(LARc0, [0] * 4, [0] * 4, np.zeros(frame_size), np.zeros(frame_size))
    reconstructed_audio.extend(reconstructed_first_frame)

    # Initialize MSE for the first frame
    mse_per_frame.append(np.mean((reconstructed_first_frame - first_frame) ** 2))
    if verbose:
        print(f"\tFrame 0 MSE: {mse_per_frame[-1]}")

    # Loop through the remaining frames to apply the encoding and decoding
    for i in range(1, n_frames):
        frame = original_audio[i * frame_size:(i + 1) * frame_size]
        assert len(frame) == frame_size, f'Frame should have {frame_size} samples, but it was {len(frame)} samples'

        # Encode the frame using RPE with long-term prediction
        LARc, Nc, bc, curr_frame_ex_full, curr_frame_st_resd = RPE_frame_slt_coder(frame, prev_frame_st_resd)

        # Decode the frame using the encoded parameters
        reconstructed_frame, _ = RPE_frame_slt_decoder(LARc, Nc, bc, curr_frame_ex_full, prev_frame_st_resd)

        # Append the reconstructed frame and its MSE
        mse_per_frame.append(np.mean((reconstructed_frame - frame) ** 2))
        if verbose:
            print(f"\tFrame {i} MSE: {mse_per_frame[-1]}")

        reconstructed_audio.extend(reconstructed_frame)

        # Update the previous frame's short-term residual for the next iteration
        prev_frame_st_resd = curr_frame_st_resd

    # De-normalize with the same parameter and clip to avoid outlier values
    reconstructed_audio = np.array(reconstructed_audio) * norm_param
    return np.clip(reconstructed_audio, -audio_range, audio_range), mse_per_frame

def main():
    # Input and output file paths
    input_wav_path = "../ena_dio_tria.wav"
    output_wav_path = "./long_term_reconstructed.wav"

    # Load input WAV file
    s0, framerate = load_wav(input_wav_path)

    # Encode and decode using RPE with long-term prediction
    reconstructed_s0, mses = encode_decode_pipeline(s0)

    # Save reconstructed audio to WAV file
    save_wav(output_wav_path, reconstructed_s0, framerate)

    # Calculate Mean Squared Error (MSE)
    mse = np.mean((s0[:len(reconstructed_s0)] - reconstructed_s0) ** 2)

    # Plot the waves in a common plot
    plot_waves(s0, reconstructed_s0, mses, fig_title='long_term_waves.png')
    print(f"Reconstruction MSE: {int(mse) / (10 ** (len(str(int(mse))) - 1)) : .3f} * 10 ^ {len(str(int(mse))) - 1}")

if __name__ == '__main__':
    main()