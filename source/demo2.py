from source.utlis import load_wav, save_wav
from encoder import RPE_frame_st_coder
from decoder import RPE_frame_st_decoder
import numpy as np

# Input and output file paths
input_wav_path = "../ena_dio_tria.wav"
output_wav_path = "../reconstructed.wav"

# Load input WAV file
s0, framerate = load_wav(input_wav_path)

# Process in frames (160 samples per frame)
frame_size = 160
n_frames = len(s0) // frame_size
reconstructed_audio = []

for i in range(n_frames):
    frame = s0[i * frame_size:(i + 1) * frame_size]

    # Encode and decode the frame
    LARc, curr_frame_st_resd = RPE_frame_st_coder(frame)
    reconstructed_frame = RPE_frame_st_decoder(LARc, curr_frame_st_resd)

    mse = np.mean((reconstructed_frame - frame) ** 2)
    print(f"\tFrame MSE: {mse}")

    # Append reconstructed frame
    reconstructed_audio.extend(reconstructed_frame)

# Convert reconstructed audio to NumPy array
reconstructed_audio = np.array(reconstructed_audio)

# Save reconstructed audio to WAV file
save_wav(output_wav_path, reconstructed_audio, framerate)

# Calculate Mean Squared Error (MSE)
print('-'*40)
mse = np.mean((s0[:len(reconstructed_audio)] - reconstructed_audio) ** 2)
print(f"Reconstruction MSE: {mse}")