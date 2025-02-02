import numpy as np
import matplotlib.pyplot as plt
import wave

# Hard coded quantization parameters described in GSM 06.10 standard
A = [20.0, 20.0, 20.0, 20.0, 13.637, 15.0, 8.334, 8.824]
B = [0.0, 0.0, -5.0, -5.0, 0.184, -0.5, -0.666, -2.235]
min_values = [-32, -32, -16, -16, -8, -8, -4, -4]
max_values = [31, 31, 15, 15, 7, 7, 3, 3]
DLB = [0.2, 0.5, 0.8]
QLB = [0.1, 0.35, 0.65, 1.0]

def decode_LARc_to_reflection(LARc: np.ndarray) -> np.ndarray:
    """ Decode quantized LARc values back into reflection coefficients. """
    LAR = (LARc - np.array(B)) / np.array(A)
    reflection_coefficients = []
    for lar in LAR:
        abs_lar = abs(lar)
        if abs_lar < 0.675:
            reflection_coefficients.append(lar)
        elif 0.675 <= abs_lar < 1.35:
            reflection_coefficients.append(np.sign(lar) * (0.5 * (abs_lar + 0.675)))
        else:  # abs_lar >= 1.35
            reflection_coefficients.append(np.sign(lar) * ((abs_lar + 6.375) / 8))
    return np.array(reflection_coefficients)

def load_wav(file_path):
    """ Load a WAV file and return the audio samples normalized to [0, 1]. """
    with wave.open(file_path, 'r') as wav_file:
        # Read parameters
        n_channels = wav_file.getnchannels()
        framerate = wav_file.getframerate()
        n_frames = wav_file.getnframes()

        # Ensure the WAV file is mono
        assert n_channels == 1, "Only mono audio is supported."

        # Read audio data
        audio_data = np.frombuffer(wav_file.readframes(n_frames), dtype=np.int16)

    return audio_data, framerate

def save_wav(file_path, audio_data, framerate):
    """ Save a NumPy array as a WAV file. """
    # Convert to int16
    audio_data = audio_data.astype(np.int16)

    with wave.open(file_path, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(framerate)
        wav_file.writeframes(audio_data.tobytes())

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
