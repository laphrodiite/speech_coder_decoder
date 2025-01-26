from encoder import *
from decoder import *
import matplotlib.pyplot as plt
# Generate a random speech frame
s0 = np.random.normal(0, 1, size=160)

# Encode the signal
LARc, curr_frame_st_resd = RPE_frame_st_coder(s0)

# Decode the signal
reconstructed_s0 = RPE_frame_st_decoder(LARc, curr_frame_st_resd)

# Calculate Mean Squared Error (MSE) between original and reconstructed signal
mse = np.mean((s0 - reconstructed_s0) ** 2)
print(np.min(s0), np.max(s0))
print(np.min(reconstructed_s0), np.max(reconstructed_s0))
plt.plot(s0)
plt.show()
plt.plot(reconstructed_s0)
plt.show()
print(f"Reconstruction MSE: {mse}")
assert mse < 1e-3, "Reconstruction error is too high!"
