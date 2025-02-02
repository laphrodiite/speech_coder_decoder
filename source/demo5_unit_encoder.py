from encoder import *
from decoder import *

# Generate a random speech frame
s0 = np.random.normal(0, 1, size=160)
s1 = np.random.normal(0, 1, size=160)

# Encode the signal
LARc0, prev_frame_st_resd = RPE_frame_st_coder(s0)
LARc1, Nc, bc, curr_frame_ex_full, curr_frame_st_resd = RPE_frame_slt_coder(s0, prev_frame_st_resd)

