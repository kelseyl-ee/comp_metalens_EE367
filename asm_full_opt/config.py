# Units
MM = 1e-3
UM = 1e-6
NM = 1e-9

# Lens parameters
GRID_N = 459 # grid pixel size 115 and 459
LENS_D = 80 * UM
PIX_SIZE = 350 * NM
WAVL = 532 * NM
EFL = 100 * UM
Z = 100 * UM
HFOV = 15 # degrees

# Field sampling
H_OBJ = 28
W_OBJ = 28
FIELD_STRATEGY = "block"   # "block" or "full"
BLOCK_SIZE = 7            # only used if strategy == "block"
SV = False

# Image construction
PSF_WINDOW_N = 75
TUKEY_ALPHA = 0.01

# Kernels
KERNEL_SIZE = 7
KERNEL_N = 8
