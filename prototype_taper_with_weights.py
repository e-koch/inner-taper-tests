
import numpy as np
import matplotlib.pyplot as plt

from casatools import table


# Path to your measurement set
# ms_path = 'your_measurement_set.ms'  # Change to your MS path

# # Open the measurement set
# tb = table()
# tb.open(ms_path)

# # Read UVW (baseline vectors)
# uvw = tb.getcol('UVW')  # shape: (3, nrows)

# uvw_units = tb.getcolkeyword("UVW", "QuantumUnits")
# print(f"UVW units: {uvw_units}")

# # Compute baseline lengths
# baseline_lengths = np.sqrt(np.sum(uvw**2, axis=0))

# # Plot histogram
# plt.figure(figsize=(8, 5))
# plt.hist(np.log10(baseline_lengths), bins=50, color='skyblue', edgecolor='black')
# plt.xlabel(f'log10 Baseline Length ({uvw_units[0]})')
# plt.ylabel('Count')
# plt.title('Histogram of Baseline Lengths (UVW)')
# plt.tight_layout()
# plt.show()

# tb.close()


# Tests to generate custom weights within the weight column


# Path to your measurement set
this_vis = 'ngVLA_214_ant_60s_noisy-model_innergauss_weights.ms'

# Open the measurement set
tb = table()
tb.open(this_vis, nomodify=False)

# Read UVW (baseline vectors) and current WEIGHT values
# This is a terrible approach that reads everything in for a single
# SPW, single channel, with 2 correlations.
uvw = tb.getcol('UVW')  # shape: (3, nrows)
weights = tb.getcol('WEIGHT')  # shape: (nspw * ncorr, nrows)

baseline_lengths = np.sqrt(np.sum(uvw**2, axis=0))


def inner_gaussian_taper(uvw_arr, taper_size_m=1.e3):
    # Calculate baseline length (meters)
    gauss_arg = (uvw_arr[0]**2 + uvw_arr[1]**2) / (2 * taper_size_m**2)
    # Example: Inverse square weighting
    return 1 - np.exp(-gauss_arg)

def outer_gaussian_taper(uvw_arr, taper_size_m=1.e3):
    # Calculate baseline length (meters)
    gauss_arg = (uvw_arr[0]**2 + uvw_arr[1]**2) / (2 * taper_size_m**2)
    # Example: Inverse square weighting
    return np.exp(-gauss_arg)



taper_size_m = 1.e3  # Example taper size in meters
new_weights = inner_gaussian_taper(uvw, taper_size_m=taper_size_m)

# Tile to match the shape of WEIGHT if necessary
if new_weights.ndim == 1:
    new_weights = np.tile(new_weights[np.newaxis, :], (weights.shape[0], 1))

    
assert new_weights.shape == weights.shape

# Update WEIGHT column
tb.putcol('WEIGHT', new_weights)

tb.close()


tag = 'ppdisk_all'
this_weight_tag = f'{tag}_inneruv_gauss_weights_{taper_size_m}m'

outer_cell = '0.3mas'
outer_cell_val = outer_cell.split('mas')[0] / 1000.0  # Convert to arcseconds

outer_imsize = 2560
outer_thr = '2e-5Jy'

robust     = 2.0
# niter      = 10000
niter      = 0

# Make an image with the new weights
tclean(vis            = this_vis,
       imagename      = this_weight_tag,
       datacolumn     = 'data',
       specmode       = 'mfs',
       weighting      = 'briggs', 
       robust=robust,
       cell           = outer_cell,
       imsize         = outer_imsize,
       niter          = 0,
       threshold      = outer_thr,
       deconvolver    = 'multiscale',
       scales         = [0,5,15,45],
       savemodel      = 'none',
       interactive    = False)


# SUCCESS. Try a range of taper sizes to see how it affects the weights and the resulting images.

arcsec_to_rad = 3600. * 180. / (2 * np.pi)  # Convert radians to arcseconds


this_vis = 'ngVLA_214_ant_60s_noisy-model_innergauss_weights.ms'


tag = 'ppdisk_all'
freq = 90.e9 # 90 GHz
wavelength = 3.e8 / freq  # in meters

# Approx LAS for the pp disk model is ~0.25 arcsec. Consider the taper size
# to ~2x that value.
las_arcsec = 0.25 * 2
las_m = wavelength / (las_arcsec / arcsec_to_rad)

taper_sizes_m = [1.e3, 2.e3, 5.e3, 1.e4]  # Example taper sizes in meters


# Optionally add an outer taper to the weights:
sigma_outer = 0.005 # arcsec
# Convert to meters
outer_taper_size_m = wavelength / (sigma_outer / arcsec_to_rad)  # Convert arcsec to meters

apply_outer_taper = False  # Set to True to apply outer taper

outer_cell = '0.3mas'
outer_cell_val = float(outer_cell.split('mas')[0]) / 1000.0  # Convert to arcseconds

outer_imsize = 8192
outer_thr = '2e-5Jy'

# robust     = 2.0
# niter      = 10000
niter      = 0

# Iterate through robust values
for this_robust in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0,, 1.5, 2.0]:

    for apply_outer_taper in [False, True]:

        for taper_size_m in taper_sizes_m:

            print(f'Processing taper size: {taper_size_m} m')


            this_weight_tag = f'{tag}_robust_{this_robust}_inneruv_gauss_weights_{int(taper_size_m)}m'

            if apply_outer_taper:
                this_weight_tag = f'{this_weight_tag}_outeruv_gauss_weights_{sigma_outer}as'

            tb = table()
            tb.open(this_vis, nomodify=False)

            # Read UVW (baseline vectors) and current WEIGHT values
            # This is a terrible approach that reads everything in for a single
            # SPW, single channel, with 2 correlations.
            uvw = tb.getcol('UVW')  # shape: (3, nrows)
            weights = tb.getcol('WEIGHT')  # shape: (nspw * ncorr, nrows)

            baseline_lengths = np.sqrt(np.sum(uvw**2, axis=0))


            new_weights = inner_gaussian_taper(uvw, taper_size_m=taper_size_m)

            if apply_outer_taper:
                new_weights *= outer_gaussian_taper(uvw, taper_size_m=outer_taper_size_m)

            # Tile to match the shape of WEIGHT if necessary
            if new_weights.ndim == 1:
                new_weights = np.tile(new_weights[np.newaxis, :], (weights.shape[0], 1))

                
            assert new_weights.shape == weights.shape

            # Update WEIGHT column
            tb.putcol('WEIGHT', new_weights)

            tb.close()

            tclean(vis            = this_vis,
                imagename      = this_weight_tag,
                datacolumn     = 'data',
                specmode       = 'mfs',
                weighting      = 'briggs', 
                robust=this_robust,
                cell           = outer_cell,
                imsize         = outer_imsize,
                niter          = 0,
                threshold      = outer_thr,
                deconvolver    = 'multiscale',
                scales         = [0,5,15,45],
                savemodel      = 'none',
                interactive    = False)


# Plot x and y slices through the center of the output PSF images for each taper size
from casatools import image as casaimage

psf_tag = tag  # Use the same tag as above
taper_sizes_m = [1.e3, 2.e3, 5.e3, 1.e4]
psf_images = [f"{psf_tag}_inneruv_gauss_weights_{ts}m.psf" for ts in taper_sizes_m]

center_slices_x = []
center_slices_y = []
labels = []

for img_name, taper_size in zip(psf_images, taper_sizes_m):
    ia = casaimage()
    try:
        ia.open(img_name)
        arr = ia.getchunk()
        ia.close()
    except Exception as e:
        print(f"Could not open image {img_name}: {e}")
        continue
        
    arr2d = arr.squeeze()
    nx, ny = arr2d.shape
    center_x = nx // 2
    center_y = ny // 2
    center_slices_x.append(arr2d[center_x, :])
    center_slices_y.append(arr2d[:, center_y])
    labels.append(f"{int(taper_size)} m")

# Plot all x slices on one subplot, all y slices on another
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for slc, lbl in zip(center_slices_x, labels):
    axs[0].plot(slc, label=lbl)
axs[0].set_title('Central X Slices')
axs[0].set_xlabel('Pixel')
axs[0].set_ylabel('PSF Value')
axs[0].legend()

for slc, lbl in zip(center_slices_y, labels):
    axs[1].plot(slc, label=lbl)
axs[1].set_title('Central Y Slices')
axs[1].set_xlabel('Pixel')
axs[1].set_ylabel('PSF Value')
axs[1].legend()

plt.tight_layout()
plt.show()

plt.savefig(f'{psf_tag}_inneruv_gauss_weights_slices.png', dpi=300, bbox_inches='tight')

# Zoom to the central regions of the slices
plt.xlim([center_x-50, center_x+50])


plt.savefig(f'{psf_tag}_inneruv_gauss_weights_slices_zoom.png', dpi=300, bbox_inches='tight')


def make_radial_profile(arr2d, bin_size=1, max_radius=None):
    nx, ny = arr2d.shape
    center_x = nx // 2
    center_y = ny // 2

    y, x = np.indices((nx, ny))
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Apply max_radius mask before binning
    if max_radius is not None:
        mask = r <= max_radius
        r = r[mask]
        arr2d = arr2d[mask]

    # Bin the radii according to bin_size
    r_bin = (r / bin_size).astype(int)

    # Radial profile: mean value at each bin
    tbin = np.bincount(r_bin.ravel(), arr2d.ravel())
    nr = np.bincount(r_bin.ravel())
    radialprofile = tbin / nr

    radii = np.arange(len(radialprofile)) * bin_size

    return radii, radialprofile


def plot_radial_profile(arr2d, label=None, bin_size=1, max_radius=None):
    radii, radialprofile = make_radial_profile(arr2d,
                                               bin_size=bin_size, 
                                               max_radius=max_radius)

    if label:
        plt.title(f'Radial Profile: {label}')

    plt.plot(radii, radialprofile, label=label, drawstyle='steps-mid')
    plt.xlabel(f'Radius (pixels, bin size={bin_size})')
    plt.ylabel('Mean PSF Value')

    if label:
        plt.legend()
    plt.tight_layout()
    plt.show()
