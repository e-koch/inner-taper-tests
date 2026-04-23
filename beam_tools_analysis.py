
import os

from astropy.modeling import models
from scipy.special import erf
from scipy.interpolate import UnivariateSpline

from scipy import ndimage, signal

from astropy.stats import mad_std
import astropy.units as u
from astropy.wcs.utils import proj_plane_pixel_scales

import numpy as np
from spectral_cube import SpectralCube

import matplotlib.pyplot as plt
# plt.ion()

import seaborn as sb
sb.set_context('paper', font_scale=1.2)
sb.set_palette('colorblind')


import re


def extract_psf_params(filename):
    '''
    Extract robust and taper parameters from the PSF filename.
    e.g. "ppdisk_all_robust_1.0_inneruv_gauss_weights_1000m_outeruv_gauss_weights_0.005as.psf"
    '''

    # Extract robust value
    robust_match = re.search(r'robust_(-?[\d\.]+)', filename)
    try:
        robust = float(robust_match.group(1)) if robust_match else None
    except ValueError:
        robust = float(robust_match.group(1)[:-1]) if robust_match else np.nan

    # Extract inner gauss weights value (optional)
    inner_gauss_match = re.search(r'inneruv_gauss_weights_(\d+)m', filename)
    inner_gauss = int(inner_gauss_match.group(1)) if inner_gauss_match else np.nan

    # Extract outer gauss weights value (optional, can be float)
    outer_gauss_match = re.search(r'outeruv_gauss_weights_([\d\.]+)as', filename)
    outer_gauss = float(outer_gauss_match.group(1)) if outer_gauss_match else np.nan

    return robust, inner_gauss, outer_gauss



def rms_measure(img, mask_radius=1000):

    """
    Estimate the RMS noise of an image by masking the central region.

    Parameters
    ----------
    img : 2D array-like
        The input image data.
    mask_radius : int
        The radius (in pixels) of the central region to mask.

    Returns
    -------
    rms : float
        The estimated RMS noise of the image.
    """
    ny, nx = img.shape
    center_x = nx // 2
    center_y = ny // 2

    y, x = np.indices((ny, nx))
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Create a mask for the central region
    mask = r > mask_radius

    # Estimate RMS using MAD
    rms = mad_std(img[mask])

    return rms


fwhm_area_conv = erf(np.sqrt(np.log(2)))
fwhm_factor = 2 * np.sqrt(2 * np.log(2))  # FWHM to sigma conversion factor


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



def psf_gauss_model(fwhm, center=0.0):
    hwhm_gauss = models.Gaussian1D(amplitude=1.0,
                                    mean=center,
                                    stddev=fwhm / fwhm_factor)

    return hwhm_gauss



def kappa(radii, psf_radial, hwhm_gauss):
    '''
    Calculate kappa as the sum of the deviation of the PSF from a Gaussian within the FWHM.

    See Koch+2018 Equation 7.
    '''

    fwhm_mask = (radii <= hwhm_gauss.stddev * fwhm_factor)

    maxval = 1.0 # Always normalized to 1.0 for the PSF.
    fwhm_area = maxval * np.sqrt(2 * np.pi) * hwhm_gauss.stddev * fwhm_area_conv

    kappa = np.sum([(spec - hwhm_gauss(vel)) for spec, vel in
                    zip(psf_radial[fwhm_mask],
                        radii[fwhm_mask])]) / fwhm_area

    return kappa


def extract_psf_slices(psf, center=None):
    """
    Extract the central slices of the PSF along the major and minor axes.
    psf: 2D array-like PSF image
    Returns: (major_axis_slice, minor_axis_slice)
    """
    if center is None:
        ny, nx = psf.shape
        center_x = nx // 2
        center_y = ny // 2
    else:
        center_y, center_x = center

    # Major axis slice (horizontal)
    major_axis_slice = psf[center_y, :]

    # Minor axis slice (vertical)
    minor_axis_slice = psf[:, center_x]

    return major_axis_slice, minor_axis_slice



def skirt_level(radii, psf_radial, fwhm_radius):
    """
    Estimate the PSF value at the FWHM radius using a spline fit.
    radii: array of radii (pixels)
    psf_radial: array of PSF values at each radius
    fwhm_radius: radius (in pixels) corresponding to the FWHM
    """
    # Fit a spline to the radial profile
    spline = UnivariateSpline(radii, psf_radial, s=0)
    # Evaluate at the FWHM radius
    skirt_val = float(spline(fwhm_radius))
    return skirt_val


def read_casa_image(imagename):
    """
    Read a CASA image and return the 2D Projection/Slice with spectral cube.
    """
    if not os.path.exists(imagename):
        raise FileNotFoundError(f"CASA image file {imagename} not found.")
    
    cube = SpectralCube.read(imagename, format='casa_image')

    # Assume we're dealing with continuum images, so we can use the first channel
    if cube.shape[0] > 1:
        raise ValueError("Expected a single-channel PSF image, but found multiple channels.")
    
    plane = cube[0]  # Get the first channel

    return plane



def measure_eta_from_psf_ALMAIMF(psf, max_npix_peak=100):
    '''
    Measure eta from the PSF image.

    Adapted from ALMA-IMF: https://github.com/ALMA-IMF/reduction/blob/e32ec1173354dc171462a664896712e89749b03f/reduction/beam_volume_tools.py

    '''
    if psf.max() <= 0:
        raise ValueError("Invalid PSF")
    center = np.unravel_index(np.argmax(psf), psf.shape)
    cy, cx = center

    beam = psf.beam
    pixels_per_beam = psf.pixels_per_beam

    cutout = psf[cy-max_npix_peak:cy+max_npix_peak+1, cx-max_npix_peak:cx+max_npix_peak+1]
    shape = cutout.shape
    sy, sx = shape
    Y, X = np.mgrid[0:sy, 0:sx]

    center = np.unravel_index(np.argmax(cutout), cutout.shape)
    cy, cx = center

    dy = (Y - cy)
    dx = (X - cx)
    # I guess these definitions already take into account the definition of PA (east from north)?
    costh = np.cos(beam.pa.to('rad'))
    sinth = np.sin(beam.pa.to('rad'))
    # Changed variable name to rminmaj (it was rmajmin)
    rminmaj =  beam.minor / beam.major

    rr = ((dx * costh + dy * sinth)**2 / rminmaj**2 +
          (dx * sinth - dy * costh)**2 / 1**2)**0.5
    rbin = (rr).astype(int)

    #From plots taking the abs looks better centered by ~ 1 pix.
    #radial_mean = ndimage.mean(cutout**2, labels=rbin, index=np.arange(max_npix_peak))
    radial_mean = ndimage.mean(np.abs(cutout), labels=rbin, index=np.arange(max_npix_peak))
    first_min_ind = signal.find_peaks(-radial_mean)[0][0]

    #cutout_posit = np.where(cutout > 0, cutout, 0.)
    radial_sum = ndimage.sum(cutout, labels=rbin, index=np.arange(first_min_ind))
    psf_sum = np.sum(radial_sum)

    clean_psf_sum = pixels_per_beam
    eta = clean_psf_sum/psf_sum
    print(f'clean_psf_sum={clean_psf_sum}, psf_sum={psf_sum}, eta={eta}')

    return eta, clean_psf_sum, psf_sum


def gaussian_eval(params, data, center):
    """
    From MAPS.

    Returns a gaussian with the given parameters.
    """

    width_x, width_y, rotation = params
    rotation = 90-rotation
    width_x = float(width_x)
    width_y = float(width_y)

    rotation = np.deg2rad(rotation)
    x, y = np.indices(data.shape) - center

    xp = x * np.cos(rotation) - y * np.sin(rotation)
    yp = x * np.sin(rotation) + y * np.cos(rotation)
    g = 1.*np.exp(
        -(((xp)/width_x)**2+
          ((yp)/width_y)**2)/2.)
    return g


def measure_eta_from_psf_MAPS(psf, npix_window=201):

    psf_data_raw = psf.value

    delta = (proj_plane_pixel_scales(psf.wcs.celestial)[0] * u.deg).to(u.arcsec).value

    major = psf.beam.major.to(u.arcsec).value
    minor = psf.beam.minor.to(u.arcsec).value
    phi = psf.beam.pa.to(u.deg).value

    print("The CASA fitted beam is " + str(major) + "x" + str(minor) + '" at ' + str(phi) + "deg")

    npix = psf_data_raw.shape[0]         # Assume image is square

    # Check if image cube, or just single psf; this example doesn't handle the full
    # polarization case - implicitly assumes we can drop Stokes
    # If single psf, add an axis so we can use a single loop
    psf_data = np.squeeze(psf_data_raw)
    if len(psf_data.shape) == 2:
        psf_data = np.expand_dims(psf_data, axis=2)

    # Roll the axes to make looping more straightforward
    psf_rolled = np.rollaxis(psf_data,2)

    # Window out the region we want to consider
    i_min = int(npix/2-(npix_window-1)/2)
    i_max = int(npix/2+(npix_window-1)/2 + 1)

    # 2D version
    psf_windowed = psf_rolled[0][i_min:i_max,i_min:i_max]

    # for j in range(10):
    #     psf_windowed = psf_rolled[12*j][i_min:i_max,i_min:i_max]
    #     if np.sum(psf_windowed) > 0:
    #         break

    # Mask out anything beyond the first null
    psf_windowed[psf_windowed<0.] = -1.
    psf_windowed = np.fft.fftshift(psf_windowed)

    for i in range(psf_windowed.shape[0]):
        left_edge = np.argmax(psf_windowed[i] < 0.)
        right_edge = 201-np.argmax(psf_windowed[i][::-1] < 0.)
        psf_windowed[i][left_edge:right_edge] = 0.

    psf_windowed = np.fft.fftshift(psf_windowed)

    # Create a clean beam to evaluate against
    clean_beam = gaussian_eval([major/2.355/delta, minor/2.355/delta, phi], psf_windowed, 
                               (npix_window-1)/2)

    # Calculate epsilon
    epsilon = np.sum(clean_beam)/np.sum(psf_windowed)
    print("Epsilon = " + str(epsilon))

    return epsilon


if __name__ == "__main__":

    print(argh)    

    from astropy.table import Table
    from pathlib import Path
    import astropy.units as u

    from astropy.wcs.utils import proj_plane_pixel_scales

    data_path = Path("/users/ekoch/lustre/inner_taper_tests")

    do_compute_params = False
    do_analysis_plots = True

    max_radius = 100  # Example value, adjust as needed
    bin_size = 1  # Example value, adjust as needed



    if do_compute_params:

        # Load the natural-weighted image to measure the RMS
        imagename_natural = data_path / "ppdisk_all_natural.image"
        img_natural = read_casa_image(imagename_natural)
        rms_natural = rms_measure(img_natural, mask_radius=int(img_natural.shape[0] / 6))
        print(f"Estimated RMS noise of natural image: {rms_natural}")

        # Get all PSF images in the data path
        psf_images = list(data_path.glob("*.psf"))

        print(f"Found {len(psf_images)} PSF images in {data_path}.")

        fwhm_majors = []
        fwhm_minors = []
        fwhm_pixels = []
        skirts = []
        kappas = []
        etas = []
        rms_list = []
        eta_weights = [] # RMS of image / RMS natural

        for psf_filename in psf_images:
            print(f"Processing PSF image: {psf_filename.name}")

            if 'noiseonly' in psf_filename.name:
                print("Skipping noiseonly image.")
                continue

            try:
                psf_img = read_casa_image(psf_filename)
            except Exception as e:
                print(f"Error processing {psf_filename.name}: {e}")
                continue

            # Load the image data from the same imaging set:
            try:
                imagename = psf_filename.with_suffix('.image')
                img = read_casa_image(imagename)
            except Exception as e:
                print(f"Error processing {imagename.name}: {e}")
                continue


            # Estimate the rms in empty parts of the image
            mask_radius = int(img.shape[0] / 6)
            rms = rms_measure(img, mask_radius=mask_radius)
            print(f"Estimated RMS noise: {rms}")

            rms_list.append(rms.value)
            eta_weights.append((rms / rms_natural).to(u.one).value)


            radii, radial_profile = make_radial_profile(psf_img.value,
                                                        bin_size=bin_size, 
                                                        max_radius=max_radius)

            # Convert fwhm to pixels
            pix_scale = proj_plane_pixel_scales(psf_img.wcs.celestial)[0] * u.deg
            fwhm_pixel = (psf_img.beam.major.to(u.deg) / pix_scale).to(u.one).value

            hwhm_gauss = psf_gauss_model(fwhm=fwhm_pixel)

            kappa_value = kappa(radii, radial_profile, hwhm_gauss)

            # try:
            #     eta_value = float(measure_eta_from_psf_ALMAIMF(psf_img)[0])
            # except Exception as e:
            #     print(f"Error calculating eta for {psf_filename.name}: {e}")
            #     eta_value = np.nan

            eta_value = measure_eta_from_psf_MAPS(psf_img)

            skirt_level_value = skirt_level(radii, radial_profile, hwhm_gauss.stddev * fwhm_factor)

            fwhm_majors.append(psf_img.beam.major.to(u.arcsec).value)
            fwhm_minors.append(psf_img.beam.minor.to(u.arcsec).value)
            fwhm_pixels.append(fwhm_pixel)

            kappas.append(kappa_value)
            etas.append(eta_value) 
            skirts.append(skirt_level_value)

        # Split the names based on robust and taper parameters
        robust_values = []
        inner_gauss_values = []
        outer_gauss_values = []

        for psf_filename in psf_images:
            if 'noiseonly' in psf_filename.name:
                print("Skipping noiseonly image.")
                continue

            robust, inner_gauss, outer_gauss = extract_psf_params(psf_filename.name)
            robust_values.append(robust)
            inner_gauss_values.append(inner_gauss)
            outer_gauss_values.append(outer_gauss)

        # Load the theoretical sensitivities from the apparentsens results
        app_sens_vals = np.load("ptsrc_sensitivities.npy")
        rms_theoretical = []
        eta_weight_theoretical = []
        for psf_filename in psf_images:

            if 'noiseonly' in psf_filename.name:
                print("Skipping noiseonly image.")
                continue

            all_tags = app_sens_vals[:, 0]
            idx = np.where(psf_filename.stem == all_tags)[0]

            if len(idx) == 0:
                raise ValueError(f"No matching sensitivity found for {psf_filename.name}")
            elif len(idx) > 1:
                print(f"Multiple matches found for {psf_filename.name}, using the first one.")

            idx = idx[0]
            rms_theoretical.append(float(app_sens_vals[idx, 1]))
            eta_weight_theoretical.append(float(app_sens_vals[idx, 2]))

        # Load the noise measured from the noise-only images
        modelfree_sens_vals = np.load("ptsrc_sensitivities_modelfree.npy")
        rms_modelfree = []
        eta_weight_modelfree = []
        for psf_filename in psf_images:

            if 'noiseonly' in psf_filename.name:
                print("Skipping noiseonly image.")
                continue

            all_tags = modelfree_sens_vals[:, 0]
            idx = np.where(psf_filename.stem == all_tags)[0]

            if len(idx) == 0:
                raise ValueError(f"No matching sensitivity found for {psf_filename.name}")
            elif len(idx) > 1:
                print(f"Multiple matches found for {psf_filename.name}, using the first one.")

            idx = idx[0]
            rms_modelfree.append(float(modelfree_sens_vals[idx, 1]))
            eta_weight_modelfree.append(float(modelfree_sens_vals[idx, 2]))

        # Construct a table with the results
        results_table = Table({
            'psf_imagename': [psf.name for psf in psf_images if 'noiseonly' not in psf.name],
            'robust': robust_values,
            'inner_gauss_m': inner_gauss_values,
            'outer_gauss_as': outer_gauss_values,
            'fwhm_major_arcsec': fwhm_majors,
            'fwhm_minor_arcsec': fwhm_minors,
            'fwhm_major_pix': fwhm_pixels,
            'kappa': kappas,
            'eta': etas,
            'skirt_at_fwhm': skirts,
            'rms_measured': rms_list,
            'eta_weight_measured': eta_weights,
            'rms_theoretical': rms_theoretical,
            'eta_weight_theoretical': eta_weight_theoretical,
            'rms_modelfree': rms_modelfree,
            'eta_weight_modelfree': eta_weight_modelfree,
            })

        results_table.write(data_path / 'psf_inner_taper_analysis_results.csv', overwrite=True)


    if do_analysis_plots:
        # Analysis: move to a notebook

        results_table = Table.read(data_path / 'psf_inner_taper_analysis_results.csv')

        no_taper_mask = np.logical_and(np.isnan(results_table['inner_gauss_m']), 
                                    np.isfinite(results_table['robust'].astype(float)))
        all_taper_mask = np.isfinite(results_table['inner_gauss_m'])

        outertaper_mask = np.logical_and(np.isfinite(results_table['outer_gauss_as']), all_taper_mask)
        innertaper_mask = np.logical_and(np.isnan(results_table['outer_gauss_as']), all_taper_mask)

        results_notaper = results_table[no_taper_mask]
        results_innertaper = results_table[innertaper_mask]
        results_outertaper = results_table[outertaper_mask]

        # Plot eta_weight_theoretical vs robust

        _ = plt.scatter(results_notaper['robust'], 
                        results_notaper['eta_weight_theoretical'], 
                        label='No taper', marker='o')

        # Grab a single inner taper size to plot
        inner_taper_size = np.unique(results_innertaper['inner_gauss_m'])[0]
        _ = plt.scatter(results_innertaper['robust'][results_innertaper['inner_gauss_m'] == inner_taper_size], 
                        results_innertaper['eta_weight_theoretical'][results_innertaper['inner_gauss_m'] == inner_taper_size], 
                        label=f'Inner taper {inner_taper_size} m', marker='D')

        # And with the outer taper also applied:
        _ = plt.scatter(results_outertaper['robust'][results_outertaper['inner_gauss_m'] == inner_taper_size], 
                        results_outertaper['eta_weight_theoretical'][results_outertaper['inner_gauss_m'] == inner_taper_size], 
                        label=f'Inner {inner_taper_size} m + Outer 0.005 as', marker='s')
        
        plt.legend()

        plt.xlabel('Robust')
        plt.ylabel('eta Weight (Theoretical)')

        plt.savefig(data_path / f'eta_weight_vs_robust_theoretical_inner_taper_{inner_taper_size}m.png', dpi=200)

        # Zoom in on the y axis:
        plt.ylim(0.8, 1.8)
        plt.savefig(data_path / f'eta_weight_vs_robust_theoretical_inner_taper_{inner_taper_size}m_zoom.png', dpi=200)


        plt.close()

        #### NEXT PLOT

        fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=False, sharex=True)

        # Iterate through inner taper sizes
        inner_taper_sizes = np.unique(results_innertaper['inner_gauss_m'])
        for taper_size in inner_taper_sizes:
            taper_mask = results_innertaper['inner_gauss_m'] == taper_size

            # Order robust values
            robusts = results_innertaper['robust'][taper_mask]
            sort_idx = np.argsort(robusts)

            _ = axs[0].plot(robusts[sort_idx], 
                        results_innertaper['eta_weight_theoretical'][taper_mask][sort_idx], 
                        label=f'Inner taper {taper_size} m', 
                        marker='o')

        axs[0].legend()
        axs[0].set_xlabel('Robust')
        axs[0].set_ylabel('eta Weight (Theoretical)')

        # And with outer taper
        for taper_size in inner_taper_sizes:
            taper_mask = results_outertaper['inner_gauss_m'] == taper_size

            # Order robust values
            robusts = results_outertaper['robust'][taper_mask]
            sort_idx = np.argsort(robusts)

            _ = axs[1].plot(robusts[sort_idx], 
                        results_outertaper['eta_weight_theoretical'][taper_mask][sort_idx], 
                        label=f'Inner taper {taper_size} m', 
                        marker='o')

        axs[1].legend()
        axs[1].set_xlabel('Robust')

        axs[0].set_title('Inner Taper Only')
        axs[1].set_title('Inner + Outer 0.005 as Taper')

        plt.savefig(data_path / f'eta_weight_vs_robust_theoretical.png', dpi=200)
        plt.close()

        #######################
        # Plot eta_weight_theoretical vs fwhm_major_arcsec

        _ = plt.scatter(results_notaper['fwhm_major_arcsec'], 
                        results_notaper['eta_weight_theoretical'], 
                        label='No taper', marker='o')

        # Grab a single inner taper size to plot
        inner_taper_size = np.unique(results_innertaper['inner_gauss_m'])[0]
        _ = plt.scatter(results_innertaper['fwhm_major_arcsec'][results_innertaper['inner_gauss_m'] == inner_taper_size], 
                        results_innertaper['eta_weight_theoretical'][results_innertaper['inner_gauss_m'] == inner_taper_size], 
                        label=f'Inner taper {inner_taper_size} m', marker='D')

        # And with the outer taper also applied:
        _ = plt.scatter(results_outertaper['fwhm_major_arcsec'][results_outertaper['inner_gauss_m'] == inner_taper_size], 
                        results_outertaper['eta_weight_theoretical'][results_outertaper['inner_gauss_m'] == inner_taper_size], 
                        label=f'Inner {inner_taper_size} m + Outer 0.005 as', marker='s')

        plt.legend()
        plt.xlabel('FWHM Major (arcsec)')
        plt.ylabel('eta Weight (Theoretical)')

        # Targeted 5 mas resolution for science goal:
        plt.savefig(data_path / f'eta_weight_vs_fwhm_major_theoretical_inner_taper_{inner_taper_size}m.png', dpi=200)

        # Zoom in on the y axis:
        plt.ylim(0.8, 1.8)
        plt.savefig(data_path / f'eta_weight_vs_fwhm_major_theoretical_inner_taper_{inner_taper_size}m_zoom.png', dpi=200)
        plt.close()

        #### NEXT PLOT

        fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=False, sharex=True)

        # Iterate through inner taper sizes
        inner_taper_sizes = np.unique(results_innertaper['inner_gauss_m'])
        for taper_size in inner_taper_sizes:
            taper_mask = results_innertaper['inner_gauss_m'] == taper_size

            # Order fwhm_major values
            fwhm_majors = results_innertaper['fwhm_major_arcsec'][taper_mask]
            sort_idx = np.argsort(fwhm_majors)

            _ = axs[0].plot(fwhm_majors[sort_idx], 
                        results_innertaper['eta_weight_theoretical'][taper_mask][sort_idx], 
                        label=f'Inner taper {taper_size} m', 
                        marker='o')

        axs[0].legend()
        axs[0].set_xlabel('FWHM Major (arcsec)')
        axs[0].set_ylabel('eta Weight (Theoretical)')

        # And with outer taper
        for taper_size in inner_taper_sizes:
            taper_mask = results_outertaper['inner_gauss_m'] == taper_size

            # Order fwhm_major values
            fwhm_majors = results_outertaper['fwhm_major_arcsec'][taper_mask]
            sort_idx = np.argsort(fwhm_majors)

            _ = axs[1].plot(fwhm_majors[sort_idx], 
                        results_outertaper['eta_weight_theoretical'][taper_mask][sort_idx], 
                        label=f'Inner taper {taper_size} m', 
                        marker='o')

        axs[1].legend()
        axs[1].set_xlabel('FWHM Major (arcsec)')

        axs[0].set_title('Inner Taper Only')
        axs[1].set_title('Inner + Outer 0.005 as Taper')

        # Targeted 5 mas resolution for science goal:
        axs[0].axvline(0.005, color='gray', linestyle='--')
        axs[1].axvline(0.005, color='gray', linestyle='--')


        plt.savefig(data_path / f'eta_weight_vs_fwhm_major_theoretical.png', dpi=200)
        plt.close()


        #############
        #############
        # Now compare skirt levels
        #############
        #############


        # Plot eta_weight_modelfree vs robust

        _ = plt.scatter(results_notaper['robust'], 
                        results_notaper['skirt_at_fwhm'], 
                        label='No taper', marker='o')

        # Grab a single inner taper size to plot
        inner_taper_size = np.unique(results_innertaper['inner_gauss_m'])[0]
        _ = plt.scatter(results_innertaper['robust'][results_innertaper['inner_gauss_m'] == inner_taper_size], 
                        results_innertaper['skirt_at_fwhm'][results_innertaper['inner_gauss_m'] == inner_taper_size], 
                        label=f'Inner taper {inner_taper_size} m', marker='D')

        # And with the outer taper also applied:
        _ = plt.scatter(results_outertaper['robust'][results_outertaper['inner_gauss_m'] == inner_taper_size], 
                        results_outertaper['skirt_at_fwhm'][results_outertaper['inner_gauss_m'] == inner_taper_size], 
                        label=f'Inner {inner_taper_size} m + Outer 0.005 as', marker='s')
        
        plt.legend()

        plt.xlabel('Robust')
        plt.ylabel('Skirt at FWHM')

        plt.savefig(data_path / f'skirt_vs_robust_inner_taper_{inner_taper_size}m.png', dpi=200)

        plt.close()

        #### NEXT PLOT

        fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True, sharex=True)

        # Plot the no taper case in gray for reference
            # Order robust values
        robusts_notaper = results_notaper['robust']
        sort_idx = np.argsort(robusts_notaper)

        for ax in axs:
            _ = ax.plot(robusts_notaper[sort_idx], 
                        results_notaper['skirt_at_fwhm'][sort_idx], 
                        label=f'No taper',
                        zorder=10,
                        color='gray', alpha=0.5, linestyle='-',
                        linewidth=3)
            
        # Iterate through inner taper sizes
        inner_taper_sizes = np.unique(results_innertaper['inner_gauss_m'])
        markers = ['o', 's', 'D', '^', 'v', 'P', '*']
        for ii, taper_size in enumerate(inner_taper_sizes):
            taper_mask = results_innertaper['inner_gauss_m'] == taper_size

            # Order robust values
            robusts = results_innertaper['robust'][taper_mask]
            sort_idx = np.argsort(robusts)

            _ = axs[0].plot(robusts[sort_idx], 
                        results_innertaper['skirt_at_fwhm'][taper_mask][sort_idx], 
                        label=f'Inner taper {taper_size} m', 
                        marker=markers[ii])

        axs[0].legend()
        axs[0].set_xlabel('Robust')
        axs[0].set_ylabel('Skirt at FWHM')

        # And with outer taper
        for ii, taper_size in enumerate(inner_taper_sizes):
            taper_mask = results_outertaper['inner_gauss_m'] == taper_size

            # Order robust values
            robusts = results_outertaper['robust'][taper_mask]
            sort_idx = np.argsort(robusts)

            _ = axs[1].plot(robusts[sort_idx], 
                        results_outertaper['skirt_at_fwhm'][taper_mask][sort_idx], 
                        label=f'Inner taper {taper_size} m', 
                        marker=markers[ii])

        # Overlay the Gaussian skirt level for reference
        # Add label above the line
        for ax in axs:
            ax.axhline(np.exp(-4 * np.log(2)), color='black', linestyle='--')
            ax.text(1., np.exp(-4 * np.log(2)) - 0.015, 
                    'Gaussian Limit', color='black', fontsize=12, 
                    ha='center', va='top', fontweight='bold')


        # axs[1].legend()
        axs[1].set_xlabel('Robust')

        axs[0].set_title('Inner Taper Only')
        axs[1].set_title('Inner + Outer 0.005 arcsec Taper')

        for ax in axs:
            ax.grid(True, which='both', linestyle='--', alpha=0.5)

        axs[0].set_ylim(0, 0.5)

        plt.savefig(data_path / f'skirt_vs_robust.png', dpi=200)
        plt.close()

        #######################
        # Plot skirt_at_fwhm vs fwhm_major_arcsec

        _ = plt.scatter(results_notaper['fwhm_major_arcsec'], 
                        results_notaper['skirt_at_fwhm'], 
                        label='No taper', marker='o', s=50)

        # Grab a single inner taper size to plot
        inner_taper_size = np.unique(results_innertaper['inner_gauss_m'])[0]
        _ = plt.scatter(results_innertaper['fwhm_major_arcsec'][results_innertaper['inner_gauss_m'] == inner_taper_size], 
                        results_innertaper['skirt_at_fwhm'][results_innertaper['inner_gauss_m'] == inner_taper_size], 
                        label=f'Inner taper {inner_taper_size} m', marker='D', s=50)

        # And with the outer taper also applied:
        _ = plt.scatter(results_outertaper['fwhm_major_arcsec'][results_outertaper['inner_gauss_m'] == inner_taper_size], 
                        results_outertaper['skirt_at_fwhm'][results_outertaper['inner_gauss_m'] == inner_taper_size], 
                        label=f'Inner {inner_taper_size} m + Outer 0.005 as', marker='s', s=50)

        plt.legend()
        plt.xlabel('FWHM Major (arcsec)')
        plt.ylabel('Skirt at FWHM')

        # Targeted 5 mas resolution for science goal:
        target_res = 0.005
        plt.axvline(target_res, color='gray', linestyle='--')


        # Find and label the closest point for each taper case
        for results, label, marker, color in [
            (results_notaper, 'No taper', 'o', sb.color_palette()[0]),
            (results_innertaper[results_innertaper['inner_gauss_m'] == inner_taper_size], f'Inner {inner_taper_size} m', 'D', sb.color_palette()[1]),
            (results_outertaper[results_outertaper['inner_gauss_m'] == inner_taper_size], f'Inner {inner_taper_size} m + Outer 0.005 as', 's', sb.color_palette()[2])
        ]:
            fwhm_vals = results['fwhm_major_arcsec']
            idx = np.argmin(np.abs(fwhm_vals - target_res))
            x = fwhm_vals[idx]
            y = results['skirt_at_fwhm'][idx]
            robust_val = results['robust'][idx]
            plt.text(x, y + 0.015, f'robust={robust_val:.2f}', 
                     color=color, fontsize=10, 
                     ha='right', va='bottom')


        plt.axhline(np.exp(-4 * np.log(2)), color='black', linestyle='--')
        plt.text(1., np.exp(-4 * np.log(2)) - 0.015, 
                'Gaussian Limit', color='black', fontsize=12, 
                ha='center', va='top', fontweight='bold')


        plt.savefig(data_path / f'skirt_vs_fwhm_major_modelfree_inner_taper_{inner_taper_size}m.png', dpi=200)
        plt.close()

        #### NEXT PLOT

        fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=False, sharex=True)

        # Iterate through inner taper sizes
        inner_taper_sizes = np.unique(results_innertaper['inner_gauss_m'])
        for taper_size in inner_taper_sizes:
            taper_mask = results_innertaper['inner_gauss_m'] == taper_size

            # Order fwhm_major values
            fwhm_majors = results_innertaper['fwhm_major_arcsec'][taper_mask]
            sort_idx = np.argsort(fwhm_majors)

            _ = axs[0].plot(fwhm_majors[sort_idx], 
                        results_innertaper['skirt_at_fwhm'][taper_mask][sort_idx], 
                        label=f'Inner taper {taper_size} m', 
                        marker='o')

        axs[0].legend()
        axs[0].set_xlabel('FWHM Major (arcsec)')
        axs[0].set_ylabel('Skirt at FWHM')

        # And with outer taper
        for taper_size in inner_taper_sizes:
            taper_mask = results_outertaper['inner_gauss_m'] == taper_size

            # Order fwhm_major values
            fwhm_majors = results_outertaper['fwhm_major_arcsec'][taper_mask]
            sort_idx = np.argsort(fwhm_majors)

            _ = axs[1].plot(fwhm_majors[sort_idx], 
                        results_outertaper['skirt_at_fwhm'][taper_mask][sort_idx], 
                        label=f'Inner taper {taper_size} m', 
                        marker='o')

        axs[1].legend()
        axs[1].set_xlabel('FWHM Major (arcsec)')

        axs[0].set_title('Inner Taper Only')
        axs[1].set_title('Inner + Outer 0.005 as Taper')

        # Targeted 5 mas resolution for science goal:
        axs[0].axvline(0.005, color='gray', linestyle='--')
        axs[1].axvline(0.005, color='gray', linestyle='--')


        plt.savefig(data_path / f'skirt_vs_fwhm_major_modelfree.png', dpi=200)
        plt.close()


        #######################
        # Plot kappa vs fwhm_major_arcsec

        _ = plt.scatter(results_notaper['fwhm_major_arcsec'], 
                        results_notaper['kappa'], 
                        label='No taper', marker='o', s=50)

        # Grab a single inner taper size to plot
        inner_taper_size = np.unique(results_innertaper['inner_gauss_m'])[0]
        _ = plt.scatter(results_innertaper['fwhm_major_arcsec'][results_innertaper['inner_gauss_m'] == inner_taper_size], 
                        results_innertaper['kappa'][results_innertaper['inner_gauss_m'] == inner_taper_size], 
                        label=f'Inner taper {inner_taper_size} m', marker='D', s=50)

        # And with the outer taper also applied:
        _ = plt.scatter(results_outertaper['fwhm_major_arcsec'][results_outertaper['inner_gauss_m'] == inner_taper_size], 
                        results_outertaper['kappa'][results_outertaper['inner_gauss_m'] == inner_taper_size], 
                        label=f'Inner {inner_taper_size} m + Outer 0.005 as', marker='s', s=50)

        plt.legend()
        plt.xlabel('FWHM Major (arcsec)')
        plt.ylabel(r'$\kappa$')

        # Targeted 5 mas resolution for science goal:
        target_res = 0.005
        plt.axvline(target_res, color='gray', linestyle='--')

        plt.axhline(0.0, color='black', linestyle='--')
        plt.text(0.02, 0.005, 
                'Gaussian', color='black', fontsize=12, 
                ha='center', va='bottom', fontweight='bold')

        plt.savefig(data_path / f'kappa_vs_fwhm_major_inner_taper_{inner_taper_size}m.png', dpi=200)
        plt.close()

        #### NEXT PLOT

        fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=False, sharex=True)

        # Iterate through inner taper sizes
        inner_taper_sizes = np.unique(results_innertaper['inner_gauss_m'])
        for taper_size in inner_taper_sizes:
            taper_mask = results_innertaper['inner_gauss_m'] == taper_size

            # Order fwhm_major values
            fwhm_majors = results_innertaper['fwhm_major_arcsec'][taper_mask]
            sort_idx = np.argsort(fwhm_majors)

            _ = axs[0].plot(fwhm_majors[sort_idx], 
                        results_innertaper['kappa'][taper_mask][sort_idx], 
                        label=f'Inner taper {taper_size} m', 
                        marker='o')

        # axs[0].legend()
        axs[0].set_xlabel('FWHM Major (arcsec)')
        axs[0].set_ylabel(r'$\kappa$')

        # And with outer taper
        for taper_size in inner_taper_sizes:
            taper_mask = results_outertaper['inner_gauss_m'] == taper_size

            # Order fwhm_major values
            fwhm_majors = results_outertaper['fwhm_major_arcsec'][taper_mask]
            sort_idx = np.argsort(fwhm_majors)

            _ = axs[1].plot(fwhm_majors[sort_idx], 
                        results_outertaper['kappa'][taper_mask][sort_idx], 
                        label=f'Inner taper {taper_size} m', 
                        marker='o')

        axs[1].legend()
        axs[1].set_xlabel('FWHM Major (arcsec)')

        axs[0].set_title('Inner Taper Only')
        axs[1].set_title('Inner + Outer 0.005 as Taper')

        # Targeted 5 mas resolution for science goal:
        axs[0].axvline(0.005, color='gray', linestyle='--')
        axs[1].axvline(0.005, color='gray', linestyle='--')

        for ax in axs:
            ax.axhline(0.0, color='black', linestyle='--')
            ax.text(0.02, 0.005, 
                    'Gaussian', color='black', fontsize=12, 
                    ha='center', va='bottom', fontweight='bold')


        plt.savefig(data_path / f'kappa_vs_fwhm_major_modelfree.png', dpi=200)
        plt.close()


        #############
        #############
        # Now using more realistic model-free estimates
        #############
        #############


        # Plot eta_weight_modelfree vs robust

        _ = plt.scatter(results_notaper['robust'], 
                        results_notaper['eta_weight_modelfree'], 
                        label='No taper', marker='o')

        # Grab a single inner taper size to plot
        inner_taper_size = np.unique(results_innertaper['inner_gauss_m'])[0]
        _ = plt.scatter(results_innertaper['robust'][results_innertaper['inner_gauss_m'] == inner_taper_size], 
                        results_innertaper['eta_weight_modelfree'][results_innertaper['inner_gauss_m'] == inner_taper_size], 
                        label=f'Inner taper {inner_taper_size} m', marker='D')

        # And with the outer taper also applied:
        _ = plt.scatter(results_outertaper['robust'][results_outertaper['inner_gauss_m'] == inner_taper_size], 
                        results_outertaper['eta_weight_modelfree'][results_outertaper['inner_gauss_m'] == inner_taper_size], 
                        label=f'Inner {inner_taper_size} m + Outer 0.005 as', marker='s')
        
        plt.legend()

        plt.xlabel('Robust')
        plt.ylabel(r'$\eta_{\mathrm{weight}} = \sigma / \sigma_{\mathrm{NA}}$ (Model-free)')

        plt.savefig(data_path / f'eta_weight_vs_robust_modelfree_inner_taper_{inner_taper_size}m.png', dpi=200)

        plt.close()

        #### NEXT PLOT

        fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True, sharex=True)

        # Plot the no taper case in gray for reference
            # Order robust values
        robusts_notaper = results_notaper['robust']
        sort_idx = np.argsort(robusts_notaper)

        for ax in axs:
            _ = ax.plot(robusts_notaper[sort_idx], 
                        results_notaper['eta_weight_modelfree'][sort_idx], 
                        label=f'No taper',
                        zorder=10,
                        color='gray', alpha=0.5, linestyle='-',
                        linewidth=3)
            
        # Iterate through inner taper sizes
        inner_taper_sizes = np.unique(results_innertaper['inner_gauss_m'])
        markers = ['o', 's', 'D', '^', 'v', 'P', '*']
        for ii, taper_size in enumerate(inner_taper_sizes):
            taper_mask = results_innertaper['inner_gauss_m'] == taper_size

            # Order robust values
            robusts = results_innertaper['robust'][taper_mask]
            sort_idx = np.argsort(robusts)

            _ = axs[0].plot(robusts[sort_idx], 
                        results_innertaper['eta_weight_modelfree'][taper_mask][sort_idx], 
                        label=f'Inner taper {taper_size} m', 
                        marker=markers[ii])

        axs[0].legend()
        axs[0].set_xlabel('Robust')
        axs[0].set_ylabel(r'$\eta_{\mathrm{weight}} = \sigma / \sigma_{\mathrm{NA}}$ (Model-free)')

        # And with outer taper
        for ii, taper_size in enumerate(inner_taper_sizes):
            taper_mask = results_outertaper['inner_gauss_m'] == taper_size

            # Order robust values
            robusts = results_outertaper['robust'][taper_mask]
            sort_idx = np.argsort(robusts)

            _ = axs[1].plot(robusts[sort_idx], 
                        results_outertaper['eta_weight_modelfree'][taper_mask][sort_idx], 
                        label=f'Inner taper {taper_size} m', 
                        marker=markers[ii])

        # axs[1].legend()
        axs[1].set_xlabel('Robust')

        axs[0].set_title('Inner Taper Only')
        axs[1].set_title('Inner + Outer 0.005 arcsec Taper')

        for ax in axs:
            ax.grid(True, which='both', linestyle='--', alpha=0.5)

        plt.savefig(data_path / f'eta_weight_vs_robust_modelfree.png', dpi=200)
        plt.close()

        #######################
        # Plot eta_weight_modelfree vs fwhm_major_arcsec

        _ = plt.scatter(results_notaper['fwhm_major_arcsec'], 
                        results_notaper['eta_weight_modelfree'], 
                        label='No taper', marker='o', s=50)

        # Grab a single inner taper size to plot
        inner_taper_size = np.unique(results_innertaper['inner_gauss_m'])[0]
        _ = plt.scatter(results_innertaper['fwhm_major_arcsec'][results_innertaper['inner_gauss_m'] == inner_taper_size], 
                        results_innertaper['eta_weight_modelfree'][results_innertaper['inner_gauss_m'] == inner_taper_size], 
                        label=f'Inner taper {inner_taper_size} m', marker='D', s=50)

        # And with the outer taper also applied:
        _ = plt.scatter(results_outertaper['fwhm_major_arcsec'][results_outertaper['inner_gauss_m'] == inner_taper_size], 
                        results_outertaper['eta_weight_modelfree'][results_outertaper['inner_gauss_m'] == inner_taper_size], 
                        label=f'Inner {inner_taper_size} m + Outer 0.005 as', marker='s', s=50)

        plt.legend()
        plt.xlabel('FWHM Major (arcsec)')
        plt.ylabel(r'$\eta_{\mathrm{weight}} = \sigma / \sigma_{\mathrm{NA}}$ (Model-free)')

        # Targeted 5 mas resolution for science goal:
        target_res = 0.005
        plt.axvline(target_res, color='gray', linestyle='--')


        # Find and label the closest point for each taper case
        colors = list(sb.color_palette()[:2])
        for results, label, marker, color in [
            (results_notaper, 'No taper', 'o', sb.color_palette()[0]),
            (results_innertaper[results_innertaper['inner_gauss_m'] == inner_taper_size], f'Inner {inner_taper_size} m', 'D', sb.color_palette()[1]),
            (results_outertaper[results_outertaper['inner_gauss_m'] == inner_taper_size], f'Inner {inner_taper_size} m + Outer 0.005 as', 's', sb.color_palette()[2])
        ]:
            fwhm_vals = results['fwhm_major_arcsec']
            idx = np.argmin(np.abs(fwhm_vals - target_res))
            x = fwhm_vals[idx]
            y = results['eta_weight_modelfree'][idx]
            robust_val = results['robust'][idx]
            plt.text(x, y + 0.05, f'robust={robust_val:.2f}', 
                     color=color, fontsize=10, 
                     ha='left', va='bottom')


        plt.savefig(data_path / f'eta_weight_vs_fwhm_major_modelfree_inner_taper_{inner_taper_size}m.png', dpi=200)

        # Zoom in on the y axis:
        plt.ylim(0.8, 1.8)
        plt.savefig(data_path / f'eta_weight_vs_fwhm_major_modelfree_inner_taper_{inner_taper_size}m_zoom.png', dpi=200)
        plt.close()

        #### NEXT PLOT

        fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=False, sharex=True)

        # Iterate through inner taper sizes
        inner_taper_sizes = np.unique(results_innertaper['inner_gauss_m'])
        for taper_size in inner_taper_sizes:
            taper_mask = results_innertaper['inner_gauss_m'] == taper_size

            # Order fwhm_major values
            fwhm_majors = results_innertaper['fwhm_major_arcsec'][taper_mask]
            sort_idx = np.argsort(fwhm_majors)

            _ = axs[0].plot(fwhm_majors[sort_idx], 
                        results_innertaper['eta_weight_modelfree'][taper_mask][sort_idx], 
                        label=f'Inner taper {taper_size} m', 
                        marker='o')

        axs[0].legend()
        axs[0].set_xlabel('FWHM Major (arcsec)')
        axs[0].set_ylabel(r'$\eta_{\mathrm{weight}} = \sigma / \sigma_{\mathrm{NA}}$ (Model-free)')

        # And with outer taper
        for taper_size in inner_taper_sizes:
            taper_mask = results_outertaper['inner_gauss_m'] == taper_size

            # Order fwhm_major values
            fwhm_majors = results_outertaper['fwhm_major_arcsec'][taper_mask]
            sort_idx = np.argsort(fwhm_majors)

            _ = axs[1].plot(fwhm_majors[sort_idx], 
                        results_outertaper['eta_weight_modelfree'][taper_mask][sort_idx], 
                        label=f'Inner taper {taper_size} m', 
                        marker='o')

        axs[1].legend()
        axs[1].set_xlabel('FWHM Major (arcsec)')

        axs[0].set_title('Inner Taper Only')
        axs[1].set_title('Inner + Outer 0.005 as Taper')

        # Targeted 5 mas resolution for science goal:
        axs[0].axvline(0.005, color='gray', linestyle='--')
        axs[1].axvline(0.005, color='gray', linestyle='--')


        plt.savefig(data_path / f'eta_weight_vs_fwhm_major_modelfree.png', dpi=200)
        plt.close()






        #############
        #############
        # Make a gallery of the PSFs and dirty images:
        #############
        #############

        # Assuming results_table is already loaded
        unique_robusts = np.unique(results_table['robust'])
        for robust_val in unique_robusts:
            # Select rows with this robust value
            mask = results_table['robust'] == robust_val
            selected = results_table[mask]
            if len(selected) == 0:
                continue

            fig, axes = plt.subplots(3, 3, 
                                     figsize=(12, 12), 
                                     squeeze=False)
            axes = axes.ravel()  # 1D array

            # Set the order of rows: no taper, inner taper, outer taper
            # No taper has nan for both inner and outer gauss
            no_taper_idx = np.where(np.isnan(selected['inner_gauss_m']) & 
                                   np.isnan(selected['outer_gauss_as']))[0]
            inner_taper_idx = np.where(np.isfinite(selected['inner_gauss_m']) & 
                                      np.isnan(selected['outer_gauss_as']))[0]
            outer_taper_idx = np.where(np.isfinite(selected['inner_gauss_m']) & 
                                      np.isfinite(selected['outer_gauss_as']))[0]

            # Sort by taper size within each group
            inner_taper_idx = inner_taper_idx[np.argsort(selected['inner_gauss_m'][inner_taper_idx])]
            outer_taper_idx = outer_taper_idx[np.argsort(selected['inner_gauss_m'][outer_taper_idx])]

            ordered_indices = np.concatenate([no_taper_idx, inner_taper_idx, outer_taper_idx])

            for i, row in enumerate(selected[ordered_indices]):
                psf_name = row['psf_imagename'] + ".psf" if not row['psf_imagename'].endswith(".psf") else row['psf_imagename']
                psf_path = data_path / psf_name

                inner_label = "None" if np.isnan(row['inner_gauss_m']) else row['inner_gauss_m']
                outer_label = "None" if np.isnan(row['outer_gauss_as']) else row['outer_gauss_as']
                label = f"Inner: {inner_label} m, Outer: {outer_label} as"

                try:
                    psf = read_casa_image(psf_path)

                    center = np.unravel_index(np.argmax(psf), psf.shape)
                    cy, cx = center

                    beam = psf.beam
                    # pixels_per_beam = psf.pixels_per_beam
                    # cutout_window = 5 * int(pixels_per_beam**0.5)
                    pix_scale = proj_plane_pixel_scales(psf.wcs.celestial)[0] * u.deg
                    cutout_window = 5 * int((beam.major / pix_scale).to(u.one).value)


                    cutout = psf[cy-cutout_window:cy+cutout_window+1, 
                                 cx-cutout_window:cx+cutout_window+1]

                    axes[i].imshow(cutout.value, origin='lower', cmap='gray_r',
                                   vmin=0.0, vmax=1.0)
                    axes[i].contour(cutout.value, levels=[0.2, 0.5], colors='red', linewidths=1)

                    # Evaluate the beam model for plotting
                    axes[i].contour(gaussian_eval([beam.major.to(u.pixel).value / 2.355 / pix_scale.to(u.arcsec).value,
                                                   beam.minor.to(u.pixel).value / 2.355 / pix_scale.to(u.arcsec).value,
                                                   beam.pa.to(u.deg).value],
                                                  cutout.value,
                                                  (cutout.shape[0]//2, cutout.shape[1]//2)),
                                   levels=[0.2, 0.5], colors='cyan', linewidths=1,)

                    axes[i].set_title(label)
                    # axes[i].axis('off')
                except Exception as e:
                    axes[i].set_title(f"Error: {label}")
                    # axes[i].axis('off')
                    print(f"Could not load {psf_path}: {e}")

            plt.suptitle(f"Robust = {robust_val}")
            plt.tight_layout()

            plt.savefig(data_path / f'psf_grid_robust_{robust_val}.png', dpi=200)
            plt.close()


        # Same but gallery of the images:
        unique_robusts = np.unique(results_table['robust'])
        for robust_val in unique_robusts:
            # Select rows with this robust value
            mask = results_table['robust'] == robust_val
            selected = results_table[mask]
            if len(selected) == 0:
                continue

            fig, axes = plt.subplots(3, 3, 
                                     figsize=(12, 12), 
                                     squeeze=False)
            axes = axes.ravel()  # 1D array

            # Set the order of rows: no taper, inner taper, outer taper
            # No taper has nan for both inner and outer gauss
            no_taper_idx = np.where(np.isnan(selected['inner_gauss_m']) & 
                                   np.isnan(selected['outer_gauss_as']))[0]
            inner_taper_idx = np.where(np.isfinite(selected['inner_gauss_m']) & 
                                      np.isnan(selected['outer_gauss_as']))[0]
            outer_taper_idx = np.where(np.isfinite(selected['inner_gauss_m']) & 
                                      np.isfinite(selected['outer_gauss_as']))[0]

            # Sort by taper size within each group
            inner_taper_idx = inner_taper_idx[np.argsort(selected['inner_gauss_m'][inner_taper_idx])]
            outer_taper_idx = outer_taper_idx[np.argsort(selected['inner_gauss_m'][outer_taper_idx])]

            ordered_indices = np.concatenate([no_taper_idx, inner_taper_idx, outer_taper_idx])

            for i, row in enumerate(selected[ordered_indices]):
                image_name = row['psf_imagename'].replace(".psf", ".image")
                image_path = data_path / image_name

                inner_label = "None" if np.isnan(row['inner_gauss_m']) else row['inner_gauss_m']
                outer_label = "None" if np.isnan(row['outer_gauss_as']) else row['outer_gauss_as']
                label = f"Inner: {inner_label} m, Outer: {outer_label} as"

                try:
                    image = read_casa_image(image_path)

                    cy = image.shape[0] // 2
                    cx = image.shape[1] // 2

                    beam = image.beam
                    # pixels_per_beam = image.pixels_per_beam
                    # cutout_window = 5 * int(pixels_per_beam**0.5)
                    pix_scale = proj_plane_pixel_scales(psf.wcs.celestial)[0] * u.deg
                    # cutout_window = 5 * int((beam.major / pix_scale).to(u.one).value)

                    # Disk size in model is ~0.1". Make a ~0.3" per size cutout
                    cutout_window = int((0.15 * u.arcsec / pix_scale).to(u.one).value)

                    cutout = image[cy-cutout_window:cy+cutout_window+1, 
                                   cx-cutout_window:cx+cutout_window+1]

                    axes[i].imshow(cutout.value, origin='lower', cmap='gray')
                    axes[i].contour(cutout.value, levels=[0.2, 0.5], colors='red', linewidths=1)
                    axes[i].set_title(label)
                    # axes[i].axis('off')
                except Exception as e:
                    axes[i].set_title(f"Error: {label}")
                    # axes[i].axis('off')
                    print(f"Could not load {psf_path}: {e}")

            plt.suptitle(f"Robust = {robust_val}")
            plt.tight_layout()

            plt.savefig(data_path / f'image_grid_robust_{robust_val}.png', dpi=200)
            plt.close()


        #############
        # 1D profiles of PSFs
        #############

        # Assuming results_table is already loaded
        unique_robusts = np.unique(results_table['robust'])
        for robust_val in unique_robusts:
            # Select rows with this robust value
            mask = results_table['robust'] == robust_val
            selected = results_table[mask]
            if len(selected) == 0:
                continue

            fig, axes = plt.subplots(1, 2, 
                                     figsize=(9, 4), 
                                     squeeze=True,
                                     sharey=True,
                                     sharex=True)

            # Set the order of rows: no taper, inner taper, outer taper
            # No taper has nan for both inner and outer gauss
            no_taper_idx = np.where(np.isnan(selected['inner_gauss_m']) & 
                                   np.isnan(selected['outer_gauss_as']))[0]
            inner_taper_idx = np.where(np.isfinite(selected['inner_gauss_m']) & 
                                      np.isnan(selected['outer_gauss_as']))[0]
            outer_taper_idx = np.where(np.isfinite(selected['inner_gauss_m']) & 
                                      np.isfinite(selected['outer_gauss_as']))[0]

            # Sort by taper size within each group
            inner_taper_idx = inner_taper_idx[np.argsort(selected['inner_gauss_m'][inner_taper_idx])]
            outer_taper_idx = outer_taper_idx[np.argsort(selected['inner_gauss_m'][outer_taper_idx])]

            ordered_indices = np.concatenate([no_taper_idx, inner_taper_idx, outer_taper_idx])

            for i, row in enumerate(selected[ordered_indices]):
                psf_name = row['psf_imagename'] + ".psf" if not row['psf_imagename'].endswith(".psf") else row['psf_imagename']
                psf_path = data_path / psf_name

                inner_label = "None" if np.isnan(row['inner_gauss_m']) else row['inner_gauss_m']
                outer_label = "None" if np.isnan(row['outer_gauss_as']) else row['outer_gauss_as']
                label = f"Inner: {inner_label} m, Outer: {outer_label} as"

                if outer_label != "None":
                    linestyle = '-.'
                else:
                    linestyle = '-'

                try:
                    psf = read_casa_image(psf_path)

                    center = np.unravel_index(np.argmax(psf), psf.shape)
                    cy, cx = center

                    y_slice, x_slice = extract_psf_slices(psf, center)

                    beam = psf.beam
                    # pixels_per_beam = psf.pixels_per_beam
                    # cutout_window = 5 * int(pixels_per_beam**0.5)
                    pix_scale = proj_plane_pixel_scales(psf.wcs.celestial)[0] * u.deg
                    cutout_window = 50

                    y_vals = (np.arange(len(y_slice)) - cy) * pix_scale.to(u.arcsec).value
                    x_vals = (np.arange(len(x_slice)) - cx) * pix_scale.to(u.arcsec).value

                    # Y slices:
                    axes[0].plot(y_vals[cy-cutout_window:cy+cutout_window+1], 
                                 y_slice[cy-cutout_window:cy+cutout_window+1], 
                                 label=label, 
                                 linestyle=linestyle
                                 )

                    # X slices:
                    axes[1].plot(x_vals[cx-cutout_window:cx+cutout_window+1],
                                 x_slice[cx-cutout_window:cx+cutout_window+1], 
                                 linestyle=linestyle
                                #  label=label
                                 )


                except Exception as e:
                    # axes[i].set_title(f"Error: {label}")
                    # axes[i].axis('off')
                    print(f"Could not load {psf_path}: {e}")

            plt.subplots_adjust(right=0.667)

            # fig.legend(loc='center right', bbox_to_anchor=(0.75, 0.5), ncol=1, frameon=True)
            fig.legend(loc='center right', ncol=1, frameon=True)

            axes[0].set_title("Y Slice through PSF Center")
            axes[1].set_title("X Slice through PSF Center")
            axes[0].set_ylabel("Normalized Intensity")
            axes[0].set_xlabel("arcsec")
            axes[1].set_xlabel("arcsec")

            plt.suptitle(f"Robust = {robust_val}")
            # plt.tight_layout()

            plt.savefig(data_path / f'psf_1D_slices_robust_{robust_val}.png', dpi=200)
            plt.close()
