
import numpy as np
import matplotlib.pyplot as plt

import os

from casatools import table, image
from casatasks import tclean, apparentsens, imstat, rmtables, imhead, immath

tb = table()
ia = image()

arcsec_to_rad = 3600. * 180. / (2 * np.pi)  # Convert radians to arcseconds

model_image = 'ppmodel_image_93GHz.image'

this_vis = 'ngVLA_214_ant_60s_noisy-model_innergauss_weights.ms'
this_vis_orig = 'ngVLA_214_ant_60s_noisy-model.ms'


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
outer_nsigma_thr = 5.0

niter = 0


# Tapering functions

def inner_gaussian_taper(uvw_arr, taper_size_m=1.e3):
    gauss_arg = (uvw_arr[0]**2 + uvw_arr[1]**2) / (2 * taper_size_m**2)
    return 1 - np.exp(-gauss_arg)

def outer_gaussian_taper(uvw_arr, taper_size_m=1.e3):
    gauss_arg = (uvw_arr[0]**2 + uvw_arr[1]**2) / (2 * taper_size_m**2)
    return np.exp(-gauss_arg)


# Image fidelity
def calc_fidelity(inimg,refimg,pbimg='',psfimg='',fudge_factor=1.0,scale_factor=1.0,pb_thresh=0.25,clean_up=True,outfile=''):
    """Calculate fidelity of inimg with reference to refimg. 

    Implementation from: https://github.com/bmason72/casapy/blob/master/combo_utils.py#L203

    Use Gaussian PSF with parameters described in inimg header, unless
    an explicit psfimg is provided.

    If a primary beam image (pbimg) is provided, use it to restrict
    the area over which the fidelity is calculated, using pb_thresh as
    the lower limit (relative to max(pbimg)).

    clean_up controls whether intermediate files created in the
    process are removed or not (all contain the string TMP). These can
    be useful for sanity checking, but proper behavior is not
    guaranteed if any are present already when the routine is called.

    outfile specifies the file-name root for a fidelity image and a fractional error
    image that will be created.
       --- ****a pbimg is required to creat the outfile
       
    inimg, refimg, [psfimg], and [pbimg] should be CASA images. outfile is a CASA image.
    All input images should have the same axes and axis order.

    ***pbimg, if provided, should furthermore have the same pixel coordinates as inimg
       (Cell size, npix, coordinate reference pixel, etc)***

    ***all input images should have the same number and ordering of axes!!!

    fudge_factor multiples the beamwidth obtained from the input image, before 
      convolving refimg for comparison
    scale_factor multiplies the inimg pixel values (i.e. it recalibrates them)
      --> use these reluctantly and only if you know what you are doing

    OUTPUTS:  a dictionary containing

      f1 = 1 - max(abs(inimg-refimg)) / max(refimg) - 'classic' definition

      f2 = 1 - sum( refimg .* abs(inimg-refimg) ) / sum( refimg .* inimg)
         --> this is a somewhat poorly behaved fidelity definition that was evaluated for ngVLA
              (appearing in the draft ngVLA science requirements, May 2019)
         --> it is equivalent to a weighted sum of fractional errors, with the fraction taken with
               respect to the formed image inimg and the weight being inimg*refimg

      f2b = 1 - sum( refimg .* abs(inimg-refimg) ) / sum( refimg .* refimg)
q        --> this is the original (ngVLA science requirememts, Nov. 2017) and better-behaved 
               ngVLA fidelity definition, with the fraction taken with respect to the model (refimg),
               and the weight being refimg^2

      f3 = 1 - sum( beta .* abs(inimg-refimg) ) / sum( beta.^2 )
         --> this is the current ngVLA fidelity definition that has been adopted, where
                beta_i = max(abs(inimg_i,),abs(refimg_i))

      In all of the above "i" is a pixel index, .* and .^ are element- (pixel-) wise operations,
       and sums are over pixels

      Various ALMA-adopted fidelity measures are also reported (above 0.1%, 1%, 3%, 10%), 
      and the correlation coefficient


    HISTORY: 
      August/September 2019 - B. Mason (nrao) - original version

    """

    # ia = iatool()

    ia.open(inimg)
    # average over the stokes axis to get it down to 3 axes which is what our other one has
    imvals=np.squeeze(ia.getchunk()) * scale_factor
    img_cs = ia.coordsys()
    # how to trim the freq axis--
    #img_shape = (ia.shape())[0:3]
    img_shape = ia.shape()
    ia.close()
    # get beam info
    hdr = imhead(imagename=inimg,mode='summary')
    bmaj_str = str(hdr['restoringbeam']['major']['value'] * fudge_factor)+hdr['restoringbeam']['major']['unit']
    bmin_str = str(hdr['restoringbeam']['minor']['value'] * fudge_factor)+hdr['restoringbeam']['minor']['unit']
    bpa_str =  str(hdr['restoringbeam']['positionangle']['value'])+hdr['restoringbeam']['positionangle']['unit']

    # i should probably also be setting the beam * fudge_factor in the *header* of the input image

    if len(pbimg) > 0:
        ia.open(pbimg)
        pbvals=np.squeeze(ia.getchunk())
        pbvals /= np.max(pbvals)
        pbvals = np.where( pbvals < pb_thresh, 0.0, pbvals)
        #good_pb_ind=np.where( pbvals >= pb_thresh)
        #bad_pb_ind=np.where( pbvals < pb_thresh)
        #pbvals[good_pb_ind] = 1.0
        #if bad_pb_ind[0]:
        #    pbvals[bad_pb_ind] = 0.0
    else:
        pbvals = imvals*0.0 + 1.0
        #good_pb_ind = np.where(pbvals)
        #bad_pb_ind = [np.array([])]

    ##

    ##############
    # open, smooth, and regrid reference image
    #

    smo_ref_img = refimg+'.TMP.smo'

    # if given a psf image, use that for the convolution. need to regrid onto input
    #   model coordinate system first. this is mostly relevant for the single dish
    #   if the beam isn't very gaussian (as is the case for alma sim tp)
    if len(psfimg) > 0:
        # consider testing and fixing the case the reference image isn't jy/pix
        ia.open(refimg)
        ref_cs=ia.coordsys()
        ref_shape=ia.shape()
        ia.close()
        ia.open(psfimg)
        psf_reg_im=ia.regrid(csys=ref_cs.torecord(),shape=ref_shape,outfile=psfimg+'.TMP.regrid',overwrite=True,axes=[0,1])
        psf_reg_im.done()
        ia.close()
        ia.open(refimg)
        # default of scale= -1.0 autoscales the PSF to have unit area, which preserves "flux" in units of the input map
        #  scale=1.0 sets the PSF to have unit *peak*, which results in flux per beam in the output 
        ref_convd_im=ia.convolve(outfile=smo_ref_img,kernel=psfimg+'.TMP.regrid',overwrite=True,scale=1.0)
        ref_convd_im.setbrightnessunit('Jy/beam')
        ref_convd_im.done()
        ia.close()
        if clean_up:
            rmtables(psfimg+'.TMP.regrid')
    else:
        # consider testing and fixing the case the reference image isn't jy/pix
        ia.open(refimg)    
        im2=ia.convolve2d(outfile=smo_ref_img,axes=[0,1],major=bmaj_str,minor=bmin_str,pa=bpa_str,overwrite=True)
        im2.done()
        ia.close()

    smo_ref_img_regridded = smo_ref_img+'.TMP.regrid'
    ia.open(smo_ref_img)
    im2=ia.regrid(csys=img_cs.torecord(),shape=img_shape,outfile=smo_ref_img_regridded,overwrite=True,axes=[0,1])
    refvals=np.squeeze(im2.getchunk())
    im2.done()
    ia.close()

    ia.open(smo_ref_img_regridded)
    refvals=np.squeeze(ia.getchunk())
    ia.close()

    # set all pixels to zero where the PB is low - to avoid NaN's
    imvals = np.where(pbvals,imvals,0.0)
    refvals = np.where(pbvals,refvals,0.0)
    #if len(bad_pb_ind) > 0:
        #imvals[bad_pb_ind] = 0.0
        #refvals[bad_pb_ind] = 0.0

    deltas=(imvals-refvals).flatten()
    # put both image and model values in one array to calculate Beta for F_3- 
    allvals = np.array( [np.abs(imvals.flatten()),np.abs(refvals.flatten())])
    # the max of (image_pix_i,model_pix_i), in one flat array of length nixels
    maxvals = allvals.max(axis=0)

    # carilli definition. rosero eq1
    f_eq1 = 1.0 - np.max(np.abs(deltas))/np.max(refvals)
    f_eq2 = 1.0 - (refvals.flatten() * np.abs(deltas)).sum() / (refvals * imvals).sum()
    f_eq2b = 1.0 - (refvals.flatten() * np.abs(deltas)).sum() / (refvals * refvals).sum()
    #f_eq3 = 1.0 - (maxvals[gi] * np.abs(deltas[gi])).sum() / (maxvals[gi] * maxvals[gi]).sum()
    f_eq3 = 1.0 - (pbvals.flatten() * maxvals * np.abs(deltas)).sum() / (pbvals.flatten() * maxvals * maxvals).sum()

    # if an output image was requested, and a pbimg was given; make one.
    if ((len(outfile)>0) & (len(pbimg)>0)):
        weightfile= 'mypbweight.TMP.im'
        rmtables(weightfile)
        immath(imagename=[pbimg],mode='evalexpr',expr='ceil(IM0/max(IM0) - '+str(pb_thresh)+')',outfile=weightfile)
        betafile = 'mybeta.TMP.im'
        rmtables(betafile)
        immath(imagename=[inimg,smo_ref_img_regridded],mode='evalexpr',expr='iif(abs(IM0) > abs(IM1),abs(IM0),abs(IM1))',outfile=betafile)
        # 19sep19 - change to the actual F_3 contrib ie put abs() back in
        rmtables(outfile)
        print(" Writing fidelity error image: "+outfile)
        immath(imagename=[inimg,smo_ref_img_regridded,weightfile,betafile],expr='IM3*IM2*abs(IM0-IM1)/sum(IM3*IM3*IM2)',outfile=outfile)
        # 19sep19 - add fractional error (rel to beta) to output
        rmtables(outfile+'.frac')
        print(" Writing fractional error image: "+outfile+'.frac')
        immath(imagename=[inimg,smo_ref_img_regridded,weightfile,betafile],expr='IM2*(IM0-IM1)/IM3',outfile=outfile+'.frac')
        if clean_up:
            rmtables(weightfile)
            rmtables(betafile)

    # pearson correlation coefficient evaluated above beta = 1% peak reference image
    gi = np.where( np.abs(maxvals) > 0.01 * np.abs(refvals.max()) )
    ii = imvals.flatten()
    mm = refvals.flatten()
    mm -= mm.min()
    # (x-mean(x)) * (y-mean(y)) / sigma_x / sigma_y
    cc = (ii[gi] - ii[gi].mean()) * (mm[gi] - mm[gi].mean()) / (np.std(ii[gi]) * np.std(mm[gi]))
    #cc = (ii[gi] - ii[gi].mean()) * (mm[gi] - mm[gi].mean()) / (np.std(mm[gi]))**2
    corco = cc.sum() / cc.shape[0]

    fa = np.abs(mm) / np.abs(mm - ii)
    fa_0p1 = np.median( fa[ (np.abs(ii) > 1e-3 * mm.max()) | (np.abs(mm) > 1e-3 * mm.max())  ])
    fa_1 = np.median( fa[ (np.abs(ii) > 1e-2 * mm.max()) | (np.abs(mm) > 1e-2 * mm.max())  ])
    fa_3 = np.median( fa[ (np.abs(ii) > 3e-2 * mm.max()) | (np.abs(mm) > 3e-2 * mm.max())  ])
    fa_10 = np.median( fa[ (np.abs(ii) > 1e-1 * mm.max()) | (np.abs(mm) > 1e-1 * mm.max()) ] )

    #gi2 = (np.abs(ii) > 1e-3 * mm.max()) | (np.abs(mm) > 1e-3 * mm.max())  

    print("*************************************")
    print('image: ',inimg,'reference image:',refimg)
    print("Eq1  / Eq2  / Eq2b  / Eq3 / corrCoeff ")
    print(f_eq1, f_eq2, f_eq2b, f_eq3,corco)
    print(' ALMA (A_0.1%, A_1%, A_3%, A_10%): ',fa_0p1,fa_1,fa_3,fa_10)
    print("*************************************")

    fidelity_results = {'f1': f_eq1, 'f2': f_eq2, 'f2b': f_eq2b, 'f3': f_eq3, 'falma': [fa_0p1, fa_1, fa_3, fa_10]}

    if clean_up:
        rmtables(smo_ref_img)
        rmtables(smo_ref_img_regridded)

    return fidelity_results


# print(argh)

# Record apparent sensitivities for each combination:
ptsrc_sensitivities_appsens = []
ptsrc_sensitivities_modelfree = []

image_fidelity = []


do_dirty_image = True
do_deconvolution = True
do_noiseonly_image = True

# Iterate through robust values
# for this_robust in [0.5]:
for this_robust in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]:

    this_orig_tag = f'{tag}_robust_{this_robust}'

    # Original versions without reweighting
    if do_dirty_image:
        tclean(vis            = this_vis_orig,
            imagename      = this_orig_tag,
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

        # Make a dirty image backup to compare with the cleaned image fidelity.
        if not os.path.exists(f'{this_orig_tag}_dirty.image'):
            os.rename(f'{this_orig_tag}.image', f'{this_orig_tag}_dirty.image')

        fidel_dict = calc_fidelity(f'{this_orig_tag}_dirty.image', model_image)
        image_fidelity.append([f'{this_orig_tag}_dirty', this_robust, fidel_dict['f3']])

    if do_deconvolution:
        if not os.path.exists(f'{this_orig_tag}_dirty.image'):
            # If the dirty image wasn't made, make sure to remove any existing image first
            os.system(f"cp -r {this_orig_tag}.image {this_orig_tag}_dirty.image")

        tclean(vis            = this_vis_orig,
            imagename      = this_orig_tag,
            datacolumn     = 'data',
            specmode       = 'mfs',
            weighting      = 'briggs', 
            robust=this_robust,
            cell           = outer_cell,
            imsize         = outer_imsize,
            niter          = 10000,
            nsigma=outer_nsigma_thr,
            deconvolver    = 'multiscale',
            scales         = [0,5,15,45],
            savemodel      = 'none',
            interactive    = False,
            calcpsf=False, calcres=False, restart=True)

        fidel_dict = calc_fidelity(f'{this_orig_tag}.image', model_image)
        image_fidelity.append([this_orig_tag, this_robust, fidel_dict['f3']])

    # Now image with the true model subtracted to get noise-only image
    if do_noiseonly_image:
        tclean(vis            = this_vis_orig,
            imagename      = f'{this_orig_tag}_noiseonly',
            datacolumn     = 'data',
            startmodel=model_image, 
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
            interactive    = False,
            calcpsf=False)
        
        this_stat = imstat(f'{this_orig_tag}_noiseonly.residual')
        ptsrc_sensitivities_modelfree.append([this_orig_tag, this_stat['rms'], np.nan])

        # Clean up the noise-only images
        # rmtables(f'{this_orig_tag}_noiseonly.image')
        rmtables(f'{this_orig_tag}_noiseonly.residual')
        rmtables(f'{this_orig_tag}_noiseonly.model')
        rmtables(f'{this_orig_tag}_noiseonly.pb')
        rmtables(f'{this_orig_tag}_noiseonly.psf')
        rmtables(f'{this_orig_tag}_noiseonly.sumwt')


    # This doesn't seem to work properly. It should reflect the WEIGHT column but doesn't.
    # out = \
    #     apparentsens(vis=this_vis_orig,
    #                  specmode  = 'mfs',
    #                  weighting = 'briggs', 
    #                  robust=this_robust,
    #                  cell=outer_cell,
    #                  imsize=outer_imsize)
    # ptsrc_sensitivities_appsens.append([this_orig_tag, out['effSens'], out['relToNat']])

    for apply_outer_taper in [False, True]:

        for taper_size_m in taper_sizes_m:

            print(f'Processing taper size: {taper_size_m} m')

            this_weight_tag = f'{tag}_robust_{this_robust}_inneruv_gauss_weights_{int(taper_size_m)}m'

            if apply_outer_taper:
                this_weight_tag = f'{this_weight_tag}_outeruv_gauss_weights_{sigma_outer}as'
                print(f'  with outer taper sigma={sigma_outer} as')

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

            # Make a dirty image
            if do_dirty_image:
                tclean(vis=this_vis,
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

                # Make a dirty image backup to compare with the cleaned image fidelity.
                if not os.path.exists(f'{this_weight_tag}_dirty.image'):
                    os.rename(f'{this_weight_tag}.image', f'{this_weight_tag}_dirty.image')

                fidel_dict = calc_fidelity(f'{this_weight_tag}_dirty.image', model_image)
                image_fidelity.append([f'{this_weight_tag}_dirty', this_robust, fidel_dict['f3']])

            # Now deconvolve
            if do_deconvolution:
                if not os.path.exists(f'{this_weight_tag}_dirty.image'):
                    # If the dirty image wasn't made, make sure to remove any existing image first
                    os.system(f"cp -r {this_weight_tag}.image {this_weight_tag}_dirty.image")

                tclean(vis=this_vis,
                    imagename      = this_weight_tag,
                    datacolumn     = 'data',
                    specmode       = 'mfs',
                    weighting      = 'briggs', 
                    robust=this_robust,
                    cell           = outer_cell,
                    imsize         = outer_imsize,
                    niter          = 10000,
                    nsigma=outer_nsigma_thr,
                    deconvolver    = 'multiscale',
                    scales         = [0,5,15,45],
                    savemodel      = 'none',
                    interactive    = False,
                    calcpsf=False, calcres=False, restart=True)

                fidel_dict = calc_fidelity(f'{this_weight_tag}.image', model_image)
                image_fidelity.append([this_weight_tag, this_robust, fidel_dict['f3']])

            # Now image with the true model subtracted to get noise-only image
            if do_noiseonly_image:
                tclean(vis=this_vis,
                    imagename      = f'{this_weight_tag}_noiseonly',
                    datacolumn     = 'data',
                    startmodel=model_image, 
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
                    interactive    = False,
                    calcpsf=False)
            
                this_stat = imstat(f'{this_weight_tag}_noiseonly.residual')
                ptsrc_sensitivities_modelfree.append([this_weight_tag, this_stat['rms'], np.nan])
                # Clean up the noise-only images
                # rmtables(f'{this_weight_tag}_noiseonly.image')
                rmtables(f'{this_weight_tag}_noiseonly.residual')
                rmtables(f'{this_weight_tag}_noiseonly.model')
                rmtables(f'{this_weight_tag}_noiseonly.pb')
                rmtables(f'{this_weight_tag}_noiseonly.psf')
                rmtables(f'{this_weight_tag}_noiseonly.sumwt')

                # Fidelity with a perfect model corrupted by the non-Gaussian PSF.
                fidel_dict = calc_fidelity(f'{this_weight_tag}_noiseonly.image', model_image)
                image_fidelity.append([f'{this_weight_tag}_perfectmodel_dirty', this_robust, fidel_dict['f3']])


            # out = \
            #     apparentsens(vis=this_vis,
            #                 specmode='mfs',
            #                 weighting='briggs', 
            #                 robust=this_robust,
            #                 cell=outer_cell,
            #                 imsize=outer_imsize)
            # ptsrc_sensitivities_appsens.append([this_weight_tag, out['effSens'], out['relToNat']])

# Make a natural weight image for rms reference
natural_tag = f'{tag}_natural'

# Original versions without reweighting
if do_dirty_image:
    tclean(vis            = this_vis_orig,
        imagename      = natural_tag,
        datacolumn     = 'data',
        specmode       = 'mfs',
        weighting      = 'natural', 
        robust=0.0,
        cell           = outer_cell,
        imsize         = outer_imsize,
        niter          = 0,
        threshold      = outer_thr,
        deconvolver    = 'multiscale',
        scales         = [0,5,15,45],
        savemodel      = 'none',
        interactive    = False)

    if not os.path.exists(f'{natural_tag}_dirty.image'):
        os.rename(f'{natural_tag}.image', f'{natural_tag}_dirty.image')

    fidel_dict = calc_fidelity(f'{natural_tag}_dirty.image', model_image)
    image_fidelity.append([f'{natural_tag}_dirty', np.nan, fidel_dict['f3']])


if do_deconvolution:

    if not os.path.exists(f'{natural_tag}_dirty.image'):
        # If the dirty image wasn't made, make sure to remove any existing image first
        os.system(f"cp -r {natural_tag}.image {natural_tag}_dirty.image")

    tclean(vis            = this_vis_orig,
        imagename      = natural_tag,
        datacolumn     = 'data',
        specmode       = 'mfs',
        weighting      = 'natural', 
        robust=0.0,
        cell           = outer_cell,
        imsize         = outer_imsize,
        niter          = 10000,
        nsigma=outer_nsigma_thr,
        deconvolver    = 'multiscale',
        scales         = [0,5,15,45],
        savemodel      = 'none',
        interactive    = False,
        calcpsf=False, calcres=False, restart=True)

    fidel_dict = calc_fidelity(f'{natural_tag}.image', model_image)
    image_fidelity.append([natural_tag, np.nan, fidel_dict['f3']])

if do_noiseonly_image:
    # Now image with the true model subtracted to get noise-only image
    tclean(vis            = this_vis_orig,
        imagename      = f'{natural_tag}_noiseonly',
        datacolumn     = 'data',
        startmodel=model_image, 
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
        calcpsf=False,
        interactive    = False)
    
    this_stat = imstat(f'{natural_tag}_noiseonly.residual')
    ptsrc_sensitivities_modelfree.append([natural_tag, this_stat['rms'], 1.0])
    # Clean up the noise-only images
    # rmtables(f'{natural_tag}_noiseonly.image')
    rmtables(f'{natural_tag}_noiseonly.residual')
    rmtables(f'{natural_tag}_noiseonly.model')
    rmtables(f'{natural_tag}_noiseonly.pb')
    rmtables(f'{natural_tag}_noiseonly.psf')
    rmtables(f'{natural_tag}_noiseonly.sumwt')


# out = \
#     apparentsens(vis=this_vis_orig,
#                 specmode='mfs',
#                 weighting='briggs', 
#                 robust=this_robust,
#                 cell=outer_cell,
#                 imsize=outer_imsize)
# ptsrc_sensitivities_appsens.append([natural_tag, out['effSens'], out['relToNat']])

# ptsrc_filename = 'ptsrc_sensitivities_appsens.npy'
# if os.path.exists(ptsrc_filename):
#     os.remove(ptsrc_filename)
# np.save(ptsrc_filename, ptsrc_sensitivities_appsens)
# print(f'Saved point source sensitivities to {ptsrc_filename}')


if do_noiseonly_image:
    # Calculate relative to natural for model-free sensitivities
    nat_rms = ptsrc_sensitivities_modelfree[-1][1]
    for i in range(len(ptsrc_sensitivities_modelfree)):
        ptsrc_sensitivities_modelfree[i][2] = ptsrc_sensitivities_modelfree[i][1] / nat_rms

    # Make sure everything is a float
    for ii in range(len(ptsrc_sensitivities_modelfree)):
        ptsrc_sensitivities_modelfree[ii][1] = float(ptsrc_sensitivities_modelfree[ii][1])
        ptsrc_sensitivities_modelfree[ii][2] = float(ptsrc_sensitivities_modelfree[ii][2])

    ptsrc_filename = 'ptsrc_sensitivities_modelfree.npy'
    if os.path.exists(ptsrc_filename):
        os.remove(ptsrc_filename)
    np.save(ptsrc_filename, ptsrc_sensitivities_modelfree)
    print(f'Saved model-free point source sensitivities to {ptsrc_filename}')

fidelity_filename = 'image_fidelity.npy'
if os.path.exists(fidelity_filename):
    os.remove(fidelity_filename)
np.save(fidelity_filename, image_fidelity)
