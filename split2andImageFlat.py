# ------------------------------------------------------------
# ngvla_twoTier_flat.py   – 2‑tier hierarchical imaging
# ------------------------------------------------------------
from casatasks import split, tclean, ft, uvsub, imregrid, feather
import os, shutil

# -------- PARAMETERS ----------------------------------------
ms_in      = 'ngVLA_214_ant_60s_noisy-model.ms'   # full MS
root       = 'ppdisk'                       # base name

uv_core    = '0~4300m'                      # ≤4.3 km
uv_outer   = '4300~1300000m'                # >4.3 km

# imaging params
core_cell  = '30mas'
core_imsize  = 256
core_thr  = '7e-6Jy'

outer_cell = '0.3mas'
outer_imsize = 2560
outer_thr = '2e-5Jy'

robust     = 2.0
niter      = 10000
# ------------------------------------------------------------

# ========== 0. SPLIT ========================================
ms_core  = f'{root}_core.ms'
ms_outer = f'{root}_outer.ms'

if not os.path.exists(ms_core):
    print(f'Splitting CORE {uv_core} → {ms_core}')
    split(vis=ms_in, outputvis=ms_core, datacolumn='data', uvrange=uv_core)
else:
    print(f'{ms_core} exists – skipping split')

if not os.path.exists(ms_outer):
    print(f'Splitting OUTER {uv_outer} → {ms_outer}')
    split(vis=ms_in, outputvis=ms_outer, datacolumn='data', uvrange=uv_outer)
else:
    print(f'{ms_outer} exists – skipping split')

# ========== 1. CLEAN CORE ===================================
core_tag = f'{root}_core'
# os.system(f'rm -rf {core_tag}*')  # clean up previous runs

print(f'Cleaning CORE → {core_tag}.image')
tclean(vis            = ms_core,
       imagename      = core_tag,
       datacolumn     = 'data',
       specmode       = 'mfs',
       weighting      = 'briggs', 
       robust=robust,
       cell           = core_cell,
       imsize         = core_imsize,
       niter          = niter,
       threshold      = core_thr,
       deconvolver    = 'multiscale',
       scales         = [0,5,15,45],
       savemodel      = 'modelcolumn',
       interactive    = False)

core_model  = core_tag  + '.model'
core_image  = core_tag  + '.image'

# ========== 2. SEED OUTER & CLEAN ============================
print('\nInserting CORE model into OUTER vis …')

ft(vis=ms_outer, model=core_model, usescratch=True)
uvsub(vis=ms_outer)   # residuals in DATA column

outer_tag = f'{root}_outer'
# os.system(f'rm -rf {outer_tag}*')  # clean up previous runs

print(f'Cleaning OUTER → {outer_tag}.image')
tclean(vis            = ms_outer,
       imagename      = outer_tag,
       datacolumn     = 'data',
       specmode       = 'mfs',
       weighting      = 'briggs', 
       robust=robust,
       cell           = outer_cell,
       imsize         = outer_imsize,
       niter          = niter,
       threshold      = outer_thr,
       deconvolver    = 'multiscale',
       scales         = [0,5,15,45],
       savemodel      = 'modelcolumn',
       interactive    = False)

outer_image = outer_tag + '.image'

# ========== 3. REGRID CORE IMAGE TO OUTER GRID ===============
core_regrid = core_image + '.regrid'
if os.path.isdir(core_regrid): shutil.rmtree(core_regrid)
print('\nRegridding CORE image onto OUTER grid …')
imregrid(imagename=core_image, template=outer_image, output=core_regrid)

# ========== 4. FEATHER COMBINATION ===========================
final_image = f'{root}_multiscale.image'
if os.path.isdir(final_image): shutil.rmtree(final_image)

print('\nFeathering CORE+OUTER → final combined image')
feather(imagename = final_image,
        highres   = outer_image,
        lowres    = core_regrid)

print('\n=== DONE ===')
print(f'Final multiscale product: {final_image}')

# ========== 5. COMPARISON TO ALL-IN imaging ===========================
all_tag = f'{root}_all'

print(f'Cleaning ALL → {all_tag}.image')
tclean(vis            = ms_in,
       imagename      = all_tag,
       datacolumn     = 'data',
       specmode       = 'mfs',
       weighting      = 'briggs', 
       robust=robust,
       cell           = outer_cell,
       imsize         = outer_imsize,
       niter          = niter,
       threshold      = outer_thr,
       deconvolver    = 'multiscale',
       scales         = [0,5,15,45],
       savemodel      = 'modelcolumn',
       interactive    = False)


# Keep a dirty image for comparison
all_tag = f'{root}_all_dirty'

print(f'Making dirty ALL → {all_tag}.image')
tclean(vis            = ms_in,
       imagename      = all_tag,
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

