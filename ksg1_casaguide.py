
# From https://casaguides.nrao.edu/index.php/Simulating_ngVLA_Data-CASA6.7.0

import os

print(config.measurespath +'/alma/simmos')
os.system('cp '+config.measurespath +'/alma/simmos/ngvla-revF.main.cfg .')

# Make simulated ms
simobserve(project='ngVLA_214_ant_60s_noise_free-model', 
           skymodel='ppmodel_image_93GHz.fits', 
           setpointings=True, 
           integration='60s', 
           obsmode='int', 
           antennalist='ngvla-revF.main.cfg', 
           hourangle='transit', 
           totaltime='14400s',  
           thermalnoise='', 
           graphics='none')

# Add noise

## Create a copy of the noise-free MS:
os.system('cp -r ngVLA_214_ant_60s_noise_free-model/ngVLA_214_ant_60s_noise_free-model.ngvla-revF.main.ms  ngVLA_214_ant_60s_noisy-model.ms')

## Open the MS we want to add noise to with the sm tool:
sm.openfromms('ngVLA_214_ant_60s_noisy-model.ms')

## Set the noise level using the simplenoise parameter estimated in the section on Estimating the Scaling Parameter for Adding Thermal Noise:
sigma_simple='1.4mJy'
sm.setnoise(mode='simplenoise', simplenoise=sigma_simple)

## Add noise to the 'DATA' column (and the 'CORRECTED_DATA' column if present):
sm.corrupt()

## Close the sm tool:
sm.done()


# Check added noise structure
importfits( fitsimage='ppmodel_image_93GHz.fits', 
            imagename='ppmodel_image_93GHz.image')  

# Make the noise-only image
tclean(vis='ngVLA_214_ant_60s_noisy-model.ms', 
       datacolumn='data', 
       imagename='sm_clean_noisy', 
       imsize=2560, 
       cell='0.3mas', 
       startmodel='ppmodel_image_93GHz.image', 
       specmode='mfs', 
       gridder='standard', 
       deconvolver='hogbom', 
       weighting='natural', 
       niter=0)

stat = imstat('sm_clean_noisy.residual')
print(stat['rms'])



# Different weighting comparisons:
tclean(vis='ngVLA_214_ant_60s_noisy-model.ms', 
       datacolumn='data', 
       imagename='ppdisk-sim-rob-1', 
       imsize=2560, 
       cell='0.3mas', 
       specmode='mfs', 
       weighting='briggs', 
       robust=-1, 
       niter = 10000, 
       threshold='1e-5Jy')

tclean(vis='ngVLA_214_ant_60s_noisy-model.ms', 
       datacolumn='data', 
       imagename='ppdisk-sim-rob0', 
       imsize=2560, 
       cell='0.3mas', 
       specmode='mfs', 
       weighting='briggs', 
       robust=0, 
       niter = 10000, 
       threshold='5e-6Jy')

tclean(vis='ngVLA_214_ant_60s_noisy-model.ms', 
       datacolumn='data', 
       imagename='ppdisk-sim-rob1', 
       imsize=2560, 
       cell='0.3mas', 
       specmode='mfs', 
       weighting='briggs', 
       robust=1, 
       niter = 10000, 
       threshold='3e-6Jy')

for i in ['-1','0','1']:
    stat = imstat(imagename='ppdisk-sim-rob'+str(i)+'.image',box='0,0,200,200')
    head = imhead(imagename='ppdisk-sim-rob'+str(i)+'.image')
    print('ppdisk-sim-rob'+str(i)+'.image:'+' rms: '+str(stat['rms'])+'; bmaj: '+str(head['restoringbeam']['major'])+'; bmin: '+str(head['restoringbeam']['minor'])+'\n')


# Example with outer tapering

tclean(vis='ngVLA_214_ant_60s_noisy-model.ms', 
       datacolumn='data',
       imagename='ppdisk-sim-rob0.15',
       imsize=2560,
       cell='0.3mas',
       specmode='mfs',
       weighting='briggs',
       robust=0.15,
       niter = 10000,
       threshold='6e-6Jy')

tclean(vis='ngVLA_214_ant_60s_noisy-model.ms',
       datacolumn='data',
       imagename='ppdisk-sim-rob-1.81-taper4.68mas',
       imsize=2560,
       cell='0.3mas',
       specmode='mfs',
       weighting='briggs', 
       robust=-1.81, 
       niter=10000, 
       threshold='1.5e-5Jy',
       uvtaper=['4.68mas','4.68mas','0deg'])


for i in ['0.15','-1.81-taper4.68mas']:
    stat = imstat(imagename='ppdisk-sim-rob'+str(i)+'.image',box='0,0,200,200')
    head = imhead(imagename='ppdisk-sim-rob'+str(i)+'.image')
    print('ppdisk-sim-rob'+str(i)+'.image:'+' rms: '+str(stat['rms'])+'; bmaj: '+str(head['restoringbeam']['major'])+'; bmin: '+str(head['restoringbeam']['minor'])+'\n')


