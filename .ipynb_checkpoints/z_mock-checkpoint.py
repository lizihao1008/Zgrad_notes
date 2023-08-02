import os, sys
import emcee
import yaml
import astropy.io.ascii as ascii
from scipy import ndimage
from astropy.io import fits
from astropy.table import Table,join
import astropy as ap
from multiprocessing import Pool,Value
import warnings
from functools import partial
import corner
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
cosmos = FlatLambdaCDM(Om0=0.3,H0=70)
import h5py
from scipy.ndimage import gaussian_filter
import zg_utils
import numpy as np
from zg_utils import corrDust

# method = 'bian18'
method = 'sanders23'

rootpath_caldb     = os.path.expanduser('')
if method == 'sanders23':
    fn_caldb = 'calibr_sanders23.yml'
elif method == 'bian18':
    fn_caldb = 'calibr_bian18.yml'
    
print(f'use {fn_caldb}')
caldb = yaml.safe_load(open(os.path.join(rootpath_caldb, fn_caldb), 'r'))
oh12_norm = caldb['norm']
caldb_name = caldb['name'].replace('+','')


def z_mock(z,snr,Av,n=100,method='sanders23'):
    '''
    generate n mock datasets given metallicity z, snr, and dust Av.
    '''
    
    # print(f'use {fn_caldb} for mock.')
    if method == 'bian18':
        ratio_OIII = 10**np.polyval(caldb['OIII']['bestfit'][::-1],z)
    else:
        ratio_OIII = 10**np.polyval(caldb['OIII']['bestfit'][::-1],z)*4/3
        
    ratio_OII = 10**np.polyval(caldb['OII']['bestfit'][::-1],z)
    ratio_Hg = 10**np.polyval(caldb['Hg']['bestfit'][::-1],z)
    ratio_Ha = 10**np.polyval(caldb['Ha']['bestfit'][::-1],z)

    # based on OIII
    OIII = np.ones(n)*10
    Hb = OIII/ratio_OIII
    OII = Hb*ratio_OII
    Hg = Hb*ratio_Hg
    Ha = Hb*ratio_Ha
    
    OIII = corrDust(OIII, 5008.24, Av, output='observed')
    OII = corrDust(OII, 3727.092, Av, output='observed')
    Hg = corrDust(Hg, 4341.692, Av, output='observed')
    Hb = corrDust(Hb, 4862.71, Av, output='observed')
    Ha = corrDust(Ha, 6563, Av, output='observed')

    
    sig = OIII/snr
    Hg += sig*np.random.randn(n)
    OIII += sig*np.random.randn(n)
    OII += sig*np.random.randn(n)
    Hb += sig*np.random.randn(n)
    Ha += sig*np.random.randn(n)
    
    # set 1 sigma upper limit
    OIII = np.where(OIII<sig,sig,OIII)
    OII = np.where(OII<sig,sig,OII)
    Hb = np.where(Hb<sig,sig,Hb)
    Hg = np.where(Hg<sig,sig,Hg)
    Ha = np.where(Ha<sig,sig,Ha)

    
    out = Table({'id':np.arange(n)})
    out['flux_OIII'] = OIII
    out['flux_OII'] = OII
    out['flux_Hb'] = Hb
    out['flux_Ha'] = Ha
    out['flux_Hg'] = Hg
    out['err_OIII'] = sig
    out['err_Hb'] = sig
    out['err_Ha'] = sig
    out['err_Hg'] = sig
    out['err_OII'] = sig
    out['z_input'] = z
    out['root'] = 'mock'
    return out

from astropy.table import vstack

snr_arr = np.arange(2,22,1)
z_arr = np.arange(7.5,8.5,0.05)
Av_arr = np.arange(0,2,0.5)
# Av_arr = np.array([0,0.5,1])

dic = {}

n = 100

for Av in Av_arr:
    Av_key = f'Av_{Av:.1f}'
    dic[Av_key] = {}
    for snr in snr_arr:
        snr_key = f'snr_{snr}'
        dic[Av_key][snr_key] = {}
        for z in z_arr:
            z_key = f'Z_{z:.2f}'
            dic[Av_key][snr_key][z_key] = z_mock(z,snr,Av,n,method)


metal_analysis = zg_utils.metal_analysis(fn_caldb=fn_caldb,calib_range='upper',nwalkers=12,N2Ha=0,Av_0=0.5)

out_dic = {}
import time
N = len(snr_arr)*len(z_arr)*len(Av_arr)
i = 0
t0 = time.time()

for Av_key in dic:
    out_dic[Av_key] = {}
    for snr_key in dic[Av_key]:
        out_dic[Av_key][snr_key] = {}
        for z_key in dic[Av_key][snr_key]:
            print(f'run {Av_key} {snr_key} {z_key} {i}/{N}')

            data = dic[Av_key][snr_key][z_key]
            out_z = metal_analysis.run_from_catalog(data,verbose=False,plot=False,save_path='.',nproc=128,
                                            lines=['OIII','OII','Ha','Hb','Hg'])
            out_dic[Av_key][snr_key][z_key] = out_z
            i += 1
            t1 = time.time()
            dt = t1 - t0
            estimated_time = (dt/i*N-dt)/60
            print(f'Estimeted time remaining: {estimated_time} min!')
        
np.save('z_mock_grid.npy',out_dic)