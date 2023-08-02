import re
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table,join
from matplotlib.patches import Ellipse, Circle
from astropy.cosmology import FlatLambdaCDM
import linmix
import statsmodels.api as sm
from scipy.optimize import curve_fit
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.stats import binned_statistic
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from scipy import ndimage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter
from matplotlib import gridspec
import matplotlib.ticker as ticker
from multiprocessing import Pool,Value
from functools import partial
import corner
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import (detect_sources,make_2dgaussian_kernel)
from astropy.convolution import convolve
import yaml
import emcee
import os, sys
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from astropy.stats import sigma_clip
import random
import h5py
from astropy.io.misc.hdf5 import read_table_hdf5
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'cm'
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def makeMCMCsampleCornerPlot(data, passflag,save_path='.',ndim=3, burnin=2e2, plot_lnprob=True, dpi=100, verbose=True,):
    paramlatex = [r'12+log(O/H)', r'$A^\mathrm{N}_\mathrm{V}$', r'$f_{\mathrm{H}\beta}$']

    try:
        ndim = data['ndim']     # By default, use ndim and burnin values contained in the npy file, if present.
        burnin = data['burnin']
    except:
        if verbose: print(' =   =   >   NO keys of ndim and burnin found in the npy file, use input keyword values')
    name = data['name']
    samples = data['chain'][:, int(burnin):, :].reshape((-1, ndim))
    # samples[:,3] = np.log10(samples[:,3])
    if ndim==3:
        labels = paramlatex[:3]
    elif ndim==4:
        labels = paramlatex
    else:
        sys.exit(' ERR: keyword value out of range/empty/incorret')

    fs_label = 16
    fig = corner.corner(samples, labels=labels, label_kwargs={'fontsize':fs_label},
                        quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 16},
                        plot_datapoints=False, fill_contours=True, color="g", smooth=1.,
                        levels=1.0 - np.exp(-0.5 * np.arange(1.0, 2.1, 1.0) ** 2))
    if passflag==True:
        fig.suptitle('Converged',y=0.9)
    else:
        fig.suptitle('Not converged',y=0.9)

    if plot_lnprob:
        #-------- make the flatlnprob plot to see if chains converge
        fig.axes[ndim-1].plot(data['lnprob'][:, int(burnin):].flatten(),'.',color='k',markersize=0.3, markeredgewidth=0.1)
        fig.axes[ndim-1].set_ylabel(r'ln likelihood',fontsize=fs_label)
        fig.savefig(os.path.join(save_path,f'{name}_corner.png'),dpi=dpi, bbox_inches='tight', pad_inchies=0.0)



def estmode_savgol(sample, histNbin=50, savgolWindow=11, savgolpoly=3, density=True, return_full=False):
	sample = np.atleast_1d(sample)
	if len(sample.shape) != 1:
		print(' -   -   >   Flattening the input data array')
		sample = sample.ravel()

	(pdf, bin_edge) = np.histogram(sample, bins=histNbin, density=density)
	bin_ctr = 0.5*(bin_edge[1:]+bin_edge[:-1])
	pdf_smooth = savgol_filter(pdf, savgolWindow, savgolpoly, deriv=0)

	ind = np.where(pdf_smooth == np.max(pdf_smooth))[0]     # this finds all maximum indices, whereas np.argmax only returns the first one
	mode = bin_ctr[ind]
	if len(ind) > 1:
		print(' -   -   -   >   NOTE: there exists multiple mode points')

	if return_full:
		return mode, ind, pdf_smooth, bin_ctr
	else:
		return mode
	
	
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def vetPDF(sample, pdf_lowlim=0.48, histNbin=50, savgolWindow=11, savgolpoly=3, savelogfn=None, savelogmode='a',verbose=False):
    passflag = False
    # check if the input data array is 1D. if not, convert it.
    sample = np.atleast_1d(sample)
    if len(sample.shape) != 1:
        print(' -   -   Flattening the input data array')
        sample = sample.ravel()

    # obtain probability *density* function, normalized such that the *integral* over the range is 1!
    (pdf, bin_edge) = np.histogram(sample, bins=histNbin, density=True)
    bin_ctr = 0.5*(bin_edge[1:]+bin_edge[:-1])
    assert len(pdf)==len(bin_ctr), ' ERR: dimensions do not match!'

    # smooth pdf and calc the derivative of PDF at bin_ctr  =>  use Savitzky-Golay filter, see: https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
    pdf_smooth = savgol_filter(pdf, savgolWindow, savgolpoly, deriv=0)
    deriv = savgol_filter(pdf, savgolWindow, savgolpoly, deriv=1)

    # obtain -1sigma, mean, +1sigma, as well as their PDF and slope values
    percnt = np.percentile(sample, [16, 50, 84], axis=0)
    pdf_percnt = np.interp(percnt, bin_ctr, pdf_smooth)
    deriv_percnt = np.interp(percnt, bin_ctr, deriv)

    
    # set up log saving
    fp = sys.stdout
    if verbose:
        if savelogfn is not None:
            assert isinstance(savelogfn, str), ' ERR: wrong type (only string acceptable)'
            fp = open(savelogfn, savelogmode)
        # # print out result or save it to log

        print(' -   -   -   -   -   -   -   -   -   -   -   - ', file=fp)
        print(' -       PDF(50%)={:.4g} '.format(pdf_percnt[1]), file=fp)
        print(' -       PDF(16%)={:.4g}, deriv={:.4g}'.format(pdf_percnt[0], deriv_percnt[0]), file=fp)
        print(' -       PDF(84%)={:.4g}, deriv={:.4g}'.format(pdf_percnt[2], deriv_percnt[2]), file=fp)
        print(' -   -   -   -   -   -   -   -   -   -   -   - ', file=fp)
    # screening
    if pdf_percnt[1]>=pdf_lowlim:
        if pdf_percnt[1] != np.min(pdf_percnt):
            if pdf_percnt[1] == np.max(pdf_percnt):
                passflag = True
            elif pdf_percnt[0] == np.max(pdf_percnt):   # PDF(16%) > PDF(50%) => slope(16%)>0
                if deriv_percnt[0]>0: passflag = True
                elif verbose: print(' >   >   screened off in Step 3, PDF(16%) max and wrong deriv', file=fp)
            elif pdf_percnt[2] == np.max(pdf_percnt):   # PDF(84%) > PDF(50%) => slope(84%)<0
                if deriv_percnt[2]<0: passflag = True
                elif verbose: print(' >   >   screened off in Step 3, PDF(84%) max and wrong deriv', file=fp)
            elif verbose: sys.exit(' ERR: sth wrong wih np.max(pdf_percnt)')
        elif verbose:
            print(' >   >   screened off in Step 2', file=fp)
    elif verbose:
        print(' >   >   screened off in Step 1', file=fp)

    if passflag&verbose:
        print(' >   >   passed!', file=fp)
    if fp is not sys.stdout:
        fp.close()


    return passflag



def create_walkers(ndim, nwalkers, par0, scale):
	# set up the initial param value for all the walkers
	oh12_rand = np.random.uniform(par0[0] - scale[0], par0[0] + scale[0], size=nwalkers)
	Av_rand   = np.random.uniform(par0[1] - scale[1], par0[1] + scale[1], size=nwalkers)
	fHb_rand = np.random.uniform(par0[2] - scale[2], par0[2] + scale[2], size=nwalkers)
	par0arr = np.concatenate([oh12_rand, Av_rand, fHb_rand])
	return par0arr.reshape(ndim, nwalkers).T

def A_lam(lam0, Av, Rv=3.1, verbose=False):
	lam0 = np.atleast_1d(lam0)
	assert len(lam0)==1, ' ERR: only allow len=1!'
	x   = 1. / lam0
	# assert (np.max(lam0) <= 1. / 1.1) and (np.min(lam0) >= 1. / 3.3), ' ERR: keyword value out of range'

	# Case 1: optical/NIR
	if 1.1<=x<=3.3:
		if verbose: print(' -   -   >   correcting dust for optical/NIR')
		xx  = x-1.82    # subtract off 1.82, which corresp to V-band in micron^-1
		#-------- define a(x) and b(x)
		a = lambda y: 1. + 0.17699*y - 0.50447*y**2. - 0.02427*y**3. + 0.72085*y**4. + 0.01979*y**5. - 0.77530*y**6. + 0.32999*y**7.
		b = lambda y: 1.41338*y + 2.28305*y**2. + 1.07233*y**3. - 5.38434*y**4. - 0.62251*y**5. + 5.30260*y**6. - 2.09002*y**7.
		#-------- calc A(lambda)
		Alam = Av * (a(xx) + b(xx) / Rv)

	# Case 2: infrared
	elif 0.3<=x<1.1:
		if verbose: print(' -   -   >   correcting dust for infrared')
		#-------- define a(x) and b(x)
		a = lambda y: 0.574*y**1.61
		b = lambda y: -0.527*y**1.61
		#-------- calc A(lambda)
		Alam = Av * (a(x) + b(x) / Rv)

	# Case 3: ultraviolet
	elif 3.3<x<=8.:
		if verbose: print(' -   -   >   correcting dust for ultraviolet')
		if x>=5.9:
			Fa = lambda y: -0.04473*(y-5.9)**2. - 0.009779*(y-5.9)**3.
			Fb = lambda y: 0.2130*(y-5.9)**2. + 0.1207*(x-5.9)**3.
		else:
			Fa = lambda y: 0
			Fb = lambda y: 0
		#-------- define a(x) and b(x)
		a = lambda y: 1.752 - 0.316*y - 0.104/((y-4.67)**2. + 0.341) + Fa(y)
		b = lambda y: -3.090 + 1.825*y + 1.206/((y-4.62)**2. + 0.263) + Fb(y)
		#-------- calc A(lambda)
		Alam = Av * (a(x) + b(x) / Rv)

	return Alam

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def corrDust(flux, lineORlam0, Av, Rv=3.1, output='intrinsic', verbose=False):
#     print(lineORlam0)
	try:
#         lam0 = ELinfo[lineORlam0]['lam0']
		lam0 = ELinfo[lineORlam0]
	except:
		lam0 = lineORlam0
	if output == 'intrinsic':
		coef = 1.
		if verbose: print(' -   -   -   >   correct for dust to get *intrinsic* line flux')
	elif output == 'observed':
		coef = -1.
		if verbose: print(' -   -   -   >   correct for dust to get *observed* line flux')
	else:
		sys.exit(' ERR: keyword options not supported')
#     print([type(flux),type(coef),type(A_lam),type(lam0),type(Av),type(Rv)])
	return flux * 10.**(coef * 0.4 * A_lam(lam0 / 1.e4, Av, Rv=Rv))

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def save_sample(samplearr,outputname,verbose=True):
	"""
	Saving the sample array to file
	--- Additional Note ---
	* Borrowed from paper2_LBGsAbove7InGLASS.py
	"""
	if verbose: print(' - Saving sample array to '+outputname)
	hdrstr = '  '.join(np.asarray(samplearr.dtype.names))
	np.savetxt(outputname,samplearr,fmt="%s",header=hdrstr)
	
	return

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# DEF likelihood and prior functions
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

def lnlike(theta, fobs, N2Ha,corr=True):
    '''
    corr: wthether to correct dust using balmer decrement.
    '''
    oh12, Av, fHb = theta
    Z = oh12 - oh12_norm
    linelist = list(fobs.keys())
    fobs_nd = {}
    ratio = {}
    chisq = 0.0
    SIIexist = False
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 1st loop: computing universal quantities
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    for line in linelist:
        #-------- obtain fobs_nodust from fobs
        if corr:
            fobs_nd.update({line: corrDust(fobs[line], line, Av, output='intrinsic', verbose=False)})
        else:
            fobs_nd.update({line: fobs[line]})
        #-------- obtain line ratio and ratio uncert
        if caldb_name == 'Bian18' and line=='OIII':
            log10bf = np.polyval(caldb[line]['bestfit'][::-1],Z)
            ratio.update({line: np.asarray([10.**log10bf*3./4., 10.**log10bf*3./4.*np.log(10.)*caldb[line]['rms']])})
        else:
            log10bf = np.polyval(caldb[line]['bestfit'][::-1],Z)
            if caldb_name == 'Maiolino08' or line=='SII':
                log10up = np.polyval(caldb[line]['upper'][::-1],Z)
                log10low = np.polyval(caldb[line]['lower'][::-1],Z)
                ratio.update({line: np.asarray([10.**log10bf, 0.5*(10.**log10up - 10.**log10low)])})
            else:
                ratio.update({line: np.asarray([10.**log10bf, 10.**log10bf * np.log(10.) * caldb[line]['rms']])})
    #-------- correct for NII in Ha, if Ha is in linelist
    # print(ratio)
    # print(fobs_nd)
    if 'Ha' in linelist:
        fobs_nd['Ha']/=1.+N2Ha
    if 'NeIII' in linelist:
        linelist.pop(linelist.index('NeIII'))
    if 'SII' in linelist:
        SIIexist = True
        linelist.pop(linelist.index('SII'))
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 2nd loop: calc chi2, specifically dealing with NeIII/OII and SII/Ha ratios
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    for line in linelist:
        #--------- calc chi2
        chisq += (fobs_nd[line][0]-ratio[line][0]*fHb)**2./(fobs_nd[line][1]**2.+(fHb**2.)*(ratio[line][1]**2.))
    if SIIexist and 'Ha' in linelist:
        fHa = fobs_nd['Ha'][0]
        uHa = fobs_nd['Ha'][1]
        chisq += (fobs_nd['SII'][0]-ratio['SII'][0]*fHa)**2./\
                 (fobs_nd['SII'][1]**2.+(fHa**2.)*(ratio['SII'][1]**2.)+(uHa**2.)*(ratio['SII'][0]**2.))
    # print 'oh12={:.2g}, Av={:.2g}, fHb={:.2g}, chi2={:.3g}'.format(oh12,Av,fHb,chisq)
    return -0.5*chisq

def lnprior(theta):
    oh12, Av, fHb = theta
    if oh12_low < oh12 < oh12_up and Av_low < Av < Av_up and fHb_low < fHb < fHb_up:
        return -np.log(fHb)                 # Use Jeffrey's prior on fHb, and flat prior on oh12 and Av
        # return 0.0                          # Use flat prior on oh12, Av, and fHb
    else:
        return -np.inf

def lnprob(theta, fobs, N2Ha,corr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, fobs, N2Ha,corr)

    

class metal_analysis():
    
    def __init__(self,oh12_scale = 0.3,Av_low = 0.,Av_up = 4.,Av_0 = 2.0,
                 Av_scale = 0.5,fHb_low = 0.,fHb_up = 1000.,fHb_0 = 10.,fHb_scale = 5.,
                 ndim=3, nwalkers = 32,niter =  int(1e3),burnin = 2e2,N2Ha = 0.1,
                 ELinfo={'Ha': 6564.61,'Hg': 4340.471,'Hb': 4861.333,'OIII': 5006.843,'OII': 3728.815},
                 calib_range='upper',fn_caldb='/home/zihaoli/data/JWST/metallicity/calibr_sanders23.yml',Nbin=50,
                 savgolWindow=11,savgolpoly=3):
        '''
        fn_caldb: path to the metallicity calibration file.
        '''
        
        self.caldb = yaml.load(open(fn_caldb, 'r'))

        if calib_range == 'upper':
            self.oh12_low = self.caldb['urange'][0]
            self.oh12_up = self.caldb['urange'][1]
        elif calib_range == 'lower':
            self.oh12_low = self.caldb['lrange'][0]
            self.oh12_up = self.caldb['lrange'][1]
        oh12_0 = (self.oh12_up + self.oh12_low)/2
        # self.catalog = catalog
        self.oh12_0 = oh12_0
        self.oh12_scale = oh12_scale
        self.Av_low = Av_low
        self.Av_up = Av_up
        self.Av_0 = Av_0
        self.Av_scale = Av_scale
        self.fHb_low = fHb_low
        self.fHb_up = fHb_up
        self.fHb_0 = fHb_0
        self.fHb_scale = fHb_scale
        self.ndim = 3
        self.nwalkers = nwalkers
        self.niter = niter
        self.burnin = burnin
        self.ELinfo = ELinfo
        self.caldb_name = self.caldb['name'].replace('+','')
        self.oh12_norm = self.caldb['norm']
        self.N2Ha = N2Ha
        self.params2global(oh12_0 = oh12_0,oh12_scale = oh12_scale,Av_low = Av_low,Av_up = Av_up,Av_0 = Av_0,
                         Av_scale = Av_scale,fHb_low = fHb_low,fHb_up = fHb_up,fHb_0 = fHb_0,fHb_scale = fHb_scale,
                         ndim=ndim, nwalkers = nwalkers,niter =  niter,burnin = burnin,N2Ha = N2Ha,
                         ELinfo=ELinfo,caldb_name=self.caldb_name,oh12_norm=self.oh12_norm,oh12_low=self.oh12_low,
                         oh12_up=self.oh12_up,caldb=self.caldb,
                         calib_range=calib_range,fn_caldb=fn_caldb,Nbin=Nbin,savgolWindow=savgolWindow,savgolpoly=savgolpoly)
    
    def get_default_params(self):
        
        self.params2global(oh12_0 = 8.0,oh12_scale = 0.3,Av_low = 0.,Av_up = 4.,Av_0 = 2.0,
                             Av_scale = 0.5,fHb_low = 0.,fHb_up = 1000.,fHb_0 = 10.,fHb_scale = 5.,
                             ndim=3, nwalkers = 32,niter =  int(1e3),burnin = 2e2,N2Ha = 0.1,
                             ELinfo={'Ha': 6564.61,'Hg': 4340.471,'Hb': 4861.333,'OIII': 5006.843,'OII': 3728.815},
                             calib_range='upper',fn_caldb='/home/zihaoli/data/JWST/metallicity/calibr_sanders23.yml',Nbin=50,
                             savgolWindow=11,savgolpoly=3)
    @staticmethod
    def params2global(**kwargs):
        for i in kwargs:
            globals()[i] = kwargs[i]
            
        
    def metal_mp(self,dat_ELflux,plot=False,save_path='.',corr=True,verbose=False,nproc=128):

        p = Pool(min([len(dat_ELflux),nproc]))
        output = p.map(partial(metal_func,plot=plot,save_path=save_path,corr=corr,verbose=verbose),dat_ELflux)
        p.close()
        return output
    

    def run_from_catalog(self,catalog,lines=['OIII','OII','Ha','Hb','Hg'],plot=False,save_path='.',corr=True,verbose=True,nproc=128):
        from astropy.table import Table,hstack
        
        dat_ELflux = self.grizli_cat2input(catalog,lines=lines)
        output = self.metal_mp(dat_ELflux,plot=plot,save_path=save_path,corr=corr,verbose=verbose,nproc=nproc)
        titles = ['me', 'me_low2sig', 'me_low1sig' ,'me_up1sig', 
                  'me_up2sig', 'bf','mode','mode_percnt','mo_low2sig',
                  'mo_low1sig','mo_up1sig','mo_up2sig','passflag','me_std']
        out_dict = {titles[j]:[output[i][j] for i in range(len(output))] for j in range(len(titles))}
        out_cat = Table(out_dict)
        return hstack([catalog,out_cat])
    
    def run_from_Elines(self,Elines,plot=False,save_path='.',corr=True,verbose=True,nproc=128):
        '''
        Note flux_OIII is OIII 4959.
        '''
        Elines_list = []
        for i in range(len(Elines['flux_OIII'])):
            this_dic = {'name':i}
            for key in Elines:
                if 'err' not in key:
                    # print(f'find {key}.')
                    this_dic.update({key.split('_')[1]:np.array([[Elines[key][i]],[Elines[key.replace('flux','err')][i]]])})
            Elines_list.append(this_dic)
        output = self.metal_mp(Elines_list,plot=plot,save_path=save_path,corr=corr,verbose=verbose,nproc=nproc)
        titles = ['me', 'me_low2sig', 'me_low1sig' ,'me_up1sig', 
                  'me_up2sig', 'bf','mode','mode_percnt','mo_low2sig',
                  'mo_low1sig','mo_up1sig','mo_up2sig','passflag','me_std']
        
        out_dict = {titles[j]:[output[i][j] for i in range(len(output))] for j in range(len(titles))}
        out_dict.update(Elines)
        return out_dict
    
    @staticmethod
    def grizli_cat2input(cat,lines=['OIII','OII','Ha','Hb','Hg']):
        '''
        convert the grizli catatalog to the input format to analyse metallicity.
        '''
        factor = 10**np.ceil(-np.log10(cat['flux_Hb']))
        Elines = {}
        for line in lines:
            flux_list = []
            err_list = []
            for i in range(len(cat)):
                flux_list.append(cat[i]['flux_'+line])
                err_list.append(cat[i]['err_'+line])
            if line == 'OIII':
                Elines['flux_'+line] = np.array(flux_list)*0.75*factor
                Elines['err_'+line] = np.array(err_list)*0.75*factor
            else:
                Elines['flux_'+line] = np.array(flux_list)*factor
                Elines['err_'+line] = np.array(err_list)*factor

        Elines_list = []
        for i in range(len(cat)):
            if 'id' not in cat[0]:
                ID = i
            else:
                ID = cat[i]['id']
            if 'root' not in cat[i]:
                root = 'run'
            else:
                root = cat[i]['root']
            name = f'{root}-{ID:05d}'
            this_dic = {'name':name}
            for key in Elines:
                if 'err' not in key:
                    this_dic.update({key.split('_')[1]:np.array([[Elines[key][i]],[Elines[key.replace('flux','err')][i]]])})
            Elines_list.append(this_dic)

        return Elines_list
    
    
def metal_func(fobs,plot=False,save_path='.',corr=True,verbose=True):

    fobs_input = {}
    # print('Working on pixel %d'%val.value,end='\r')
    if np.isnan(fobs['OIII'][0]):
        # val.value += 1
        # print('pixel_%d is nan'%val.value)
        return tuple(np.nan*np.ones(14))

    # assuming NII/Ha=0.1

    # delete lines with no data
    for i_line in fobs.copy():
        if i_line == 'name':
            continue
        if (fobs[i_line][0] == 0)&((fobs[i_line][1] == 0)):
            # fobs.pop(i_line)
            pass
        else:
            fobs_input.update({i_line:fobs[i_line]})

    # only use Ha when Hg is avaibale.
    # if ('Ha' in fobs)&('Hg' in fobs):
    #     fobs.pop('Hg')
    pos = create_walkers(ndim, nwalkers, 
                         [oh12_0, Av_0, fHb_0], 
                         [oh12_scale, Av_scale, fHb_scale])
    #-------- design the sampler. Here have to use [] to wrap up fobsii, otherwise it will not be treated as an entity when passed to |lnlike|
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[fobs_input, N2Ha,corr])
    sampler.run_mcmc(pos, niter,progress=False)
    try:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[fobs_input, N2Ha,corr])
        sampler.run_mcmc(pos, niter,progress=False)
        rslt = {'chain':sampler.chain, 'lnprob':sampler.lnprobability, 'accpfrac':sampler.acceptance_fraction, 'burnin':burnin,
                'ndim':ndim, 'nwalkers':nwalkers, 'niter':niter,'name':fobs['name']}
        try: rslt['acor'] = sampler.acor
        
        except: pass

        samples = sampler.chain[:, int(burnin):, :].reshape((-1, ndim))
        #-------- convert the column of fHb to log(fHb)
        samples[:,2] = np.log(samples[:,2])

        #-------- obtain best-fit values (maximum likelihood soln)
        flatlnprob = sampler.lnprobability[:, int(burnin):].ravel()
        indmax = flatlnprob.argmax()
        bfs = samples[indmax,:]

        paramname = ['oh12']
        percnt = np.asarray([0.025, 0.16, 0.5, 0.84, 0.975])
        for ii, param in enumerate(paramname):
            sample = samples[:,ii]
            passflag = vetPDF(sample,verbose=verbose)
            me_val = np.percentile(sample, list(percnt*100.))
            mode = estmode_savgol(sample,histNbin=Nbin, savgolWindow=savgolWindow, savgolpoly=savgolpoly, density=True, return_full=False)

            temp_percnt = percnt-.5
            temp_percnt += len(np.where(sample<=mode[0])[0])/float(len(sample))

            # replace >100% and <0% with 1. and 0.
            temp_percnt = np.ma.masked_greater(temp_percnt, 1.)
            temp_percnt = temp_percnt.filled(1.)
            temp_percnt = np.ma.masked_less(temp_percnt, 0.)
            temp_percnt = temp_percnt.filled(0.)
            mo_val  = np.percentile(sample, list(temp_percnt*100.))
            if plot:
                makeMCMCsampleCornerPlot(rslt,passflag=passflag,save_path=save_path)
            return tuple([me_val[2], me_val[0],me_val[1], me_val[3], me_val[4],bfs[ii],mode[0], temp_percnt[2], mo_val[0], mo_val[1], mo_val[3], mo_val[4],int(passflag),np.std(sample)])
    except Exception as e:
        print(e)
        return tuple(np.nan*np.ones(14))
    

def O3Hb_metallicity(ratio,ratio_err=None,branch='upper',method='naka',intrinsic_scatter=0.05):
    '''
    ratio: log10(OIII5007/Hb)
    '''
    
    if isinstance(ratio,(float,np.float64)):
        ratio = np.array([ratio])
        if ratio_err is not None:
            ratio_err = np.array([ratio_err])
    R3_bian = lambda Z: (43.9836-21.6211*Z+3.4277*Z**2-0.1747*Z**3)+np.log10(0.75)
    R3_naka_all = lambda Z: np.poly1d(np.array([-0.277, -3.182, -2.832, -0.637])[::-1])(Z-8.69)
    R3_naka_large = lambda Z: np.poly1d(np.array([0.628, -0.660, -0.522])[::-1])(Z-8.69)
    R3_naka_small = lambda Z: np.poly1d(np.array([0.780, -0.072, -0.316])[::-1])(Z-8.69)
    R3_sanders = lambda Z: np.poly1d(np.array([0.834,-0.072, -0.453])[::-1])(Z-8)
    # R3_li = lambda Z: np.poly1d(np.array([[0.0814326,-2.18244839,19.4251872,-56.52058853]]))(Z)
    R3_li = lambda Z: np.poly1d(np.array([-4.44960746e-03, -2.05584738e-01,  4.29845451, -1.80536097e+01]))(Z)

    # R3_maiolino = lambda Z: np.poly1d(np.array([0.1549, -1.5031, -0.9790, -0.0297])[::-1])(Z-8.69)
    
    if method == 'bian':
        Zrange_upper = np.linspace(7.7729,9,100)
        Zrange_lower = np.linspace(6.5,7.7729,100)
        Z_lower = interp1d(R3_bian(Zrange_lower),Zrange_lower,fill_value=(6.5,7.8),bounds_error=False)
        Z_upper = interp1d(R3_bian(Zrange_upper),Zrange_upper,fill_value=7.8,bounds_error=False)
        
        dZdr = lambda Z: 1/(-21.6211+2*3.4277*Z-3*0.1747*Z**2)
        
    elif method == 'naka':
        Zrange_lower = np.linspace(7.1,8.1,100)
        Zrange_upper = np.linspace(8.1,8.9,100)
        Z_lower = interp1d(R3_naka_large(Zrange_lower),Zrange_lower,fill_value=8.1,bounds_error=False)
        Z_upper = interp1d(R3_naka_all(Zrange_upper),Zrange_upper,fill_value=8.1,bounds_error=False)
        if branch == 'upper':
            dZdr = lambda Z: 1/(-3*0.637*(Z-8.69)**2-2*2.832*(Z-8.69)-3.182)
        elif branch == 'lower':
            dZdr = lambda Z: 1/(-2*0.522*(Z-8.69)-0.66)
    
    elif method == 'li':
        Zrange_lower = np.linspace(6.5,8.25,100)
        Z_lower = interp1d(R3_li(Zrange_lower),Zrange_lower,fill_value=(6.5,8.25),bounds_error=False)
        dZdr = lambda Z: 1/(4.29845451-2*2.05584738e-01*Z-3*4.44960746e-03*Z**2)
        
    elif method == 'sanders':
        Zrange_lower = np.linspace(5,7.9207,100)
        Zrange_upper = np.linspace(7.9207,8.9,100)
        Z_lower = interp1d(R3_sanders(Zrange_lower),Zrange_lower,fill_value=(7,7.9207),bounds_error=False)
        Z_upper = interp1d(R3_sanders(Zrange_upper),Zrange_upper,fill_value=(8.9,7.9207,),bounds_error=False)

        dZdr = lambda Z: 1/(-2*0.453*(Z-8) - 0.072)
        
        
    # if np.isnan(ratio):
    #     return np.array(np.nan)
    if branch == 'upper':
        Z = Z_upper(ratio)
        
    if branch == 'lower':
        Z = Z_lower(ratio)
        
    if ratio_err is not None:
        Z_err  = (intrinsic_scatter**2+dZdr(Z)**2*ratio_err**2)**0.5
        if method == 'bian':
            Z_err[Z==7.8] = intrinsic_scatter*3
        if method == 'li':
            Z_err[Z==8.25] = intrinsic_scatter*3
        if method == 'sanders':
            Z_err[Z==7.9207] = intrinsic_scatter*3
        if method == 'naka':
            Z_err[Z==8.1] = intrinsic_scatter*3
        return Z,Z_err
    else:
        return Z


def h5_tree(val, pre=''):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ')
            else:
                print(pre + '└── ' + key + ' (%d)' % len(val))
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ')
            else:
                print(pre + '├── ' + key + ' (%d)' % len(val))