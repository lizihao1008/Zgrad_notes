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
# plt_param={'font.family':'serif',
#            'font.serif':'Times New Roman',
#            'lines.linewidth':1,
#            'xtick.labelsize':12,
#            'ytick.labelsize':12,
#            'axes.labelsize':16,
#            'xtick.direction':'in',
#            'ytick.direction':'in',
#            'font.style':'italic',
#            'font.weight':'normal',
#            'figure.figsize':[5,5],'xtick.minor.visible':True,
#            'ytick.minor.visible':True,
#            'xtick.major.size':4,
#            'ytick.major.size':4,
#            'xtick.minor.size':2,
#            'ytick.minor.size':2,
#            'xtick.major.width':.8,
#            'ytick.major.width':.8,
#            'xtick.top':True,
#            'ytick.right':True,
#            'axes.spines.bottom':True,
#            'axes.spines.top':True,
#            'axes.spines.left':True,
#            'axes.spines.right':True,
#            'xtick.bottom':True,
#            'xtick.labelbottom':True,
#            'ytick.left':True,
#            'ytick.labelleft':True}
def cut_down(num, c):
    c=10**(-c)
    return (num//c)*c

def cut_up(num, c):
    c=10**(-c)
    return (num//c)*c+c

def display_voronoi(x, y, binNum, pixelsize, z_2d,ax=False,fig=False,cmap='jet',auto_colorbar=True):

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    nx = int(np.round((xmax - xmin)/pixelsize) + 1)
    ny = int(np.round((ymax - ymin)/pixelsize) + 1)
    img = np.full((nx, ny), np.nan)  # use nan for missing data
    j = np.round((x - xmin)/pixelsize).astype(int)
    k = np.round((y - ymin)/pixelsize).astype(int)
#     _,_,bin_signal,noise_signal = voronoi_binned_data(x,y,flux,err,binNum)
#     img[j, k] = array([(bin_signal/noise_signal)[binNum[i]==np.unique(binNum)][0] for i in range(len(binNum))])
    img[j, k] = np.array([(z_2d)[binNum[i]==np.unique(binNum)][0] for i in range(len(binNum))])

    if ax != False:
        # im = ax.imshow(np.rot90(img), interpolation='nearest', cmap='jet',
        #        extent=[xmin - pixelsize/2, xmax + pixelsize/2,
        #                ymin - pixelsize/2, ymax + pixelsize/2],aspect='auto')
        img2 = img[~np.isnan(img)]
        im = ax.imshow(np.rot90(img), interpolation='nearest', cmap=cmap,
               extent=[xmin-1, xmax-1,ymin-1, ymax-1],aspect=1,vmin=cut_down(np.min(img2),1)-0.02,vmax=cut_up(np.max(img2),1)+0.02)
        # im = ax.imshow(np.rot90(img), interpolation='nearest', cmap='jet',
        #        extent=[xmin-1, xmax-1,ymin-1, ymax-1],aspect='auto')
        
        if auto_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.new_vertical(size='5%', pad=0)
            fig.add_axes(cax)
            cb = plt.colorbar(im,cax=cax,orientation='horizontal')

            cb.set_label(r'$12+\log(\rm{O}/\rm{H})$',fontdict={'size':10},labelpad=-40)

            cax.xaxis.set_major_locator(ticker.MaxNLocator(5))
            cb.ax.tick_params(labelsize=10,rotation=45)
            cax.xaxis.set_ticks_position('bottom')
            return ax,cax,cb
        else:
            return im,ax
    else:
        plt.imshow(np.rot90(img), interpolation='nearest', cmap=cmap,
                   extent=[xmin - pixelsize/2, xmax + pixelsize/2,
                           ymin - pixelsize/2, ymax + pixelsize/2],aspect=1)
        plt.colorbar()
        ax = plt.gca()
    return ax

# def kpc2pix(r,redshift):
#     pix = r/(cosmo.angular_diameter_distance(redshift).to('kpc').value*0.03*2*np.pi/360/3600)
#     return pix

def kpc2pix(r,redshift,pixsec):
    pix = r*cosmo.arcsec_per_kpc_proper(redshift).value/pixsec
    return pix


"""
Extract information from GALFIT output FITS file.
This is based on astronomeralex's galfit-python-parser:
    https://github.com/astronomeralex/galfit-python-parser
Modified by Song Huang to include more features
"""




class GalfitComponent(object):
    """Stores results from one component of the fit."""

    def __init__(self, galfitheader, component_number, verbose=True):
        """
        Read GALFIT results from output file.
        takes in the fits header from HDU 3 (the galfit model) from a
        galfit output file and the component number to extract
        """
        assert component_number > 0
        assert "COMP_" + str(component_number) in galfitheader

        self.component_type = galfitheader["COMP_" + str(component_number)]
        self.component_number = component_number
        headerkeys = [i for i in galfitheader.keys()]
        comp_params = []

        for i in headerkeys:
            if str(component_number) + '_' in i:
                comp_params.append(i)

        setattr(self, 'good', True)
        for param in comp_params:
            paramsplit = param.split('_')
            val = galfitheader[param]
            """
            we know that val is a string formatted as 'result +/- uncertainty'
            """
            if "{" in val and "}" in val:
                if verbose:
                    print(" ## One parameter is constrained !")
                val = val.replace('{', '')
                val = val.replace('}', '')
                val = val.split()
                if verbose:
                    print (" ## Param - Value : ", param, val)
                setattr(self, paramsplit[1].lower(), float(val[0]))
                setattr(self, paramsplit[1].lower() + '_err', np.nan)
            elif "[" in val and "]" in val:
                if verbose:
                    print(" ## One parameter is fixed !")
                val = val.replace('[', '')
                val = val.replace(']', '')
                val = val.split()
                if verbose:
                    print(" ## Param - Value : ", param, val)
                setattr(self, paramsplit[1].lower(), float(val[0]))
                setattr(self, paramsplit[1].lower() + '_err', np.nan)
            elif "*" in val:
                if verbose:
                    print(" ## One parameter is problematic !")
                val = val.replace('*', '')
                val = val.split()
                if verbose:
                    print(" ## Param - Value : ", param, val)
                setattr(self, paramsplit[1].lower(), float(val[0]))
                setattr(self, paramsplit[1].lower() + '_err', -1.0)
                setattr(self, 'good', True)
            else:
                val = val.split()
                setattr(self, paramsplit[1].lower(), float(val[0]))
                setattr(self, paramsplit[1].lower() + '_err', float(val[2]))


class GalfitResults(object):

    """
    This class stores galfit results information.
    Currently only does one component
    """

    def __init__(self, galfit_fits_file, hduLength=4, verbose=True):
        """
        Init method for GalfitResults.
        Take in a string that is the name of the galfit output fits file
        """
        hdulist = fits.open(galfit_fits_file)
        # Now some checks to make sure the file is what we are expecting
        assert len(hdulist) == hduLength
        galfitmodel = hdulist[hduLength - 2]
        galfitheader = galfitmodel.header
        galfit_in_comments = False
        for i in galfitheader['COMMENT']:
            galfit_in_comments = galfit_in_comments or "GALFIT" in i
        assert True == galfit_in_comments
        assert "COMP_1" in galfitheader
        # Now we've convinced ourselves that this is probably a galfit file

        self.galfit_fits_file = galfit_fits_file
        # Read in the input parameters
        self.input_initfile = galfitheader['INITFILE']
        self.input_datain = galfitheader["DATAIN"]
        self.input_sigma = galfitheader["SIGMA"]
        self.input_psf = galfitheader["PSF"]
        self.input_constrnt = galfitheader["CONSTRNT"]
        self.input_mask = galfitheader["MASK"]
        self.input_magzpt = galfitheader["MAGZPT"]

        # Fitting region
        fitsect = galfitheader["FITSECT"]
        fitsect = re.findall(r"[\w']+", fitsect)
        self.box_x0 = fitsect[0]
        self.box_x1 = fitsect[1]
        self.box_y0 = fitsect[2]
        self.box_y1 = fitsect[3]

        # Convolution box
        convbox = galfitheader["CONVBOX"]
        convbox = convbox.split(",")
        self.convbox_x = convbox[0]
        self.convbox_y = convbox[1]

        # Read in the chi-square value
        self.chisq = galfitheader["CHISQ"]
        self.ndof = galfitheader["NDOF"]
        self.nfree = galfitheader["NFREE"]
        self.reduced_chisq = galfitheader["CHI2NU"]
        self.logfile = galfitheader["LOGFILE"]

        # Find the number of components
        num_components = 1
        while True:
            if "COMP_" + str(num_components + 1) in galfitheader:
                num_components = num_components + 1
            else:
                break
        self.num_components = num_components

        for i in range(1, self.num_components + 1):
            setattr(self, "component_" + str(i),
                    GalfitComponent(galfitheader, i, verbose=verbose),
                    )

        hdulist.close()
import os  
def load_data(source_id,zpath='output_vor_30pix/',full_path='../j0305/Extractions/',
              root='j0305',method='wx',use_pass_flag=True,load_lines=['OIII','OII','Hb','Hg'],w=15,seg_sigma=3):
    # sed_paras = Table.read('sedcat_merge_210623_2.fits')
    metallicity = []
    sfr = []
    sfr_err_upper = []
    sfr_err_lower = []

    mass = []
    mass_err_upper = []
    mass_err_lower = []

    z_table_vor = []
    z_2d_vor = []
    z_2d_up1_vor = []
    z_2d_low1_vor = []
    z_2d_std = []
    Elines_id =[]
    for id_name in source_id:

#         sed_idx = sed_paras['ID']==id_name
#         sfr.append(sed_paras[sed_idx]['sfr_50'][0])
#         # sfr.append(sed_paras[sed_idx]['sfr_50'])

#         sfr_err_upper.append(sed_paras[sed_idx]['sfr_84']-sfr[-1])
#         sfr_err_lower.append(sfr[-1]-sed_paras[sed_idx]['sfr_16'])
#         mass.append(sed_paras[sed_idx]['stellar_mass_50'])
#         mass_err_upper.append(sed_paras[sed_idx]['stellar_mass_84']-mass[-1])
#         mass_err_lower.append(mass[-1]-sed_paras[sed_idx]['stellar_mass_16'])

        # z = np.load(zpath+'%s_%s_oh12_2d_id%s.npy'%(root,method,id_name),allow_pickle=True).item()
        z = np.load(zpath+'%s_oh12_2d_id%s.npy'%(root,id_name),allow_pickle=True).item()

        if method =='wx':
            if use_pass_flag:
                passflag = np.array(z['passflag'])
            else:
                passflag = np.ones_like(np.array(z['me']))
            passflag = np.where(passflag==0,np.nan,passflag)
            z_table_vor.append(z)
            z_2d_vor.append(np.array(z['me']).copy()*passflag) # median
            # z_2d_vor.append(np.array(z['bf']).copy()*passflag) # best fitting
            z_2d_up1_vor.append(np.array(z['me_up1sig']).copy()*passflag)
            z_2d_low1_vor.append(np.array(z['me_low1sig']).copy()*passflag)
            z_2d_std.append(np.array(z['me_std']).copy()*passflag)
            
        elif method=='bian':
            z_table_vor.append(z)
            z_2d_std.append(np.array(z['Z_err']).copy())
            z_2d_vor.append(np.array(z['Z']).copy())

        hdu = fits.open(os.path.join(full_path,'%s_%s.full.fits'%(root,id_name.zfill(5))))
    
        ra = hdu[0].header['RA']
        dec = hdu[0].header['DEC']
        redshift = hdu[0].header['redshift']

        Elines = {}
        # w = 15

        wcs_h = WCS(hdu[5].header)
        c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
        x = wcs_h.world_to_pixel(c)[0]
        y = wcs_h.world_to_pixel(c)[1]

        # seg = hdu['seg'].data.astype(float)
        img = hdu['dsci'].data
        # mean, _, std = sigma_clipped_stats(img)
        # kernel = make_2dgaussian_kernel(2, size=5)  # FWHM = 3.
        # convolved_data = convolve(img, kernel)
#         seg = detect_sources(convolved_data, seg_sigma*std, npixels=5).data.astype(float)
        
#         seg = Cutout2D(seg,(x,y),(2*w,2*w)).data
#         seg[seg!=seg[len(seg)//2,len(seg)//2]] = np.nan
#         seg /= seg[len(seg)//2,len(seg)//2]
        seg = 1

        for line in load_lines:
            try:
                cutout = Cutout2D(hdu['line',line].data, (x,y), (2*w,2*w), wcs=wcs_h)
                EL_img = cutout.data
                EL_img_sm = ndimage.gaussian_filter(EL_img,3.333/2.355)

                Elines['flux_'+line] = (seg*EL_img_sm).flatten()

                cutout = Cutout2D(hdu['linewht',line].data, (x,y), (2*w,2*w), wcs=wcs_h)
                err_img = 1/(cutout.data)**0.5
                Elines['err_'+line] = (seg*err_img).flatten()
            except:
                print(f'Line {line} not found!')
                Elines['flux_'+line] = None
                Elines['err_'+line] = None
                
        
        cutout = Cutout2D(img, (x,y), (2*w,2*w), wcs=wcs_h)
        Elines['img'] = cutout.data
        Elines_id.append(Elines)
        
    mass_set = {'m':mass,'errl':mass_err_lower,'erru':mass_err_upper}
    sfr_set = {'sfr':sfr,'errl':sfr_err_lower,'erru':sfr_err_upper}

    h_stamp = []
    galfit = []
#     for i, n in enumerate(source_id):
#         gal = GalfitResults("A2744-ID%s.galfit.fits"%n,verbose=False)

#         x_center = gal.component_1.xc-52
#         y_center = gal.component_1.yc-52
#         R_e = gal.component_1.re
#         ar = gal.component_1.ar
#         angle = gal.component_1.pa

#         a = R_e
#         b = R_e*ar
#         galfit.append(np.array([x_center,y_center,a,b,angle,ar]))

#         cutout = Cutout2D(hdu[5].data, (x,y), (2*w,2*w), wcs=wcs_h)
#         h_stamp.append(cutout.data)
#         hdu.close()

        
    mass_set, sfr_set = None,None
    return z_table_vor, z_2d_vor, z_2d_up1_vor, z_2d_low1_vor, z_2d_std, h_stamp, galfit, redshift, Elines_id, mass_set, sfr_set

# EL = ['flux_OIII','flux_Hb','flux_Hg','flux_OII']

def pix2arcsec(x, pos,pixsec=0.03):

    # return f'{0.06*(x-15)+0.03:.1f}'
    return f'{pixsec*(x-15)+pixsec*0.5:.1f}'


pix2arcsec = FuncFormatter(pix2arcsec)

from random import choice, gauss
from numpy import polyfit

def split_normal(mus, sigmas_u68, sigmas_l68):
    """
    RET: A split-normal value.
    """

    split_normal = []

    for mu, sigma_u68, sigma_l68 in zip(mus, sigmas_u68, sigmas_l68):

        sigma = choice([sigma_u68, -sigma_l68])
        g = abs(gauss(0.0, 1.0)) * sigma + mu
        split_normal.append(g)

    return split_normal

def errors_84_16(x):
    """
    RET: 1-sigma upper/lower error bars from the 84/16th percentile
         from the median.
  index_med  """

    n = len(x)

    index_med = n // 2 # median.
    index_84 = int(round(n * 0.84135)) # 84th percentile from median.
    index_16 = int(round(n * 0.15865))

    x_sorted = sorted(x)
    x_med = x_sorted[index_med]
    x_u68 = x_sorted[index_84] - x_med # 1-sigma upper error.
    x_l68 = x_med - x_sorted[index_16] # 1-sigma lower error.

    return x_med, x_u68, x_l68


def assymetric_polyfit(x, y, y_u68, y_l68, n_mc=2000):
    """
    DES: Solves y = a + b * x for assymentric y error bars.
    RET: [a, a_u68, a_l68, b, b_u68, b_l68].
    """

    a_mc = []
    b_mc = []

    for i in range(0, n_mc):
        y_mc = split_normal(y, y_u68, y_l68)
        pars = polyfit(x, y_mc, 2)
        a_mc.append(pars[2])
        b_mc.append(pars[1])

    a, a_u68, a_l68 = errors_84_16(a_mc)
    b, b_u68, b_l68 = errors_84_16(b_mc)

    return a, a_u68, a_l68, b, b_u68, b_l68


def plot_result(source_id, fit='scipy', output=False, h_cmap='pink',
                EL_cmap=['Blues', 'Greens', 'Oranges', 'Purples'],
                savepath=False, mask_kpc=False, pixsec=0.03, zpath='output_vor_30pix/'):

    z_table_vor, z_2d_vor, z_2d_up1_vor, z_2d_low1_vor, z_2d_std, h_stamp, galfit, redshift, Elines_id, mass_set, sfr_set = load_data(source_id,zpath)

    slope_vor = []
    slope_vor_err = []

    # vorbins = []
    for i, id_name in enumerate(source_id):

        fig = plt.figure(figsize=(20,11))
        gs = gridspec.GridSpec(2, 4)

        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[0, 2:])
        ax4 = plt.subplot(gs[1, 0])
        ax5 = plt.subplot(gs[1, 1])
        ax6 = plt.subplot(gs[1, 2])
        ax7 = plt.subplot(gs[1, 3])
        plt.subplots_adjust(wspace=0.01, hspace=0.25)
        ax_EL = [ax4,ax5,ax6,ax7]
        ax_all = [ax1,ax2,ax3,ax4,ax5,ax6,ax7]

        for axi in ax_all:
            axi.tick_params(which='both',direction='in',top=True,right=True,left=True,labelsize=20)
        ax2.tick_params(top=False)
        ax3.yaxis.tick_right()
        ax3.yaxis.set_label_position("right")
        ax3.grid(ls='--')
        ax1.imshow(h_stamp[i],origin='lower',cmap=h_cmap,aspect='auto')
        ax1.set_ylabel('arcsec',fontsize=20)
        ax1.set_title('Direct',fontsize=20)

        ell1 = Ellipse(xy=(galfit[i][:2]-1), width=2*galfit[i][2], height=2*galfit[i][3],
                       angle=90+galfit[i][4], ec='k', fill=False, alpha=1.0, ls='-.')

        ax1.add_artist(ell1)

        # voronoi results
        binNum, x, y = z_table_vor[i]['binNum'].copy(),z_table_vor[i]['xNode'].copy(),z_table_vor[i]['yNode'].copy()
        x_display = z_table_vor[i]['x_display']
        y_display = z_table_vor[i]['y_display']
        display_voronoi(x_display,y_display,binNum,1,z_2d_vor[i],ax2,fig)
        x -= galfit[i][0]
        y -= galfit[i][1]
        coord = np.array([x.flatten(),y.flatten()])

        angle = -(90+galfit[i][4])*2*np.pi/360
        ab = galfit[i][3]/galfit[i][2]
        trans1 = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        trans2 = np.array([[1,0],[0,1/ab]])
        mat_trans = trans2@trans1

        x,y=mat_trans@coord
        d = (x**2+y**2)**0.5

        # d = cosmo.angular_diameter_distance(redshift).to('kpc').value*d*0.03*2*np.pi/360/3600
        d = d/cosmo.arcsec_per_kpc_proper(redshift).value*pixsec

        if mask_kpc:
            mask_idx = d>mask_kpc
            d = d[mask_idx]
            z_2d_vor[i] = z_2d_vor[i][mask_idx]
            z_2d_low1_vor[i] = z_2d_low1_vor[i][mask_idx]
            z_2d_up1_vor[i] = z_2d_up1_vor[i][mask_idx]
            z_2d_std[i] = z_2d_std[i][mask_idx]
        # mask_idx = d<2.2
        # d = d[mask_idx]
        # z_2d_vor[i] = z_2d_vor[i][mask_idx]
        # z_2d_low1_vor[i] = z_2d_low1_vor[i][mask_idx]
        # z_2d_up1_vor[i] = z_2d_up1_vor[i][mask_idx]

            
        # ax3.errorbar(d,z_2d_vor[i],yerr=(z_2d_vor[i]-z_2d_low1_vor[i],z_2d_up1_vor[i]-z_2d_vor[i]),
        #                fmt='d',ecolor='k',color='r',elinewidth=1,capsize=2,alpha=0.6,markersize=10)
        ax3.errorbar(d,z_2d_vor[i],yerr=z_2d_std[i],
                       fmt='d',ecolor='k',color='r',elinewidth=1,capsize=2,alpha=0.6,markersize=10)
        # ax3.errorbar(d,z_2d_vor[i],yerr=(z_2d_vor[i]-z_2d_low1_vor[i],z_2d_up1_vor[i]-z_2d_vor[i]),
        #                )
        # Fit linear regressing to metallicity - radius
        x = d
        y = z_2d_vor[i].flatten()
        weight = (2/(z_2d_up1_vor[i]-z_2d_low1_vor[i]))**2


        # ysig = (z_2d_up1_vor[i]-z_2d_low1_vor[i])/2
        ysig = z_2d_std[i]
        ysig_u = z_2d_up1_vor[i] - y
        ysig_l = y - z_2d_low1_vor[i]
        
        x = x[~np.isnan(weight)]
        y = y[~np.isnan(weight)]
        ysig_u = ysig_u[~np.isnan(ysig_u)]
        ysig_l = ysig_l[~np.isnan(ysig_l)]
        ysig = ysig[~np.isnan(weight)]
        weight = weight[~np.isnan(weight)]

        xmean = x.mean()
        if fit == 'scipy':
            def linear(x,k,b):
                return k*x+b
            
            x -= xmean
            popt, pcov = curve_fit(linear, x, y, sigma=ysig, absolute_sigma=True)
            # popt, pcov = curve_fit(linear, x, y, sigma=ysig)
            x += xmean
            perr = np.sqrt(np.diag(pcov))
            k, b = popt
            b -= k*xmean
            kerr = perr[0]

        elif fit == 'linmix':
            x -= xmean
            lm = linmix.LinMix(x, y, xsig=None, ysig=ysig)
            lm.run_mcmc(silent=True,maxiter=1000)
#             b = lm.chain['alpha'].mean()
#             k = lm.chain['beta'].mean()
            x += xmean
            
            b = np.percentile(lm.chain['alpha'],50)
            k = np.percentile(lm.chain['beta'],50)
            b -= k*xmean
            kerr = lm.chain['beta'].std()
            kerr_lower = k - np.percentile(lm.chain['beta'], 16)
            kerr_upper = np.percentile(lm.chain['beta'], 84) - k

        elif fit == 'wls':

            x = sm.add_constant(x)
            model = sm.WLS(y, x, weight).fit()
            b, k = model.params
            kerr = model.bse[1]

        elif fit == 'mc':

            b, b_u, b_l, k, kerr_upper, kerr_lower = assymetric_polyfit(
                x, y, ysig_u, ysig_l)

        ax3.plot(np.array([0, x.max()]), np.array([b, b+k*x.max()]), 'r--')

        if (fit == 'linmix') | (fit == 'mc'):
            ax3.text(0.02, 0.91, r'$\Delta\log(\rm{O}/\rm{H})/\Delta r=%.3f^{+%.3f}_{-%.3f}$'%(k, kerr_upper, kerr_lower),
                     transform=ax3.transAxes, fontsize=16)
        else:
            ax3.text(0.02, 0.91, r'$\Delta\log(\rm{O}/\rm{H})/\Delta r=%.3f\pm%.3f$'%(k,kerr),
                     transform=ax3.transAxes, fontsize=16)
        ax3.set_xlabel('radius (kpc)', fontsize=18)
        ax3.set_ylabel(r'$12+\log(\rm{O}/\rm{H})$', fontsize=17, labelpad=0.01)
#         plt.suptitle(source_id[i])
        ax1.text(0,27, 'A2744-ID '+id_name, size=18, color='white')

        # ax1.text(0,3,r'$\log(\rm{O}/\rm{H})=%.2f$'%metallicity[i],size=18,color='white')
        # ax1.text(0,0,r'SFR$=%.2f^{+%.2f}_{-%.2f}\ M_\odot/$yr'%(sfr_set['sfr'][i],sfr_set['erru'][i],sfr_set['errl'][i]),size=18,color='white')
        # ax1.text(0,3,r'$\log(M_*/M_\odot)=%.2f^{+%.2f}_{-%.2f}$'%(mass_set['m'][i],mass_set['erru'][i],mass_set['errl'][i]),size=18,color='white')



        ell1 = Ellipse(xy = galfit[i][:2]-1, width = 2*galfit[i][2], height = 2*galfit[i][3], angle = 90+galfit[i][4], ec= 'k',fill=False, alpha=1.0,ls='-.')
        ax2.add_artist(ell1)
        b = 6.666666666666667

        EL_name = [r'$\left[\rm{O III}\right]\lambda5008$',
                   r'$[\rm{O II}]\lambda\lambda3726,3729$',
                   r'$\rm{H}\beta$', r'$\rm{H}\gamma$']

        for j, EL in enumerate(['OIII', 'OII', 'Hb', 'Hg']):
            
            img = Elines_id[i]['flux_'+EL]/pixsec**2/100
            img2 = img[~np.isnan(img)]
            # im = ax_EL[j].imshow(img, origin='lower',
            #                      cmap=EL_cmap[j], aspect='auto',
            #                     vmin=cut_down(np.min(img2), 3)+10**np.floor(np.log10(abs(np.min(img2)))),
            #                      vmax=cut_up(np.max(img2), 3)+10**np.floor(np.log10(np.max(img2)-int(np.max(img2)))))
            
            im = ax_EL[j].imshow(img, origin='lower',
                                 cmap=EL_cmap[j], aspect='auto',
                                vmin=cut_down(np.min(img2), 3)-0.01,
                                 vmax=cut_up(np.max(img2), 3)+0.01)
            ax_EL[j].text(0, 0, EL_name[j], fontsize=18)

            divider = make_axes_locatable(ax_EL[j])
            cax = divider.new_vertical(size='5%', pad=0)
            fig.add_axes(cax)
            cb = plt.colorbar(im, cax=cax, orientation='horizontal')
            # cb.set_label(r'$S\ \rm [10^{-15}\ erg\ s^{-1}\ cm^{-2}\ arcsec^{-2}]$',
            #              fontdict={'size': 15}, labelpad=-75)
            cax.set_title(r'$S\ \rm [10^{-15}\ erg\ s^{-1}\ cm^{-2}\ arcsec^{-2}]$',fontsize=16)
            cax.xaxis.set_major_locator(ticker.MaxNLocator(5))
            cb.ax.tick_params(labelsize=14, rotation=45)
            cax.xaxis.set_ticks_position('bottom')
            ax_EL[j].xaxis.set_ticks_position('bottom')
        for radi in range(1, 6):
            ell = Ellipse(xy = galfit[i][:2]-1, width = 2*kpc2pix(radi,redshift,pixsec), height = 2*kpc2pix(radi,redshift,pixsec)*galfit[i][5], angle = 90+galfit[i][4], ec= 'k',fill=False, alpha=0.8)
            ax2.add_artist(ell)

            ell = Ellipse(xy = galfit[i][:2]-1, width = 2*kpc2pix(radi,redshift,pixsec), height = 2*kpc2pix(radi,redshift,pixsec)*galfit[i][5], angle = 90+galfit[i][4], ec= 'k',fill=False, alpha=0.8)
            ax1.add_artist(ell)

        if mask_kpc:
            ell = Ellipse(xy = galfit[i][:2]-1, width = 2*kpc2pix(mask_kpc,redshift,pixsec), height = 2*kpc2pix(mask_kpc,redshift,pixsec)*galfit[i][5], angle = 90+galfit[i][4], ec= 'k',fill=True,hatch='xxx', alpha=1,color='white')
            ax2.add_artist(ell)

        arcsec_loc = ticker.FixedLocator([14.5-2*b,14.5-b,14.5,14.5+b,14.5+2*b])

        for axi in [ax1,ax4,ax5,ax6,ax7]:
            axi.yaxis.set_major_locator(arcsec_loc)
            axi.yaxis.set_major_formatter(pix2arcsec)

        # ax1.yaxis.set_major_locator(arcsec_loc)
        # ax1.yaxis.set_major_formatter(pix2arcsec)

        ax4.set_ylabel('arcsec',fontsize=20)


        # ax2.xaxis.set_major_locator(arcsec_loc)
        # ax2.xaxis.set_major_formatter(pix2arcsec)        
        for axi in ax_EL:
            axi.set_xlim(-0.5,29.5)
            axi.set_ylim(-0.5,29.5)
            
            axi.xaxis.set_major_locator(arcsec_loc)
            axi.xaxis.set_major_formatter(pix2arcsec)
            
            axi.set_xlabel('arcsec',fontsize=20)
            ell = Ellipse(xy = galfit[i][:2]-1, width = 2*galfit[i][2], height = 2*galfit[i][3], angle = 90+galfit[i][4], ec= 'k',fill=False, alpha=1.0,ls='-.')
            axi.add_artist(ell)
            for radi in range(1,6):
                ell = Ellipse(xy = galfit[i][:2]-1, width = 2*kpc2pix(radi,redshift,pixsec), height = 2*kpc2pix(radi,redshift,pixsec)*galfit[i][5], angle = 90+galfit[i][4], ec= 'k',fill=False, alpha=0.8)
                axi.add_artist(ell)

        for axi in ax_EL[1:]:
            axi.yaxis.set_ticklabels([])
            
        for axi in [ax2]:
            axi.set_xlim(-0.5,29.5)
            axi.set_ylim(-0.5,29.5)
            axi.xaxis.set_major_locator(arcsec_loc)
            axi.yaxis.set_major_locator(arcsec_loc)
            
        ax2.yaxis.set_ticklabels([])
        ax2.xaxis.set_ticklabels([])
        ax1.xaxis.set_ticklabels([])
        slope_vor.append(k)
        slope_vor_err.append(kerr)
        if savepath:
            plt.savefig(savepath+'zgrad_ID%s.pdf'%id_name,bbox_inches='tight')
        plt.show()

    slope_vor = np.array(slope_vor)
    slope_vor_err = np.array(slope_vor_err)

    if output:
        return slope_vor, slope_vor_err
    else:
        return


def wtd(x,w):
    return np.sum(x*w)/np.sum(w)


def std_err_wtd(x, w, method='case1'):

    if method == 'case1':
        return ((np.sum(w*x**2)/np.sum(w)-wtd(x, w)**2)/(len(x)-1))**0.5
    else:
        return ((np.sum(w*x**2)/np.sum(w)-wtd(x,w)**2)*np.sum(w**2)/(np.sum(w)**2-np.sum(w**2)))**0.5

from statsmodels.stats.weightstats import DescrStatsW

def binned_mass(x,y,err,bins=11,xrange=(6,11.5)):
    x = np.asarray(x)
    y = np.asarray(y)
    bin_means, bin_edges, binnumber = binned_statistic(x,y,statistic='mean', bins=bins,range=xrange)
#     avg_err_list = np.zeros(len(bin_means))
#     weight_avg_list = np.zeros(len(bin_means))
    avg_err_list = []
    weight_avg_list = []
    x_list = []
#     for i,mean in enumerate(bin_means):
#         if np.isnan(mean):
#             avg_err_list[i] = np.nan
#             weight_avg_list[i] = np.nan

    for num in np.unique(binnumber):
        idx = np.where(binnumber==num)
        bin_x_mean = np.mean(x[idx])
        bin_y = y[idx]
        bin_err = err[idx]

        if len(bin_y)>1:
            weight = 1/bin_err**2
            # weighted_stats = DescrStatsW(bin_y, weights=weight, ddof=0)
            # weight_avg = weighted_stats.mean
            # avg_err = weighted_stats.std_mean
            
            weight_avg  = wtd(bin_y,weight)
            avg_err = std_err_wtd(bin_y,weight)
        elif len(bin_y)==1:
            avg_err = bin_err[0]
            weight_avg = bin_y[0]
        else:
            avg_err = np.nan
            weight_avg = np.nan
            print('nan')
        avg_err_list.append(avg_err)
        weight_avg_list.append(weight_avg)
        x_list.append(bin_x_mean)
    return x_list,weight_avg_list, avg_err_list,bin_edges



from astropy.stats import bootstrap

def binned_data(x,y,yerr_up_orig,yerr_low_orig,bins):
    x = np.asarray(x)
    y = np.asarray(y)
    bin_means, bin_edges, binnumber = binned_statistic(x,y,statistic='mean', bins=bins)

    err_up_list = []
    err_low_list = []
    x_list = []

    for i, num in enumerate(np.unique(binnumber)):
        idx = np.where(binnumber==num)
        bin_x_mean = np.mean(x[idx])
        bin_y = y[idx]

        if len(bin_y)>1:
            y_bs = bootstrap(bin_y,bootnum=500,bootfunc=np.mean)
            bin_p_16,bin_p_84 = np.percentile(y_bs,[16,84])
            yerr_up = bin_p_84 - bin_means[i]
            yerr_low = bin_means[i] - bin_p_16

        else:
            yerr_up = yerr_up_orig[idx][0]
            yerr_low = yerr_low_orig[idx][0]

        x_list.append(bin_x_mean)
        err_up_list.append(yerr_up)
        err_low_list.append(yerr_low)

    return x_list,bin_means, err_up_list,err_low_list

def voronoi_binned_data(x,y,signal,noise,binNum):
    signal_bin = []
    noise_bin = []
    x_center = []
    y_center = []
    for i in np.unique(binNum):
        idx = binNum==i

        if len(signal[idx]) >1:
            signal_bin.append(np.sum(signal[idx]))
            noise_bin.append(np.sqrt(np.sum(noise[idx]**2)))
            x_center.append(x[idx].mean())
            y_center.append(y[idx].mean())
        else:
            signal_bin.append(signal[idx][0])
            noise_bin.append(noise[idx][0])
            x_center.append(x[idx][0])
            y_center.append(y[idx][0])    
    return np.array(x_center),np.array(y_center),np.array(signal_bin),np.array(noise_bin)

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

# def O2metallicity(ratio):
    

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
    
    
def analysis_zgrad(ID,path,zgrad_hf,fit='scipy',use_passflag=False,show=True,
                   ax_list=None,fig=None,show_radi=True,
                   show_rand_lines=False,auto_colorbar=False,radii_fit=10,
                   z_col='me',force_redshift=6,sigma_upper=5,sigma_lower=5,cmap='jet'):
    
    try:
        param_grp = zgrad_hf[path]
        this_data = param_grp['sources'][ID]['zgrad']
    except:
        # deprecated
        this_data = zgrad_hf['zgrad'][ID]

    
    if use_passflag:
        passflag = np.array(this_data['passflag'])
    else:
        passflag = np.ones_like(this_data['me'])
    passflag = np.where(passflag==0,np.nan,passflag)
    
    binNum, x, y = this_data.attrs['binNum'],this_data.attrs['xNode'].copy(),this_data.attrs['yNode'].copy()
    x_display = this_data.attrs['x_display']
    y_display = this_data.attrs['y_display']
    w = zgrad_hf[path].attrs['w']
    z_2d = this_data[z_col]*passflag
    z_2d_std = this_data['me_std']*passflag

    if 'zgrad_input' in param_grp.keys():
        zgrad_table = read_table_hdf5(param_grp['zgrad_input'])
        zgrad_table.add_index('id')
        try:
            this_row = zgrad_table.loc[int(ID.split('-')[1])]
        except:
            this_row = zgrad_table.loc[int(ID)]
        redshift = this_row['redshift']
        x_0 = this_row['x_0']
        y_0 = this_row['y_0']
        ellip = this_row['ellip']
        theta = this_row['theta']
        r_eff = this_row['r_eff']
        a, b = r_eff, (1 - ellip) * r_eff
    else:
        print('No input sersic info found. Assuming centering face-on morphology.')
        x_0 = w
        y_0 = w
        theta = 0
        redshift = force_redshift
        a = 1
        b = 1
    
    if show:
        if ax_list is None:
            fig,ax = plt.subplots(1,2,figsize=(10.5,4),gridspec_kw={
                                'width_ratios': [1, 1.5]})
            plt.subplots_adjust(wspace=0.05)
        else:
            ax = ax_list
            fig = fig
    
        def pix2arcsec(x, pos,pixsec=0.03):

            return f'{pixsec*(x-w)+pixsec*0.5:.1f}'

        pix2arcsec = ticker.FuncFormatter(pix2arcsec)
        
        # ax[0].axis('equal')
        # ax[0].set_aspect('equal')
        display_voronoi(x_display,y_display,binNum,1,z_2d,ax=ax[0],fig=fig,auto_colorbar=auto_colorbar,cmap=cmap)

        
        ax[0].set_xlim(-0.5,w*2-0.5)
        ax[0].set_ylim(-0.5,w*2-0.5)
    
        if show_radi:
            for radi in np.arange(1, 4,1):
                i = 0
                ell = Ellipse(xy = (x_0,y_0), width = 2*kpc2pix(radi,redshift,0.03), 
                            height = 2*kpc2pix(radi,redshift,0.03)*b/a, 
                            angle = np.rad2deg(theta), ec= 'k',fill=False, alpha=0.8,linestyle='--')
                ax[i].add_artist(ell)

    pixsec = 0.03
    trans1 = np.array([[np.cos(-theta),-np.sin(-theta)],[np.sin(-theta),np.cos(-theta)]])
    trans2 = np.array([[1,0],[0,a/b]])
    mat_trans = trans2@trans1

    x -= x_0+0.5
    y -= y_0+0.5
    coord = np.array([x.flatten(),y.flatten()])

    x,y=mat_trans@coord
    
    d = (x**2+y**2)**0.5
    d = d/cosmo.arcsec_per_kpc_proper(redshift).value*pixsec
    
    idx = ~np.isnan(z_2d)

    try:
        x = d[idx]
        radius_cut = x < radii_fit

        x = x[radius_cut]
        y = z_2d[idx][radius_cut]
        

        if len(y)<10:
            print('too small points to fit!')
            return np.nan,np.nan,np.nan
        
        mask = sigma_clip(x,sigma_upper=sigma_upper,sigma_lower=50).mask
        mask |= sigma_clip(y,sigma_upper=sigma_upper,sigma_lower=sigma_lower).mask

        ysig = z_2d_std[idx][radius_cut]
        ysig = ysig[~mask]
        x = x[~mask]
        y = y[~mask]
        if show:
            ax[1].errorbar(x,y,yerr=ysig,
                    fmt='d',ecolor='gray',color='blue',elinewidth=1,capsize=1,alpha=0.6,markersize=4)
        xmean = x.mean()
        
        if fit == 'scipy':
            def linear(x,k,b):
                return k*x+b
            
            x -= xmean
            popt, pcov = curve_fit(linear, x, y, sigma=ysig, absolute_sigma=True)
            # popt, pcov = curve_fit(linear, x, y, sigma=ysig)
            x += xmean
            perr = np.sqrt(np.diag(pcov))
            k, b = popt
            # b -= k*xmean
            kerr = perr[0]
            berr = perr[1]
            x_plot = np.array([0,x.max()])
            if show:
                ax[1].plot(x_plot, b+k*(x_plot-xmean), 'r--')

                # ax[1].text(0.02, 0.91, r'$\Delta\log(\rm{O}/\rm{H})/\Delta r=%.3f\pm%.3f$'%(k, kerr),
                #                      transform=ax[1].transAxes, fontsize=16)
                ax[1].text(0.99, 0.01, r'$\nabla Z=%.3f\pm%.3f \ {\rm dex/kpc}$'%(k, kerr),
                                    transform=ax[1].transAxes, fontsize=12,
                        verticalalignment='bottom',horizontalalignment='right')
                if show_rand_lines:
                    k_sample = np.random.randn(100)*kerr+k
                    b_sample = np.random.randn(100)*berr+b
                    x_plot = np.tile(np.array([0,x.max()]),(100,1)).T
                    ax[1].plot(x_plot, np.array(b_sample).reshape(1,-1)+np.array(k_sample).reshape(1,-1)*(x_plot-xmean), alpha=0.05,c='m')
            
        else:
            x -= xmean
            lm = linmix.LinMix(x, y, xsig=None, ysig=ysig)
            lm.run_mcmc(silent=True,maxiter=500)
            #             b = lm.chain['alpha'].mean()
            #             k = lm.chain['beta'].mean()
            x += xmean

            b = np.percentile(lm.chain['alpha'],50)
            k = np.percentile(lm.chain['beta'],50)
            b -= k*xmean
            kerr = lm.chain['beta'].std()
            kerr_lower = k - np.percentile(lm.chain['beta'], 16)
            kerr_upper = np.percentile(lm.chain['beta'], 84) - k
            if show:
                ax[1].plot(np.array([0, x.max()]), np.array([b, b+k*x.max()]), 'r--')

                # ax[1].text(0.02, 0.91, r'$\Delta\log(\rm{O}/\rm{H})/\Delta r=%.3f^{+%.3f}_{-%.3f}$'%(k, kerr_upper, kerr_lower),
                #                      transform=ax[1].transAxes, fontsize=12)
                ax[1].text(0.99, 0.01, r'$\nabla Z=%.3f^{+%.3f}_{-%.3f} \ {\rm dex/kpc}$'%(k, kerr_upper, kerr_lower),
                        transform=ax[1].transAxes, fontsize=12,
                        verticalalignment='bottom',horizontalalignment='right')
                if show_rand_lines:
                    b = random.choices(lm.chain['alpha'], k=100)
                    k = random.choices(lm.chain['beta'], k=100)
                
                    kerr = lm.chain['beta'].std()
                    kerr_lower = k - np.percentile(lm.chain['beta'], 16)
                    kerr_upper = np.percentile(lm.chain['beta'], 84) - k

                    berr_lower = k - np.percentile(lm.chain['alpha'], 16)
                    berr_upper = np.percentile(lm.chain['alpha'], 84) - k

                    x_plot = np.tile(np.array([0,x.max()]),(100,1)).T
                    ax[1].plot(x_plot, np.array(b).reshape(1,-1)+np.array(k).reshape(1,-1)*(x_plot-xmean), alpha=0.05,c='m')
        if show:  
            ax[1].set_xlabel('r (kpc)',fontsize=12)
            ax[1].set_ylabel(r'$12+\log(\rm{O}/\rm{H})$',fontsize=12)
            ax[1].yaxis.set_label_position("right")
            ax[1].yaxis.tick_right()
    except Exception as e: 
        print(e)
        k = np.nan
        b = np.nan
        kerr = np.nan
        
    return k,b,kerr


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