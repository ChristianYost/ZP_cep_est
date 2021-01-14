import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import scipy.signal as sp
import scipy
from scipy.io import wavfile as wavfile
import random
import librosa
import numpy.polynomial.polynomial as poly
import os
import argparse
from scipy.spatial import distance_matrix

#return filter coefficients from the zero pole locations 
#here we don't include conjugates (maybe change this later)
def get_coefficients(zeros = [], poles = []):
    #zeros of the form zeros = [[z1_r,z1_a],[z2_r,z2_a], ...] similarly for poles
    z = sym.Symbol('z')
    n_z = len(zeros); n_p = len(poles)
    zfactors = [];pfactors = []
    n_cough = [];d_cough = []
    if zeros:
        for i in range(n_z):
            zfactors.append(z-zeros[i][0]*np.exp(1j*zeros[i][1]))
            zfactors.append(z-zeros[i][0]*np.exp(-1j*zeros[i][1])) #conjugate
        numer = sym.simplify(sym.expand(np.prod(zfactors))/z**(n_z*2))
        for i in range(n_z*2+1):
            n_cough.append(numer.sort_key()[1][1][i][-1])
    if poles:
        for i in range(n_p):
            pfactors.append(z-poles[i][0]*np.exp(1j*poles[i][1]))
            pfactors.append(z-poles[i][0]*np.exp(-1j*poles[i][1])) #conjugate
        denom = sym.simplify(sym.expand(np.prod(pfactors))/z**(n_p*2))
        for i in range(n_p*2+1):
            d_cough.append(denom.sort_key()[1][1][i][-1])

    return n_cough, d_cough

#compute complex cepstrum using fourier transform and phase unwrapping
def complex_cepstrum(x, win=False, n=None):
    
    if win:
        window = np.hanning(len(x))
    else:
        window = np.ones_like(x)
    
    def _unwrap(phase):
        samples = phase.shape[-1]
        unwrapped = np.unwrap(phase)
        center = (samples + 1) // 2
        if samples == 1:
            center = 0
        ndelay = np.array(np.round(unwrapped[..., center] / np.pi))
        unwrapped -= np.pi * ndelay[..., None] * np.arange(samples) / center
        return unwrapped, ndelay
    
    spectrum = np.fft.fft(window*x, n=n)
    unwrapped_phase, ndelay = _unwrap(np.angle(spectrum))
    log_spectrum = np.log(np.abs(spectrum)+1e-12) + 1j * unwrapped_phase
    ceps = np.fft.ifft(log_spectrum).real

    return ceps, ndelay

def real_mode_sum(n, modes):
    '''
    outputs the summation term in the analytic expression for the complex cepstrum given its 
    zero-poles
    modes: [[radius, angle], [radius,angle], ...]
    '''
    n_m = len(modes)
    term = 0
    for i in range(n_m):
        term = term + modes[i][0]**n*np.cos(modes[i][1]*n)
    
    return term / n

 
def cepstrum_expression(N, zeros = None, poles = None):
    '''
    compute the complex cepstrum from its analytic expression using zeros and poles
    zeros: [[radius, angle], [radius,angle], ...]
    poles: same as zeros
    '''
    ccep = [0]*N
    min_z = [z for z in zeros if z[0] < 1]
    max_z = [z for z in zeros if z[0] > 1]
    

    #minimum phase modes
    for i in range(1,N):
        if min_z:
            ccep[i] = ccep[i] - real_mode_sum(i,min_z)
        if max_z:
            ccep[N-i] = ccep[N-i] - real_mode_sum(i,max_z)
        
        if poles:
            ccep[i] = ccep[i] + real_mode_sum(i,poles)
        
        
    return ccep

def ad_cep(N,zeros = None, poles = None):
    '''
    computes the differential cepstrum from the analytic expression given the
    eros and poles of a system
    '''
    ccep = cepstrum_expression(N,zeros = zeros, poles = poles)
    return ccep*np.linspace(0,N,N)

def differential_cepstrum(x, win = False, n=None):
    ccep, ndelay = complex_cepstrum(x, win, n)
    N = len(ccep)
    
    return ccep*np.linspace(0, N, N) , ndelay# - np.pi * ndelay * np.arange(N) / N

def pol_to_car(r,a):
    x = r*np.cos(a)
    y = r*np.sin(a)
    return [x,y]

def car_to_pol(x,y):
    r = np.sqrt(x**2+y**2)
    a = np.arctan2(y,x)
    return [r,a]  

def to_plot_polar(pol_modes):
    x = []
    y = []
    for i in range(len(pol_modes)):
        x.append([pol_to_car(pol_modes[i][0], pol_modes[i][1])[0]])
        y.append([pol_to_car(pol_modes[i][0], pol_modes[i][1])[1]])
    
    return x, y

def plot_unit_circle(ax, N = 1024):
    t = [i*2*np.pi/N for i in range(N)]
    t2 = [2*(i - N / 2) / N for i in range(N)]
    ax.plot(np.cos(t), np.sin(t), linewidth=1)
    ax.plot(t2, np.zeros((N,1)))
    ax.plot(np.zeros((N,1)),t2)

def prony(this_x,p):
    
    N = 2*p
    x = this_x[1:N+1]
    
    T = scipy.linalg.toeplitz(x[p-1:N-1], np.flip(x[:p]))
    a = np.linalg.solve(-T, x[p:N])
    c = np.concatenate([[1],a])
    r = np.roots(c)
    
    alfa = np.log(np.abs(r))
    freq = np.arctan2(r.imag,r.real) / (2*np.pi)
    
    Z = np.matrix(np.zeros((p,p), dtype = "complex_"))
    
    pw = [i for i in range(p)]
    for i in range(len(r)):
        Z.T[i] = r[i]**pw
    
    h = np.linalg.solve(Z, x[:p])
    
    Amp = np.abs(h)
    theta = np.arctan2(h.imag,h.real)    
    
    return [Amp, alfa, freq, theta]
    
    
    
def sort_modes(Amp,alfa,freq,theta, th = 0.3):
    finalz = []; finalp = [];
    
    for i in range(len(Amp)):
        x = np.exp(alfa[i])*np.cos(2*np.pi*freq[i])
        y = np.exp(alfa[i])*np.sin(2*np.pi*freq[i])
        #if np.abs(np.abs((theta[i]-freq[i]*2*np.pi)) - np.pi) < th:
        if np.cos(theta[i]-freq[i]*2*np.pi) < 0:
            finalz.append([x,y]);
        else:
            finalp.append([x,y]);
    
    return np.array(finalz), np.array(finalp)

def match_nodes(modes, gt):
    
    return

def plot_reconstructed(N,Amp,alfa,freq,theta):
    new = np.zeros((N,1), dtype = "complex_")
    for n in range(1,N):
        for i in range(len(Amp)):
            #we subtract 2 from n because we fit d_cep(2:N), also \hat{x}[0] =
            #b_{m}, thus the first fitted sample is just the amplitude and the
            #phase offset
            new[n] = new[n] + Amp[i]*np.exp(1j*theta[i])*np.exp(alfa[i]*(n-1))*np.exp(1j*freq[i]*2*np.pi*(n-1))
            
    return new

def polar_modes_mse(modes, gt):
    
    new_modes = np.copy(modes);new_gt = np.copy(gt)
    out_modes = []; out_gt = []
    
    while new_modes.shape[0] > 0 and new_gt.shape[0] > 0:
        d_mat = distance_matrix(new_modes, new_gt)
        x1, x2 = np.where(d_mat == d_mat.min())
        
        out_modes.append(new_modes[x1])
        out_gt.append(new_gt[x2])
    
        new_modes = np.delete(new_modes, x1, 0)
        new_gt = np.delete(new_gt, x2, 0)
    
    return np.concatenate(out_modes), np.concatenate(out_gt)

def get_matlab_dcep():
    dcep = np.array([0,
    -2.40416305603426,
    -3.34328576167227e-16,
    1.75503903090501,
    0.493000000000000,
    1.29848846656410,
    4.84178685339816e-16,
    -0.972996042896794,
    -0.525390100000000,
    -0.737707841133520,
    -7.92216867543415e-16,
    0.565275221005905,
    0.427420119490000,
    0.437221325614946,
    1.52969059811903e-16,
    -0.340932247572210,
    -0.314309042428237,
    -0.267696223780917,
    -2.88997066994739e-16,
    0.211420247340896,
    0.220094879089002,
    0.167785636236272,
    4.86125908956648e-16,
    -0.133688916321074,
    -0.150088153188006,
    -0.106868854840673,
    1.01508314956216e-16,
    0.0856555052981319,
    0.100800963982705,
    0.0687996683328862,
    2.30899072674205e-16,
    -0.0553557051361333,
    -0.0670891131555650,
    -0.0446000243836252,
    3.59391309226557e-17,
    0.0359736378136849,
    0.0444077619825615,
    0.0290411221893720,
    1.10953314331160e-16,
    -0.0234608933319750,
    -0.0292959202837122,
    -0.0189633775883934,
    1.19948547961604e-17,
    0.0153347703998745,
    0.0192865842383358,
    0.0124048021379949,
    5.02033600942717e-17,
    -0.0100374181247977,
    -0.0126807693918748,
    -0.00812360685262013,
    2.34920530069131e-18,
    0.00657583238040400,
    0.00833084706539196,
    0.00532367915918757,
    2.33141442574558e-17,
    -0.00431042327482053,
    -0.00547037201152369,
    -0.00349031847236694,
    -6.53011246476021e-20,
    0.00282643835926508,
    0.00359095560874714,
    0.00228895452485936,
    1.07327800509457e-17,
    -0.00185375841560305])
    
    return dcep

if __name__ == '__main__':
    dcep = get_matlab_dcep()
    p = 4
    
    print(prony(dcep, p))
    