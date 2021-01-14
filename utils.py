import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import scipy.signal as sp
import scipy
from scipy.io import wavfile as wavfile
import random
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
    ax.plot(np.cos(t), np.sin(t), 'black', linewidth=1)
    ax.plot(t2, np.zeros((N,1)), 'black', linewidth=1)
    ax.plot(np.zeros((N,1)),t2, 'black', linewidth=1)

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

def run(dcep, lpc_order, N, modes):
    ########Run Prony Algorithm to find phase of the modes#########
    
    [Amp, alfa, freq, theta] = prony(dcep, lpc_order)
    
    ########reconstruct cepstru from approximation########
    new_dcep = plot_reconstructed(N, Amp, alfa, freq, theta)
    
    ########sort zeros from poles########
    finalz, finalp = sort_modes(Amp, alfa, freq, theta)
    
    #MSE calculation
    pcar = [pol_to_car(x[0], x[1]) for x in finalp]
    pcar = np.array([x[0]+1j*x[1] for x in pcar])
    
    zcar = [pol_to_car(x[0], x[1]) for x in finalz]
    zcar = np.array([x[0]+1j*x[1] for x in zcar])
    
    new_modes = np.array([[x] for x in np.concatenate([pcar, zcar])])
    
    new_modes, modes = polar_modes_mse(new_modes, modes)
    
    mse = np.abs(np.square(new_modes - modes)).sum() / len(modes)
    
    return finalz, finalp, new_dcep, new_modes, mse 