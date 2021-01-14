from cepstrum_utils import *
'''
Estimate the zeros and poles of a system using the Cepstrum and Prony Algoritm

TODO:
    1. write function to calculate error of estimate
    2. write function to generate random poles/zeros
    3. run tests to get a sense of the accuracy of the method
    4. integrate additive noise
    5. integrate user audio option
    6. include non-minimum phase modes 
'''

parser = argparse.ArgumentParser()

parser.add_argument('--plot', help='flag to plot')
parser.add_argument('--window', help='flag to window impulse response')
#parser.add_argument('--lpc_order', type=int, default = 5)
parser.add_argument('--N', type=int, default=512, help='spectrum size')
parser.add_argument('--noise', help='flag to add noise to impulse response')
parser.add_argument('--mode', type=int, default=1, help='0: impulse, 1: dirac comb')
parser.add_argument('--freq', type=int, default=128)
parser.add_argument('--numzeros', type=int, default=1)
parser.add_argument('--numpoles', type=int, default=3)

if __name__ == '__main__':
    
    args = parser.parse_args()
        
    ########specify parameters of the script#########
    N = args.N
    plot = 1#args.plot
    window=args.window
    noise = args.noise
    mode = args.mode
    freq = args.freq
    numzeros = args.numzeros
    numpoles = args.numpoles
    
    ########specify modes of the system in polar#########
    # zeros = [[0.9, -np.pi], [0.9, 0]]
    # poles = [[0.5, np.pi/3], [0.5, -np.pi/3], [0.9, 7*np.pi/8], [0.9, -7*np.pi/8]]
    
    # zeros = [[0.9, -np.pi/3], [0.9, np.pi/3]]
    # poles = [[0.5, np.pi/3], [0.5, -np.pi/3], [0.9, 7*np.pi/8], [0.9, -7*np.pi/8]]
    
    # zeros = [[0.9, 3*np.pi/8], [0.9, -3*np.pi/8]]
    # poles = [[0.6, 3*np.pi/16], [0.6, -3*np.pi/16]]
    
    zeros = []
    for z in range(numzeros):
        r = np.random.rand()
        a = np.random.rand()*np.pi
        zeros.append([r, a]); zeros.append([r, -a])
        
    poles = []
    for z in range(numpoles):
        r = np.random.rand()
        a = np.random.rand()*np.pi
        poles.append([r, a]); poles.append([r, -a])
    
    lpc_order = len(zeros) + len(poles)
    M = 2*lpc_order #cepstrum window size
    t = np.linspace(0,M,M)
    
    
    impulse_z = []; impulse_p = []
    if zeros:
        orig_xz, orig_yz = to_plot_polar(zeros)
        impulse_z = np.array([x[0]+1j*y[0] for (x,y) in zip(orig_xz, orig_yz)])
    if poles:
        orig_xp, orig_yp = to_plot_polar(poles)
        impulse_p = np.array([x[0]+1j*y[0] for (x,y) in zip(orig_xp, orig_yp)])
        
    
    
    modes = np.array([[x] for x in np.concatenate([impulse_p, impulse_z])])
    
    impulse = sp.dlti(impulse_z, impulse_p, [1.0]).impulse(n=N)[1][0].flatten()
    
    if mode == 1:
        comb = [1 if i % freq == 0 else 0 for i in range(N)]
        impulse = np.convolve(impulse, comb)[:N]
    
    if noise:
        impulse += 0.005*np.random.normal(0,1,N)
        
    
    
    #----------------------------A N A L Y T I C-------------------------------#
    
    ########compute analytic complex differential cepstrum#########
    a_dcep = ad_cep(N, zeros = zeros, poles = poles)
    
    ########Run Prony Algorithm to find phase of the modes#########
    
    [a_Amp, a_alfa, a_freq, a_theta] = prony(a_dcep, lpc_order)
    
    ########reconstruct cepstru from approximation########
    a_new = plot_reconstructed(N,a_Amp,a_alfa,a_freq,a_theta)
    
    ########sort zeros from poles########
    a_finalz, a_finalp = sort_modes(a_Amp, a_alfa, a_freq, a_theta)
    
    #MSE calculation
    a_pcar = [pol_to_car(x[0], x[1]) for x in a_finalp]
    a_pcar = np.array([x[0]+1j*x[1] for x in a_pcar])
    
    a_zcar = [pol_to_car(x[0], x[1]) for x in a_finalz]
    a_zcar = np.array([x[0]+1j*x[1] for x in a_zcar])
    
    a_modes = np.array([[x] for x in np.concatenate([a_pcar, a_zcar])])
    
    a_modes, modes = polar_modes_mse(a_modes, modes)
    
    a_mse = np.abs(np.square(a_modes - modes)).sum() / len(modes)
    
    #----------------------------C O M P U T E-------------------------------#
    
    ########compute complex differential cepstrum#########
    
    c_dcep, ndelay = differential_cepstrum(impulse, win=False)
    
    [c_Amp, c_alfa, c_freq, c_theta] = prony(c_dcep, lpc_order)
    
    ########reconstruct cepstru from approximation########
    c_new = plot_reconstructed(N,c_Amp,c_alfa,c_freq,c_theta)
    
    ########sort zeros from poles########
    c_finalz, c_finalp = sort_modes(c_Amp, c_alfa, c_freq, c_theta)
    
    #MSE calculation
    c_pcar = [pol_to_car(x[0], x[1]) for x in c_finalp]
    c_pcar = np.array([x[0]+1j*x[1] for x in c_pcar])
    
    c_zcar = [pol_to_car(x[0], x[1]) for x in c_finalz]
    c_zcar = np.array([x[0]+1j*x[1] for x in c_zcar])
    
    c_modes = np.array([[x] for x in np.concatenate([c_pcar, c_zcar])])
    
    c_modes, modes = polar_modes_mse(c_modes, modes)
    
    c_mse = np.abs(np.square(c_modes - modes)).sum() / len(modes)

    ########plot results########
    if plot:
        fig, ax = plt.subplots(nrows=3, ncols=3)
        
        #zero/pole location subplot
        if zeros:
            ax[0, 0].plot(orig_xz, orig_yz, 'go')
        if poles:
            ax[0, 0].plot(orig_xp, orig_yp, 'gx')
        ax[0,0].title.set_text('Zeros (o), Poles (x)')
        plot_unit_circle(ax[0,0])
        
        #system impulse response subplot
        ax[0, 1].plot(impulse)
        ax[0,1].title.set_text('Impulse Response')
        
        #system frequency response subplot
        ax[0, 2].plot(np.abs(np.fft.fft(impulse)))
        ax[0,2].title.set_text('Frequency Response')
        
        #--------------------------A N A L Y T I C-----------------------------#
        
        #system analytic differential cepstrum
        ax[1, 0].plot(a_dcep)
        ax[1,0].title.set_text('Analytic Differential Cepstrum')
        
        #plot extracted zeros/poles
        if zeros:
            ax[1, 1].plot(orig_xz, orig_yz, 'go')
        if poles:
            ax[1, 1].plot(orig_xp, orig_yp, 'gx')
        if len(a_finalz) > 0:
            ax[1, 1].plot(a_finalz.T[0], a_finalz.T[1], 'bo')
        if len(a_finalp) > 0:
            ax[1, 1].plot(a_finalp.T[0], a_finalp.T[1], 'bx')
        ax[1,1].title.set_text('Analytic Estimated Zeros (o) Poles (x).  MSE: {:.4f}'.format(a_mse))
        plot_unit_circle(ax[1,1])
        
        #system analytic differential cepstrum
        ax[1, 2].plot(a_new.real)
        ax[1,2].title.set_text('Reconstructed Differential Cepstrum')
        
        #--------------------------C O M P U T E D-----------------------------#
        
        #system analytic differential cepstrum
        ax[2, 0].plot(c_dcep)
        ax[2,0].title.set_text('Computed Differential Cepstrum')
        
        #plot extracted zeros/poles
        if zeros:
            ax[2, 1].plot(orig_xz, orig_yz, 'go')
        if poles:
            ax[2, 1].plot(orig_xp, orig_yp, 'gx')
        if len(c_finalz) > 0:
            ax[2, 1].plot(c_finalz.T[0], c_finalz.T[1], 'bo')
        if len(c_finalp) > 0:
            ax[2, 1].plot(c_finalp.T[0], c_finalp.T[1], 'bx')
        ax[2,1].title.set_text('Computed Estimated Zeros (o) Poles (x). MSE: {:.4f}'.format(c_mse))
        plot_unit_circle(ax[2,1])
        
        #system analytic differential cepstrum
        ax[2, 2].plot(c_new.real)
        ax[2,2].title.set_text('Reconstructed Differential Cepstrum')
        
        plt.show()