from utils import *
'''
Estimate the zeros and poles of a system using the Cepstrum and Prony Algoritm
'''

parser = argparse.ArgumentParser()

parser.add_argument('--plot', action='store_true', default=True, help='flag to plot')
parser.add_argument('--window', action='store_true', default=False, help='flag to window impulse response')
parser.add_argument('--N', type=int, default=512, help='spectrum size')
parser.add_argument('--noise', action='store_true', default=False, help='flag to add noise to impulse response')
parser.add_argument('--numzeros', type=int, default=2, help='number of zeros in system')
parser.add_argument('--numpoles', type=int, default=2, help='number of poles in system')
parser.add_argument('--seed', type=int, help='seed for random number generator')

if __name__ == '__main__':
    
    args = parser.parse_args()
        
    ########specify parameters of the script#########
    N = args.N
    plot = args.plot
    window=args.window
    noise = args.noise
    numzeros = args.numzeros
    numpoles = args.numpoles
    seed = args.seed
    
    assert numpoles >= numzeros
    
    np.random.seed(seed)
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
    
    if noise:
        impulse += 0.001*np.random.normal(0,1,N)
        
    
    
    #----------------------------A N A L Y T I C-------------------------------#
    
    ########compute analytic complex differential cepstrum#########
    a_dcep = ad_cep(N, zeros = zeros, poles = poles)
    
    a_finalz, a_finalp, a_new_dcep, a_modes, a_mse = run(a_dcep, lpc_order, N, modes)
    
    #----------------------------C O M P U T E-------------------------------#
    
    ########compute complex differential cepstrum#########
    
    c_dcep, ndelay = differential_cepstrum(impulse, win=False)
    
    c_finalz, c_finalp, c_new_dcep, c_modes, c_mse = run(c_dcep, lpc_order, N, modes)
    
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
        ax[1, 2].plot(a_new_dcep.real)
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
        ax[2, 2].plot(c_new_dcep.real)
        ax[2,2].title.set_text('Reconstructed Differential Cepstrum')
        
        plt.show()
