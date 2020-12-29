from cepstrum_utils import *
'''
Estimate the zeros and poles of a system using the Cepstrum and Prony Algoritm

'''

########specify parameters of the script#########
N = 64 #spectrum size
lpc_order = 6
M = 2*lpc_order #cepstrum window size

t = np.linspace(0,M,M)

########specify modes of the system#########
fname = r'/Users/christian/Desktop/folder/Code/python/ZP_cep_est/u_150hz_REDS.wav'
# fname = r'/Users/christian/Desktop/folder/Code/python/ZP_cep_est/e_vocal.wav'
sr, y = wavfile.read(fname)
x = np.copy(y)
x = x / sr

x = x-x.min(); x = x / x.max(); x = x*2 - 1

########compute complex differential cepstrum#########
dcep = differential_cepstrum(x[:256]*sp.hanning(len(x[:256])))

#import pdb; pdb.set_trace()

########Run Prony Algorithm to find phase of the modes#########

[Amp, alfa, freq, theta] = prony(dcep, lpc_order)

########reconstruct cepstru from approximation########
new = plot_reconstructed(N,Amp,alfa,freq,theta)

########sort zeros from poles########
finalz, finalp = sort_modes(Amp, alfa, freq, theta)

########plot results########
plt.figure()
plt.plot(x)
plt.title('Time Signal')
plt.show()

plt.figure()
plt.plot(dcep)
plt.title('Recorded Differential Cepstrum')
plt.show()

plt.figure()
plt.plot(new.real)
plt.title('Estimated Differential Cepstrum')
plt.show()

plt.figure()
if len(finalz) > 0:
    plt.plot(finalz.T[0], finalz.T[1], 'bo')
if len(finalp) > 0:
    plt.plot(finalp.T[0], finalp.T[1], 'bx')
plot_unit_circle()
plt.title('Estimated Poles/Zeros')
plt.show()
