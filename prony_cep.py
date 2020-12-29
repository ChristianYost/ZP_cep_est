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

########specify parameters of the script#########
N = 64 #spectrum size
lpc_order = 6
M = 2*lpc_order #cepstrum window size

t = np.linspace(0,M,M)

########specify modes of the system#########
z1 = [0.9, np.pi/4]
z1_bar = [z1[0],-z1[1]] #conjugate pair

z2 = [1.1, np.pi/4]
z2_bar = [z2[0],-z2[1]] #conjugate pair

p1 = [0.8, 3*np.pi/4]
p1_bar = [p1[0],-p1[1]] #conjugate pair

zeros = [z1,z1_bar, z2, z2_bar]
poles = [p1,p1_bar]

if zeros:
    orig_xz, orig_yz = to_plot_polar(zeros)

if poles:
    orig_xp, orig_yp = to_plot_polar(poles)


########compute complex differential cepstrum#########
dcep = ad_cep(N, zeros = zeros, poles = poles)

########Run Prony Algorithm to find phase of the modes#########

[Amp, alfa, freq, theta] = prony(dcep, lpc_order)

########reconstruct cepstru from approximation########
new = plot_reconstructed(N,Amp,alfa,freq,theta)

########sort zeros from poles########
finalz, finalp = sort_modes(Amp, alfa, freq, theta)

########plot results########
plt.figure()
plt.plot(dcep)
plt.title('Original Differential Cepstrum')
plt.show()

plt.figure()
plt.plot(orig_xz, orig_yz, 'bo')
plt.plot(orig_xp, orig_yp, 'bx')
plot_unit_circle()
plt.title('Original Poles/Zeros')
plt.show()

plt.figure()
plt.plot(new.real)
plt.title('Approximated Differential Cepstrum')
plt.show()

plt.figure()
plt.plot(finalz.T[0], finalz.T[1], 'bo')
plt.plot(finalp.T[0], finalp.T[1], 'bx')
plot_unit_circle()
plt.title('Estimated Poles/Zeros')
plt.show()

plt.figure()
plt.plot(finalz.T[0], finalz.T[1], 'bo')
plt.plot(finalp.T[0], finalp.T[1], 'bx')
plt.plot(orig_xz, orig_yz, 'go')
plt.plot(orig_xp, orig_yp, 'gx')
plot_unit_circle()
plt.title('Estimated(blue) Ground Truth(green) Poles/Zeros')
plt.show()