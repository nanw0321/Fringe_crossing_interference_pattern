import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import patches
from scipy.special import erf
plt.ion()

# Functions to be used
def sind(x):
    return np.sin(x*math.pi/180)

def cosd(x):
    return np.cos(x*math.pi/180)

def tand(x):
	return np.tan(x*math.pi/180)

def E2f(E):
    h = 4.135667516e-15 # eV
    return E/h

def abs2(x):
	return x.real**2 + x.imag**2

def IntGausProd(m1,m2,s1,s2):
	a = np.sqrt(1/s1**2 + 1/s2**2)
	b = (m1/s1**2 + m2/s2**2)/a
	c = (m1/s1)**2 + (m2/s2)**2
	# this is int[N(m1,s1)*N(m2,s2)] from 0 to inf
	#return np.exp(b**2-c) * np.sqrt(np.pi)/2 * (erf(b)+1)/a
	# this is from -inf to inf
	return np.exp(b**2-c) * np.sqrt(np.pi)/a

def IntGausSqr(m,s):
	b = s/np.sqrt(2)
	# this is int[N(m1,s1)**2] from 0 to inf
	#return np.sqrt(np.pi)/2 * b * (erf(m/b) + 1)
	# this is from -inf to inf
	return s/np.sqrt(2)

def fcross(m1,m2,s1,s2,omega1,omega2):
	a = np.sqrt(1/s1**2 + 1/s2**2)
	b = (m1/s1**2 + m2/s2**2)/a
	c = (m1/s1)**2 + (m2/s2)**2
	df = omega1 - omega2
	coef1 = np.exp(b**2-c)
	coef2 = np.sqrt(np.pi)/a
	coef3 = np.exp(-df**2/(4*a**2)-1j*b/a * df)
	return coef1 * coef2 * coef3


# List of constants
c = 3e8         # m/s
dpxl = 1 * 1e-6    # 0.5 um pixels
npxl = 1001

xx,yy = np.indices((npxl,npxl))
xx = xx-xx.mean(); xx2 = xx;
yy = yy-yy.mean(); yy2 = yy;
xx = xx * dpxl
yy = yy * dpxl

# Beam parameters
E = 9500        # eV
f = E2f(E)
wavelength = c/f
k = 2*np.pi/wavelength
z = 400         # m to detector
dely = 6.4e-3		# m prism offset
dely = dely

w0 = 50e-6/np.sqrt(2*np.log(2))/2  # beam width at waist (rms)
zR = np.pi * w0**2 / wavelength     # Rayleigh length
wz = w0 * np.sqrt(1+(z/zR)**2)		# beam width at z
Rz = z * (1+(zR/z)**2)      # radius of curvature
Rx = Rz
#Rx = Rz/3					
Ry = Rz
L = 6.5						# m spd to detector
overlap = 200e-6				# um overlap
theta_overlap = np.arctan(overlap*2/L)

# asymmetric reflection
theta = 19.87      # Bragg angle
delta = 5       # miscut
b = sind(theta+delta)/sind(theta-delta)

z2= z/b**2
w02 = w0/b**2
zR2 = np.pi * w02**2 / wavelength
wz2 = w02 * np.sqrt(1+(z2/zR2)**2)
Rz2 = z2 * (1+(zR2/z2)**2)
Rx2 = Rz2
Rx2 = Rz2/3				##### artificial correction for shape matching
Ry2 = Ry
wzx = wz/b **2

# Spatial envelope

A1 = np.exp(-(xx**2+yy**2)/wz**2) * (-0.5*(erf(yy2-overlap/dpxl)-1))
A2 = np.exp(-(xx**2/wzx**2 + yy**2/wz**2)) *(0.5*(erf(yy2+overlap/dpxl)+1))

R1 = np.exp(1j*k * (z*np.cos(theta_overlap) + xx**2/(2*Rx) + (yy+dely)**2/(2*Ry)))
#R1 = np.exp(1j*k * (z*np.cos(theta_overlap) + yy*np.sin(theta_overlap) + (xx**2+(yy)**2)/(2*Rz)))
R2 = np.exp(1j*k * (z + xx**2/(2*Rx2) + yy**2/(2*Ry2)))

plot_spatial = 1
if plot_spatial == 1:
	I = np.abs(A1*R1+A2*R2)

	fig, ax = plt.subplots(figsize=(8,8))
	plt.imshow(I.T, cmap = 'plasma')
	rect1 = patches.Rectangle((540,15),50,5.5,edgecolor='none',facecolor='white',alpha = 1)
	ax.add_patch(rect1)
	plt.title('Spatial envelope', fontsize=18)
	plt.savefig('spatial_envelope.png',transparent='true')

# Pulse front tilt
t0 = Rz/c
delay_rate = (1-1/cosd(2*theta))*(1/tand(theta-delta)-b/tand(theta))/c
Chi = (b-1) * delay_rate
#chil
#Chi = 0.41e-9

# Two-pulse
def main(dt,tau1,tau2,omega1,omega2,phase,delay,ratio=1,R1=R1,R2=R2,t0=t0):
	IntT1 = (
		ratio**2 * IntGausSqr(0,tau1) + IntGausSqr(dt,tau2)
		+ ratio * np.exp(-1j*(omega1-omega2)*t0-1j*phase)*fcross(0,dt,tau1,tau2,omega1,omega2)
		+ ratio * np.exp(-1j*(omega2-omega1)*t0+1j*phase)*fcross(dt,0,tau2,tau1,omega2,omega1))

	IntT2 = (
		ratio**2 * IntGausSqr(Chi*xx-delay,tau1) + IntGausSqr(Chi*xx+dt,tau2)
		+ ratio * np.exp(-1j*(omega1-omega2)*t0-1j*phase)*fcross(Chi*xx-delay,Chi*xx+dt-delay,tau1,tau2,omega1,omega2)
		+ ratio * np.exp(-1j*(omega2-omega1)*t0+1j*phase)*fcross(Chi*xx-delay+dt,Chi*xx-delay,tau2,tau1,omega2,omega1))

	IntT1T2 = (
		ratio**2 * IntGausProd(0,Chi*xx-delay,tau1,tau1) + IntGausProd(dt,Chi*xx+dt-delay,tau2,tau2)
		+ ratio * np.exp(-1j*(omega1-omega2)*t0-1j*phase)*fcross(0,Chi*xx+dt-delay,tau1,tau2,omega1,omega2)
		+ ratio * np.exp(-1j*(omega2-omega1)*t0+1j*phase)*fcross(dt,Chi*xx-delay,tau2,tau1,omega2,omega1))

	IntT2T1 = (
		ratio**2 * IntGausProd(Chi*xx-delay,0,tau1,tau1) + IntGausProd(Chi*xx+dt-delay,dt,tau2,tau2)
		+ ratio * np.exp(-1j*(omega1-omega2)*t0-1j*phase)*fcross(Chi*xx-delay,dt,tau1,tau2,omega1,omega2)
		+ ratio * np.exp(-1j*(omega2-omega1)*t0+1j*phase)*fcross(Chi*xx+dt-delay,0,tau2,tau1,omega2,omega1))

	eqn1 = A1**2 * IntT1
	eqn2 = A2**2 * IntT2
	eqn3 = A1*A2 * (R1*np.conj(R2)) * IntT1T2
	eqn4 = A2*A1 * (R2*np.conj(R1)) * IntT2T1

	I = np.real(eqn1 + eqn2 + eqn3 + eqn4)
	return I

def FringePlot(I,title):
	fig, ax = plt.subplots(figsize=(10,4))
	#plt.imshow(I.T[245:355,39:562],cmap='plasma')
	plt.imshow(I.T,cmap='plasma')
	plt.title(title,fontsize=18)
	rect1 = patches.Rectangle((440,10.9),50,5.5,edgecolor='none',facecolor='white',alpha = 1)
	ax.add_patch(rect1)


# Plots
## Single pulse
dt=0
tau = 15000 * wavelength/c
omega = 2*np.pi*f
phase = 0
delay = 0
I = main(0,tau,tau,omega,omega,phase,delay)
title = 'single spike\n'+'pulse width = '+str(round(tau*1e15,1))+'fs'
FringePlot(I,title)
plt.savefig('1.single_spike.png',transparent='true')

## Two same pulses
tau = 15000 * wavelength/c
omega = 2*np.pi*f
phase = 0
dt = 3.8 * tau
I = main(dt,tau,tau,omega,omega,phase,delay)
title = '2 identical spikes\n'+'dt = '+str(round(dt*1e15,2))+'fs'
FringePlot(I,title)
plt.savefig('2.double_spike.png',transparent='true')

## tweeking plots
tau1 = 15000 * wavelength/c
tau2 = 15000 * wavelength/c
omega1 = 2*np.pi*f
domega = 0.002	# percentage difference between two frequencies
omega2 = omega1 * (1+domega*1e-2)
phase = -0.2*np.pi
dt = 3 * tau1
ratio = 5
I = main(dt,tau1,tau2,omega1,omega2,phase,delay,ratio=ratio)
dphase_fringe = dt/2/np.pi * domega * 1e-2*omega1 + phase
title = r'$\Delta\omega$ = '+str(round(domega*1e3,2))+'/1000 %'
FringePlot(I,title)
plt.xlabel(r'$\Delta\phi_1$ = {:.2f}$\pi$, $\Delta\phi_2$ = {:.2f}$\pi$'.format((omega1-omega2)*dt/(2*np.pi)/np.pi, phase/np.pi),fontsize=18)

plt.savefig('3.double_spike_dphase_domega.png',transparent='true')

## Intensity ratio
tau = 15000 * wavelength/c
omega = 2*np.pi*f
phase = 0
dt = 3.8 * tau
ratio = 5		# temporal amplitude of pulse1/pulse2
I = main(dt,tau,tau,omega,omega,phase,delay,ratio=ratio)
title = 'Intensity spike1/spike2='+str(ratio)+'\n'+'dt = '+str(round(dt*1e15,2))+'fs'
FringePlot(I,title)
plt.savefig('4.Intensity_difference.png',transparent='true')

## Temporal envelope
def GausT(t,tau):
	A = np.exp(-(t/tau)**2)
	return A**2

plt.figure(figsize=(10,6))
tt = np.linspace(-3*tau1,dt+3*tau2,10001)
phase = 0
envelope = ratio * GausT(tt,tau1)+GausT(tt-dt,tau2)
wave = envelope * np.sin(3.5*tt*1e15)
plt.plot(tt*1e15,envelope,label='envelope',color = 'gray', linewidth = 5)
plt.plot(tt*1e15,wave,label='wave',color = 'k', linewidth = 2)
#plt.legend(fontsize=18)
plt.xlabel('time (fs)',fontsize=18)
plt.axis('off')
#plt.title('Temporal intensity envelope',fontsize=18)
plt.savefig('temporal_envelope.png',transparent='true')
