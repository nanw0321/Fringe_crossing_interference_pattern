from input import *

''' regular beamline '''
def define_branch_VCC(E0=9000, z_off=.12, d23=.2, alphaAsym=np.deg2rad(5), f_lens=10.,
    FOV1 = 1e-3, FOV2 = 100e-6, N = 1024):
    # crystal reflection hkl index
    hkl = [2,2,0]

    # viewing point upstream of monochromator
    im0 = optics.PPM('im0', z=400-.01, FOV=FOV1, N=N)

    slit = optics.Slit('s0', x_width=1., y_width=100e-3, dx=0, dy=-50e-3, z=400)
    
    # first crystal: symmetric reflection
    crystal1 = optics.Crystal('c1', hkl=hkl, length=10e-2, width=20e-3, z=400+z_off, E0=E0,
                              alphaAsym=0, orientation=2, pol='s', delta=0.e-6)
    
    # second crystal: asymmetric reflection, orientation flipped relative to crystal 1
    z12 = np.cos(2*crystal1.beta0)*0.02
    crystal2 = optics.Crystal('c2', hkl=hkl, length=10e-2, width=20e-3, z=crystal1.z+z12, E0=E0,
                              alphaAsym=alphaAsym, orientation=0, pol='s', delta=0.e-6)
    
    # printing crystal incidence and reflection angles for confirmation
    print('crystal 2 incidence angle: {:.2f} degrees'.format(crystal2.alpha*180/np.pi))
    print('crystal 2 exit angle: {:.2f} degrees'.format(crystal2.beta0*180/np.pi))

    im1 = optics.PPM('im1', z=crystal2.z+d23/2, FOV=FOV1, N=N)

    # third crystal, symmetric reflection, same orientation as crystal2
    z23 = d23
    crystal3 = optics.Crystal('c3', hkl=hkl, length=10e-2, width=10e-3, z=crystal2.z+z23, E0=E0,
                              alphaAsym=-alphaAsym, orientation=0, pol='s', asym_type='emergence')

    # fourth crystal, asymmetric reflection, same orientation as crystal1
    z34 = z12
    crystal4 = optics.Crystal('c4', hkl=hkl, length=10e-2, width=10e-3, z=crystal3.z+z34, E0=E0,
                              alphaAsym=0, orientation=2, pol='s', asym_type='emergence')

    # viewing point just downstream of monochromator
    im2 = optics.PPM('im2', z=crystal4.z+z_off+.01, FOV=FOV1, N=N)
    
    # spectrometer lens
    crl1 = CRL_no_abs('crl1', z=im2.z+.01, E0=E0, f=f_lens, diameter=5e-3, orientation=1)
    
    # viewing point at spectrometer lens focal plane
    im3 = optics.PPM('im3', z=crl1.z+f_lens, FOV=FOV2, N=N)
    
    # spectrometer crystal
    crystal5 = optics.Crystal('c5', hkl=[5,5,5], length=10e-2, width=10e-2, z=crl1.z+f_lens*2, E0=E0,
                              alphaAsym=0, orientation=1, pol='s', asym_type='emergence')
        
    # viewing point at crossing
    z56 = np.cos(np.pi-2*crystal5.alpha)*.1
    im4 = optics.PPM('im4', z=crystal5.z-z56, FOV=FOV2, N=N)
    
    # list of devices to propagate through
    devices = [im0, crystal1, crystal2, im1, crystal3, crystal4, im2,
               crl1, im3, crystal5, im4]

    return devices

def define_branch_CC(E0=9000, d23 = .2, alphaAsym=np.deg2rad(0), f_lens=10.,
    FOV1 = 1e-3, FOV2 = 100e-6, N = 1024):
    # crystal reflection hkl index
    hkl = [2,2,0]

    # viewing point upstream of monochromator
    im0 = optics.PPM('im0', z=400-.01, FOV = FOV1, N=N)
    
    slit = optics.Slit('s0', x_width=1., y_width=100e-3, dx=0, dy=50e-3, z=400)

    # first crystal: symmetric reflection
    crystal1 = optics.Crystal('c1', hkl=hkl, length=10e-2, width=20e-3, z=400, E0=E0,
                              alphaAsym=0, orientation=0, pol='s', delta=0.e-6)
    
    # second crystal: symmetric reflection, orientation flipped relative to crystal 1
    z12 = np.cos(2*crystal1.beta0)*0.02
    crystal2 = optics.Crystal('c2', hkl=hkl, length=10e-2, width=20e-3, z=crystal1.z+z12, E0=E0,
                              alphaAsym=alphaAsym, orientation=2, pol='s', delta=0.e-6)
    
    # printing crystal incidence and reflection angles for confirmation
    print('crystal 2 incidence angle: {:.2f} degrees'.format(crystal2.alpha*180/np.pi))
    print('crystal 2 exit angle: {:.2f} degrees'.format(crystal2.beta0*180/np.pi))

    im1 = optics.PPM('im1', z=crystal2.z+d23/2+z_off, FOV=FOV1, N=N)

    # third crystal, symmetric reflection, same orientation as crystal2
    z23 = d23 + 2*z_off
    crystal3 = optics.Crystal('c3', hkl=hkl, length=10e-2, width=10e-3, z=crystal2.z+z23, E0=E0,
                              alphaAsym=-alphaAsym, orientation=2, pol='s', asym_type='emergence')

    # fourth crystal, symmetric reflection, same orientation as crystal1
    z34 = z12
    crystal4 = optics.Crystal('c4', hkl=hkl, length=10e-2, width=10e-3, z=crystal3.z+z34, E0=E0,
                              alphaAsym=0, orientation=0, pol='s', asym_type='emergence')
    
    # viewing point just downstream of monochromator
    im2 = optics.PPM('im2', z=crystal4.z+.01, FOV=FOV1, N=N)
    
     # spectrometer lens
    crl1 = CRL_no_abs('crl1', z=im2.z+.01, E0=E0, f=f_lens, diameter=5e-3, orientation=1)

    # viewing point at spectrometer lens focal plane
    im3 = optics.PPM('im3', z=crl1.z+f_lens, FOV=FOV2, N=N)
        
    # spectrometer crystal
    crystal5 = optics.Crystal('c5', hkl=[5,5,5], length=10e-2, width=10e-2, z=crl1.z+f_lens*2, E0=E0,
                              alphaAsym=0, orientation=1, pol='s', asym_type='emergence')
    
    # viewing point at crossing
    z56 = np.cos(np.pi-2*crystal5.alpha)*.1
    im4 = optics.PPM('im4', z=crystal5.z-z56, FOV=FOV2, N=N)

    # list of devices to propagate through
    devices = [im0, crystal1, crystal2, im1, crystal3, crystal4, im2,
               crl1, im3, crystal5, im4]

    return devices


''' modifications '''
def insert_tilt(devices, delta_a=0, orientation=1):
    tilt1 = Force_tilt('tilt1', delta_a=delta_a, z=devices[-1].z+.01, orientation=1)
    devices.insert(-1, tilt1)

def change_energy(pulse, E0):
    E_old = pulse.E0
    pulse.beam_params['photonEnergy'] = E0

    pulse.E0 = E0
    pulse.energy += E0 - E_old
    pulse.f = pulse.energy / 4.136
    pulse.f0 = pulse.E0 / 4.136
    pulse.wavelength = 1239.8/pulse.energy*1e-9
    


