import time, h5py, os, sys
import numpy as np
import matplotlib.pyplot as plt
from lcls_beamline_toolbox.xraybeamline2d import beam1d as beam, optics1d as optics, beamline1d as beamline
from lcls_beamline_toolbox.xraybeamline2d.util import Util

''' misc '''
def make_dir(path):
    if not os.path.exists(path):
        print('make path')
        os.mkdir(path)
    else:
        print('path exists')

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

''' define beamline '''
def define_branch_VCC(E0=9000, z_off=.12, d23=.2, alphaAsym=np.deg2rad(5), FOV1 = 1e-3, FOV2 = 100e-6, N = 1024):
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
                              alphaAsym=0, orientation=0, pol='s', asym_type='emergence')

    # fourth crystal, asymmetric reflection, same orientation as crystal1
    z34 = z12
    crystal4 = optics.Crystal('c4', hkl=hkl, length=10e-2, width=10e-3, z=crystal3.z+z34, E0=E0,
                              alphaAsym=alphaAsym, orientation=2, pol='s', asym_type='emergence')

    # viewing point just downstream of monochromator
    im2 = optics.PPM('im2', z=crystal4.z+z_off+.01, FOV=FOV1, N=N)
    
    # viewing point at crossing
    im3 = optics.PPM('im3', z=im2.z+f_prism+.01, FOV=FOV2, N=N)

    # list of devices to propagate through
    devices = [im0, slit, crystal1, crystal2, im1, crystal3, crystal4, im2, im3]

    return devices

def define_branch_CC(E0=9000, d23 = .2, alphaAsym=np.deg2rad(0), FOV1 = 1e-3, FOV2 = 100e-6, N = 1024):
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
                              alphaAsym=0, orientation=2, pol='s', delta=0.e-6)
    
    # printing crystal incidence and reflection angles for confirmation
    print('crystal 2 incidence angle: {:.2f} degrees'.format(crystal2.alpha*180/np.pi))
    print('crystal 2 exit angle: {:.2f} degrees'.format(crystal2.beta0*180/np.pi))

    im1 = optics.PPM('im1', z=crystal2.z+d23/2+z_off, FOV=FOV1, N=N)

    # third crystal, symmetric reflection, same orientation as crystal2
    z23 = d23 + 2*z_off
    crystal3 = optics.Crystal('c3', hkl=hkl, length=10e-2, width=10e-3, z=crystal2.z+z23, E0=E0,
                              alphaAsym=0, orientation=2, pol='s', asym_type='emergence')

    # fourth crystal, symmetric reflection, same orientation as crystal1
    z34 = z12
    crystal4 = optics.Crystal('c4', hkl=hkl, length=10e-2, width=10e-3, z=crystal3.z+z34, E0=E0,
                              alphaAsym=0, orientation=0, pol='s', asym_type='emergence')
    
    # viewing point just downstream of prism
    im2 = optics.PPM('im2', z=crystal4.z+.01, FOV=FOV1, N=N)
    
    # prism
    prism = Prism('prism', x_width=x_prism, y_width=y_prism, slope=slope, material='Be', z=im2.z+.02,
                  dx=dx_prism, dy=dy_prism, orientation=1)
    
    # viewing point at crossing
    im3 = optics.PPM('im3', z=im2.z+f_prism+.01, FOV=FOV2, N=N)
    
    # list of devices to propagate through
    devices = [im0, slit, crystal1, crystal2, im1, crystal3, crystal4, im2, prism, im3]

    return devices


''' get info '''
def print_oe_pos(oe):
    print('{}, x:{}, y:{}, z:{}'.format(oe.name, oe.global_x, oe.global_y, oe.z))
    return oe.global_x, oe.global_y, oe.z

def get_pulse(pulse, image_name, x_pos=0, y_pos=0, shift=None):
    minx = np.round(np.min(pulse.x[image_name]) * 1e6)
    maxx = np.round(np.max(pulse.x[image_name]) * 1e6)
    miny = np.round(np.min(pulse.y[image_name]) * 1e6)
    maxy = np.round(np.max(pulse.y[image_name]) * 1e6)

    # get number of pixels
    M = pulse.x[image_name].size
    N = pulse.y[image_name].size

    # calculate pixel sizes (microns)
    dx = (maxx - minx) / M
    dy = (maxy - miny) / N

    # calculate indices for the desired location
    x_index = int((x_pos - minx) / dx)
    y_index = int((y_pos - miny) / dy)

    # calculate temporal intensity
    y_data = np.abs(pulse.time_stacks[image_name][y_index, x_index, :]) ** 2

    shift = -pulse.t_axis[np.argmax(y_data)]

    # coarse shift for fitting
    if shift is not None:
        y_data = np.roll(y_data, int(shift/pulse.deltaT))

    # get gaussian stats
    centroid, sx = Util.gaussian_stats(pulse.t_axis, y_data)
    fwhm = int(sx * 2.355)

    # gaussian fit
    gauss_plot = Util.fit_gaussian(pulse.t_axis, centroid, sx)

    # shift again using fit result
    shift = -centroid
    if shift is not None:
        y_data = np.roll(y_data, int(shift/pulse.deltaT))
        gauss_plot = np.roll(gauss_plot, int(shift/pulse.deltaT))
        
    # [fs], normalized intensity [simulated], [Gaussian Fit]
    return pulse.t_axis, y_data/np.max(y_data), gauss_plot

def get_spectrum(pulse, image_name, x_pos=0, y_pos=0, integrated=False):
    minx = np.round(np.min(pulse.x[image_name]) * 1e6)
    maxx = np.round(np.max(pulse.x[image_name]) * 1e6)
    miny = np.round(np.min(pulse.y[image_name]) * 1e6)
    maxy = np.round(np.max(pulse.y[image_name]) * 1e6)

    # get number of pixels
    M = pulse.x[image_name].size
    N = pulse.y[image_name].size

    # calculate pixel sizes (microns)
    dx = (maxx - minx) / M
    dy = (maxy - miny) / N

    # calculate indices for the desired location
    x_index = int((x_pos - minx) / dx)
    y_index = int((y_pos - miny) / dy)

    # calculate spectral intensity
    if integrated:
        y_data = np.sum(np.abs(pulse.energy_stacks[image_name])**2, axis=(0,1))
    else:
        y_data = np.abs(pulse.energy_stacks[image_name][y_index,x_index,:])**2

    # get gaussian stats
    centroid, sx = Util.gaussian_stats(pulse.energy, y_data)
    fwhm = sx * 2.355

    # gaussian fit to plot
    gauss_plot = Util.fit_gaussian(pulse.energy, centroid, sx)

    # change label depending on bandwidth
    if fwhm >= 1:
        width_label = '%.1f eV FWHM' % fwhm
    elif fwhm > 1e-3:
        width_label = '%.1f meV FHWM' % (fwhm * 1e3)
    else:
        width_label = u'%.1f \u03BCeV FWHM' % (fwhm * 1e6)
    
    # [eV], normalized intensity [simulated], [Gaussian Fit]
    return pulse.energy - pulse.E0, y_data/np.max(y_data), gauss_plot

''' beam crossing '''
def find_t_center(pulse, image_name):
    intensity = np.sum(np.abs(pulse.time_stacks[image_name]), axis=(0,1))
    tcenter = pulse.t_axis[np.argmax(intensity)]
    return tcenter

def find_shift(pulse1, pulse2, image_name):
    t1 = find_t_center(pulse1, image_name)
    t2 = find_t_center(pulse2, image_name)
    return t2 - t1

''' I/O '''
def save_pulse(pulse, image_name, output_name):
    with h5py.File(output_name,'w') as f:
        f.create_dataset('time_stacks', data=pulse.time_stacks[image_name])
        f.create_dataset('energy_stacks', data=pulse.energy_stacks[image_name])
        f.create_dataset('x', data=pulse.x[image_name])
        f.create_dataset('y', data=pulse.y[image_name])
        f.create_dataset('t_axis', data=pulse.t_axis)
        f.create_dataset('energy', data=pulse.energy)
