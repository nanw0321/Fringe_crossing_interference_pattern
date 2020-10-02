import numpy as np
import os, copy, h5py
import pylab as plt

# import base wavefront and beamline class
from wpg import Wavefront, Beamline

# import base OE wrappers
from wpg.optical_elements import Aperture, Drift, CRL, Empty, Use_PP

# Gaussian beam generator
from wpg.generators import build_gauss_wavefront

#import SRW core functions
from wpg.srwlib import srwl, srwl_opt_setup_CRL, SRWLOptD, SRWLOptA, SRWLOptC, SRWLOptT, SRWLOptCryst

#import some helpers functions
from wpg.wpg_uti_wf import propagate_wavefront, plot_t_wf, get_intensity_on_axis
from wpg.wpg_uti_oe import show_transmission

''' trivia '''
def E2L(e):
    hbar = 6.582119569e-16
    omega = e/hbar
    frequency = omega /2/np.pi
    wavelength = 3e8/frequency
    return wavelength

def mkdir_p(path):
    """
    Create directory with subfolders (like Linux mkdir -p)

    :param path: Path to be created
    """
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

''' crystals '''
def calc_stretching(thetaB, ang_as, range_xy):
    L = range_xy/np.tan(thetaB-ang_as)
    delta = np.cos(thetaB+ang_as)/np.sin(thetaB-ang_as)*range_xy - L
    trange = np.abs(delta)/3e8
    return trange

def calcThetaO(thetaB, ang_as, n):
    theta1 = np.arcsin(1/n * np.sin(np.pi/2-thetaB+ang_as))
    theta2 = np.arcsin(n * np.sin(theta1-2*ang_as))
    thetaOut = np.pi/2-ang_as-theta2
    return thetaOut

def set_crystal_orient(cryst, e, ang_dif_pl, flip=0):
    if flip == 1:
        ang_dif_pl = ang_dif_pl+np.pi
    nvx, nvy, nvz = cryst.find_orient(
        e,ang_dif_pl)[0][2]             # outward normal vector to crystal surface
    tvx, tvy, _ = cryst.find_orient(
        e,ang_dif_pl)[0][0]      # central tangential vector
    cryst.set_orient(nvx,nvy,nvz,tvx,tvy)

''' wf '''
def get_field(wf):
    srwl.SetRepresElecField(wf._srwl_wf, 't')
    mesh = wf.params.Mesh
    E_real = wf.get_real_part()
    E_img = wf.get_imag_part()
    axis_x = np.linspace(mesh.xMin, mesh.xMax, mesh.nx)
    axis_y = np.linspace(mesh.yMin, mesh.yMax, mesh.ny)
    return axis_x, axis_y, E_real, E_img

def get_spectra(wf):
    # change the wavefront into frequency domain, where each slice in z represent a photon energy.
    srwl.SetRepresElecField(wf._srwl_wf, 'f')
    mesh = wf.params.Mesh
    dx = (mesh.xMax - mesh.xMin) / (mesh.nx - 1)        # spatial sampling resolution
    dy = (mesh.yMax - mesh.yMin) / (mesh.ny - 1)
    int0 = wf.get_intensity().sum(axis=0).sum(axis=0)   # intensity per slice
    #int0_00 = wf.get_intensity()[int(mesh.ny/2), int(mesh.nx/2),:]  # central intensity
    dSlice = (mesh.sliceMax - mesh.sliceMin)/(mesh.nSlices-1)   # photon energy sampling resolution
    axis_ev = np.arange(mesh.nSlices) * dSlice + mesh.sliceMin  # photon energy axis

    # get meaningful slices (>1% maximum intensity)
    int0max = max(int0)                                 # maximum intensity slice
    aw = [a[0] for a in np.argwhere(int0>int0max*0.01)] # meaningful indices
    aw = np.asarray(aw)
    #int0_mean = int0[min(aw):max(aw)]                   # meaningful intensity slices
    #axis_mean = np.arange(min(aw),max(aw)) * dSlice + mesh.sliceMin # meaningful photon energy axis
    return aw, axis_ev, int0

def get_temporal(wf):
    # change the wavefront into time domain, where each slice in z represent a time
    srwl.SetRepresElecField(wf._srwl_wf, 't')
    mesh = wf.params.Mesh
    dx = (mesh.xMax - mesh.xMin) / (mesh.nx - 1)        # spatial sampling resolution
    dy = (mesh.yMax - mesh.yMin) / (mesh.ny - 1)
    int0 = wf.get_intensity().sum(axis=0).sum(axis=0)   # intensity per slice
    #int0_00 = wf.get_intensity()[int(mesh.ny/2), int(mesh.nx/2),:]  # central intensity
    dSlice = (mesh.sliceMax - mesh.sliceMin)/(mesh.nSlices-1)   # time sampling resolution
    axis_t = np.arange(mesh.nSlices) * dSlice + mesh.sliceMin  # time axis

    # get meaningful slices (>1% maximum intensity)
    int0max = max(int0)                                 # maximum intensity slice
    aw = [a[0] for a in np.argwhere(int0>int0max*0.01)] # meaningful indices
    aw = np.asarray(aw)
    #int0_mean = int0[min(aw):max(aw)]                   # meaningful intensity slices
    #axis_t = np.arange(min(aw),max(aw)) * dSlice + mesh.sliceMin # meaningful time axis
    return aw, axis_t, int0

def get_tilt(wf, ori='V'):
    srwl.SetRepresElecField(wf._srwl_wf, 't')
    [xmin, xmax, ymin, ymax] = wf.get_limits()
    mesh = wf.params.Mesh
    if ori == 'V':
        axis = np.linspace(ymin, ymax, mesh.ny)
        tilt = wf.get_intensity()[:,int(mesh.nx/2),:]
    else:
        axis = np.linspace(xmin, xmax, mesh.nx)
        tilt = wf.get_intensity()[int(mesh.ny/2),:,:]
    return axis, tilt

''' I/O '''
def compress_save(wf, fname, ori='V'):
    axis_x, axis_y, E_real, E_img = get_field(wf)
    aw, axis_ev, int0 = get_spectra(wf)
    _, axis_t, int1 = get_temporal(wf)
    axis, tilt = get_tilt(wf,ori=ori)

    with h5py.File(fname,'w') as f:
        grp0 = f.create_group("spectrum")
        grp0.create_dataset("index", data=aw)
        grp0.create_dataset("axis", data=axis_ev)
        grp0.create_dataset("spectra", data=int0)

        grp1 = f.create_group("structure")
        grp1.create_dataset("axis", data=axis)
        grp1.create_dataset("tilt",data=tilt)
        grp1.create_dataset("ori",data=ori)

        grp2 = f.create_group("temporal")
        grp2.create_dataset("axis", data=axis_t)
        grp2.create_dataset("intensity", data=int1)

        grp3 = f.create_group("spatial")
        grp3.create_dataset("x", data=axis_x)
        grp3.create_dataset("y", data=axis_y)
        grp3.create_dataset("E_real", data=E_real)
        grp3.create_dataset("E_img", data=E_img)

def get_field_from_file(fname):
    with h5py.File(fname,'r') as f:
        axis_x = f["spatial/x"][:]
        axis_y = f["spatial/y"][:]
        E_real = f["spatial/E_real"][:]
        E_img = f["spatial/E_img"][:]
    return axis_x, axis_y, E_real, E_img

def get_spectra_from_file(fname):
    with h5py.File(fname,'r') as f:
        aw = f["spectrum/index"][:]
        axis_ev = f["spectrum/axis"][:]
        int_ev = f["spectrum/spectra"][:]
    return aw, axis_ev, int_ev

def get_temporal_from_file(fname):
    with h5py.File(fname,'r') as f:
        aw = f["spectrum/index"][:]
        axis_t = f["temporal/axis"][:]
        int_ev = f["temporal/intensity"][:]
    return aw, axis_ev, int_ev

def get_tilt_from_file(fname):
    with h5py.File(fname,'r') as f:
        axis = f["structure/axis"][:]
        tilt = f["structure/tilt"][:]
        ori = f["structure/ori"][...]
        axis_t = f["temporal/axis"][:]
    return axis, tilt, axis_t, ori

def get_lineout_from_file(fname):
   axis, tilt, axis_t, ori = get_tilt_from_file(fname)
   lineout = tilt.sum(axis=1)
   return axis*1e6, lineout, ori

def get_power_from_file(fname):
    axis, lineout, ori = get_lineout_from_file(fname)
    daxis = axis[1]-axis[0]
    power = lineout.sum()*daxis
    return power

def get_throughput_from_file(fname_in, fname_OE):
    power_in = get_power_from_file(fname_in)
    power_oe = get_power_from_file(fname_OE)
    return power_oe/power_in

''' plot wavefront '''
def plot_spatial(axis_x, axis_y, E_real, E_img, label=None, if_log=0):
    img = np.sum(E_real**2 + E_img**2, axis=-1)
    title = 'profile at '+label
    if if_log == 1:
        img = np.log(img)
        title = title+', log'
    plt.imshow(img, cmap='jet',
        extent = [axis_x.min()*1e6, axis_x.max()*1e6, axis_y.max()*1e6, axis_y.min()*1e6])
    plt.colorbar()
    if if_log == 1:
        cmin = np.max(img) - 10
        plt.clim(cmin)
    plt.axis('tight')
    plt.title(title,fontsize=18)
    plt.xlabel(r'x ($\mu$m)',fontsize=18)
    plt.ylabel(r'y ($\mu$m)',fontsize=18)

def plot_spatial_from_wf(wf, label=None, if_log=0):
    axis_x, axis_y, E_real, E_img = get_field(wf)
    plot_spatial(axis_x, axis_y, E_real, E_img, label=label, if_log=if_log)

def plot_spatial_from_file(fname, label=None, if_log=0):
    axis_x, axis_y, E_real, E_img = get_field_from_file(fname)
    plot_spatial(axis_x, axis_y, E_real, E_img, label=label, if_log=if_log)

def plot_spectra(aw, axis_ev, int0, color, label=None):
    plt.plot(axis_ev, int0/int0.max(), color, label=label)
    plt.ylim([-0.1,1.1])
    plt.legend(fontsize=18)
    plt.title('spectral energy\nmeaningful range: {}-{} eV'.format(
        round(axis_ev[aw[0]],2),round(axis_ev[aw[-1]],2)),fontsize=18)
    plt.xlabel('eV',fontsize=18)
    plt.ylabel('normalized spectral energy', fontsize=18)

def plot_temporal(wf, color, label=None, fov=1e30, pulse_duration = None):
    aw, axis_t, int0 = get_temporal(wf)
    index = np.argwhere(np.abs(axis_t)<fov/2)
    norm_t = int0/int0.max()

    if norm_t.min() > 0.01:
        print(str(round(norm_t.min(),3))+', not enough sampling!!!')
    plt.plot(axis_t[index.min():index.max()]*1e15,
        norm_t[index.min():index.max()], color, label=label)
    plt.ylim([-0.1,1.1])
    plt.legend(fontsize=18)
    plt.title('temporal structure ({} fs rms pulse)'.format(
        round(pulse_duration*1e15,2)),fontsize=18)
    plt.xlabel('t (fs)',fontsize=18)
    plt.ylabel('normalized temporal energy', fontsize=18)
    return aw, axis_t, int0




''' wavefront tilting '''
def plot_tilt(axis, tilt, axis_t, label=None, ori='V', if_log=0):
    tilt = tilt/tilt.max()
    tilt = tilt + 1e-30
    if ori == 'V':
        alabel = 'y'
    else:
        alabel = 'x'
    title = 'wavefront tilt at '+label
    if if_log == 1:
        tilt = np.log(tilt)
        title = title+', log'
    plt.imshow(tilt, cmap='jet',
              extent = [axis_t.max()*3e8*1e6, axis_t.min()*3e8*1e6, axis.max()*1e6, axis.min()*1e6])
    plt.colorbar()
    if if_log == 1:
        cmin = np.max(tilt)-10
        plt.clim(cmin)
    plt.axis('tight')
    plt.title(title, fontsize=18)
    plt.xlabel('z'+r' ($\mu$m)', fontsize=18)
    plt.ylabel(alabel+r' ($\mu$m)', fontsize=18)

def plot_tilt_from_wf(wf, label=None, ori='V', if_log=0):
    axis, tilt = get_tilt(wf, ori=ori)
    _, axis_t, _ = get_temporal(wf)
    plot_tilt(axis, tilt, axis_t, label=label, ori=ori, if_log=if_log)

def plot_tilt_from_file(fname, label=None, if_log=0):
    axis, tilt, axis_t, ori = get_tilt_from_file(fname)
    plot_tilt(axis, tilt, axis_t, label=label, ori=ori, if_log=if_log)

''' spatial spectral structure '''
def plot_tilt_freq(axis, tiltfft, axis_ev, label=None, ori='V', if_log=0):
    tiltfft = tiltfft/tiltfft.max()
    tiltfft = tiltfft + 1e-30
    if ori == 'V':
        alabel = 'y'
    else:
        alabel='x'
    title = 'spatial spectrum at '+label
    if if_log == 1:
        tiltfft = np.log(tiltfft)
        title = title+', log'
    plt.imshow(tiltfft, cmap='jet',
        extent = [axis_ev.max(), axis_ev.min(),axis.min()*1e6,axis.max()*1e6])
    plt.colorbar()
    if if_log == 1:
        cmin = np.max(tiltfft)-10
        plt.clim(cmin)
    plt.axis('tight')
    plt.title(title, fontsize=18)
    plt.xlabel('eV', fontsize=18)
    plt.ylabel(alabel+r'($\mu$m)', fontsize=18)

def plot_tilt_freq_from_wf(wf, label=None, ori='V', if_log=0):
    axis, tilt = get_tilt(wf, ori=ori)
    _, axis_ev, _ = get_spectra(wf)
    tiltfft = np.fft.fft(tilt, axis=1)
    tiltfft = np.abs(np.fft.fftshift(tiltfft,axes=1))
    plot_tilt_freq(axis, tiltfft, axis_ev, label=label, ori=ori, if_log=if_log)

def plot_tilt_freq_from_file(fname, label=None, if_log=0):
    with h5py.File(fname,'r') as f:
        axis = f["structure/axis"][:]
        tilt = f["structure/tilt"][:]
        ori = f["structure/ori"][...]
        
        axis_ev = f["spectrum/axis"][:]
    tiltfft = np.fft.fft(tilt, axis=1)
    tiltfft = np.abs(np.fft.fftshift(tiltfft,axes=1))
    plot_tilt_freq(axis, tiltfft, axis_ev, label=label, ori=ori, if_log=if_log)

''' spatial profile '''
def plot_lineout(axis, lineout, color, label=None, fov=1e30, ori='V', if_log=1, if_norm=0):
    if ori == 'V':
        aname = 'y'
    else:
        aname = 'x'
    if if_norm == 1:
        lineout /= lineout.sum()
    index = np.argwhere(np.abs(axis)<fov/2)
    axis = axis[index.min():index.max()]
    lineout = lineout[index.min():index.max()]
    plt.plot(axis, lineout, color, label=label)
    plt.legend(fontsize=18)
    plt.title('lineout', fontsize=18)
    plt.xlabel(aname+r' ($\mu$m}', fontsize=18)
    plt.ylabel('beam intensity (a.u.)', fontsize=18)
    if if_log == 1:
        plt.yscale('log')

def plot_lineout_from_wf(wf, color, label=None, fov=1e30, ori='V', if_log=1, if_norm=0):
    srwl.SetRepresElecField(wf._srwl_wf, 't')
    mesh = wf.params.Mesh
    [xmin, xmax, ymin, ymax] = wf.get_limits()
    img = wf.get_intensity().sum(axis=-1)
    if ori == 'V':
        # vertical plane
        axis = np.linspace(ymin, ymax, mesh.ny) * 1e6
        lineout = img[:,int(mesh.nx/2)]
    else:
        # horizontal plane
        axis = np.linspace(xmin, xmax, mesh.nx) * 1e6
        lineout = img[int(mesh.ny/2),:]
    plot_lineout(axis, lineout, color, label=label, fov=fov, ori=ori, if_log=if_log, if_norm=if_norm)

def plot_lineout_from_file(fname, color, label=None, fov=1e30, if_log=1, if_norm=0):
    axis, lineout, ori = get_lineout_from_file(fname)
    plot_lineout(axis, lineout, color, label=label, fov=fov, ori=ori, if_log=if_log, if_norm=if_norm)

''' quantification '''
def calc_bandwidth(aw, axis_ev):
    bw = np.abs(axis_ev[aw.max()] - axis_ev[aw.min()])
    return bw

def calc_spectral_response(aw, axis_ev, int_in, int_out):
    response = int_out/int_in
    return response

def calc_spectral_intensity(aw, axis_ev, int_in, int_out):
    axis_ev = axis_ev[aw.min():aw.max()]
    int_in = int_in[aw.min():aw.max()]
    int_out = int_out[aw.min():aw.max()]
    response = calc_spectral_response(aw, axis_ev, int_in, int_out)
    aw_out = np.argwhere(int_out>int_out.max()/100.0)
    spectral_intensity = np.mean(response[aw_out])
    return spectral_intensity

def plot_spectral_response(aw, axis_ev, int_in, int_out, color, label=None):
    axis_ev = axis_ev[aw.min():aw.max()]
    int_in = int_in[aw.min():aw.max()]
    int_out = int_out[aw.min():aw.max()]
    response = calc_spectral_response(aw, axis_ev, int_in, int_out)
    ylabel = 'ratio'
    plt.plot(axis_ev, response, color, label=label)
    plt.ylim([-0.1, 1.1])
    plt.xlabel('eV', fontsize=18)
    plt.ylabel(ylabel, fontsize=18)

''' ancient functions '''
def get_tslice_lineout(wf_holder,if_norm=1):
    N = len(wf_holder)
    ny, nx, nz = wf_holder[0].get_intensity().shape
    [xmin, xmax, ymin, ymax] = wf_holder[0].get_limits()
    axis = np.linspace(ymin, ymax, ny) * 1e6
    lineout_holder = np.zeros((N,ny))
    for i in range(N):
        lineout = wf_holder[i].get_intensity().sum(axis=-1)[:,int(nx/2)]
        if if_norm == 1:
            lineout = lineout/lineout.sum()
        lineout_holder[i] = lineout
    return axis, lineout_holder

def get_tslice_spectra(wf_holder):
    N = len(wf_holder)
    ny, nx, nz = wf_holder[0].get_intensity().shape
    spectra_holder = np.zeros((N,nz))
    for i in range(N):
        aw, axis, int0 = get_spectra(wf_holder[i])
        spectra_holder[i] = int0/int0.max()
    return aw, axis, spectra_holder
    
def get_tslice_temporal(wf_holder):
    N = len(wf_holder)
    ny, nx, nz = wf_holder[0].get_intensity().shape
    temporal_holder = np.zeros((N,nz))
    for i in range(N):
        aw, axis, int0 = get_temporal(wf_holder[i])
        temporal_holder[i] = int0/int0.max()
    return aw, axis, temporal_holder
    
def load_wavefront(nslice_t, dirname_prop):
    wf_holder = []
    for i in range(nslice_t):
        fname = dirname_prop + 'wavefront_focused_slice_'+str(i)+'.h5'
        mwf_temp = Wavefront()
        mwf_temp.load_hdf5(fname)
        wf_holder.append(mwf_temp)
    return wf_holder

