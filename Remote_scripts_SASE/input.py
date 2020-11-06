import time, h5py, os, copy
import numpy as np
import matplotlib.pyplot as plt
from lcls_beamline_toolbox.xraybeamline2d import beam1d as beam, optics1d as optics, beamline1d as beamline
from tqdm import tqdm

from Functions import *
from LCLS_Optics import *

''' input parameters '''
N = 256       # number of sampling points
E0 = 9.9e3    # photon energy in eV
z_off=.12     # distance offset between VCC crystal 1 and CC crystal 1
d23=.2        # distance between VCC crystal 2 and 3
f_lens = .085 # spectrometer lens focal distance
FOV1 = 2e-3   # [m]
FOV2 = 1e-3   # [m]
alphaAsym = np.deg2rad(2)

tau = 20.
window = 4000

num_spikes = 3

# parameter dictionary. z_source is in LCLS coordinates (20 meters upstream of undulator exit)
beam_params = {
    'photonEnergy': E0,
    'N': N,
    'sigma_x': 50e-6/(2*np.sqrt(2*np.log(2))),
    'sigma_y': 50e-6/(2*np.sqrt(2*np.log(2))),
    'rangeFactor': 5,
    'scaleFactor': 10,
    'z_source': 0
}

''' dimension check '''
hbar = 0.6582
bw = 2 * np.sqrt(2) * hbar * np.sqrt(np.log(2)) / tau
E_range = 6 * bw * num_spikes
f_range = E_range / 4.136
Nz = window*f_range
print('E_range: {}eV, f_range: {}, Nz: {}'.format(E_range, f_range, Nz))