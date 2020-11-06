''' input parameters '''
from input import *

''' define beamline '''
from define_beamline_ray_tracing import *

# initialize optical elements
devices_VCC = define_branch_VCC(E0=E0, z_off=z_off, d23=d23, alphaAsym=alphaAsym, f_lens=f_lens, FOV1=FOV1, FOV2=FOV2, N=N)
branch_VCC = beamline.Beamline(devices_VCC, ordered=True)
        
devices_CC = define_branch_CC(E0=E0, d23=d23, f_lens=f_lens, FOV1=FOV1, FOV2=FOV2, N=N)
branch_CC = beamline.Beamline(devices_CC, ordered=True)


''' propagation '''
blockPrint()
tstart = time.time()
beam_params['photonEnergy'] = E0
pulse_VCC = beam.Pulse(beam_params=beam_params, tau=tau, time_window=window, SASE=True, num_spikes=num_spikes)
pulse_CC = copy.deepcopy(pulse_VCC)

pulse_VCC.propagate(beamline=branch_VCC, screen_names=['im2','im3','im4'])
pulse_CC.propagate(beamline=branch_CC, screen_names=['im2','im3','im4'])
tfin = time.time()
enablePrint()
print('propagation lasted {}s'.format(round(tfin-tstart,2)))

''' I/O '''
path = 'LCLS_output/'; make_dir(path)
case_path = path+'ray_tracing/'; make_dir(case_path)
fig_path = case_path+'{} fs/'.format(tau); make_dir(fig_path)


''' beam profiles '''
im_names = ['im2', 'im3', 'im4']
part_names = ['1.mono', '2.lens_focus', '3.crossing']

for i in tqdm(range(len(im_names))):
    im_name = im_names[i]
    part_name = part_names[i]
    
    # spatial projection (xy)
    pulse_VCC.imshow_projection(im_name)
    plt.savefig(fig_path+'profile_VCC_{}.png'.format(part_name))
    pulse_CC.imshow_projection(im_name)
    plt.savefig(fig_path+'profile_CC_{}.png'.format(part_name))
    plt.close('all')
    # time and energy component slice (x/y vs t/E)
    for dim in ['x','y']:
        pulse_VCC.imshow_time_slice(im_name, dim=dim)
        plt.savefig(fig_path+'tilt_{}_VCC_{}.png'.format(dim,part_name))
        pulse_CC.imshow_time_slice(im_name, dim=dim)
        plt.savefig(fig_path+'tilt_{}_CC_{}.png'.format(dim,part_name))
        plt.close('all')
        pulse_VCC.imshow_energy_slice(im_name, dim=dim)
        plt.savefig(fig_path+'spectrum_{}_VCC_{}.png'.format(dim, part_name))
        pulse_CC.imshow_energy_slice(im_name, dim=dim)
        plt.savefig(fig_path+'spectrum_{}_CC_{}.png'.format(dim, part_name))
        plt.close('all')

''' crossing '''
offsets = np.linspace(-2,2,5)*10
for offset in tqdm(offsets):
    im_name = im_names[-1]
    part_name = part_names[-1]+'_{}fs_offset'.format(offset)
    
    # add branches
    t_shift = find_shift(pulse_VCC, pulse_CC, im_name)+offset
    pulse_sum = pulse_VCC.add_pulse(pulse_CC, t_shift)
    
    # spatial projection (xy)
    pulse_sum.imshow_projection(im_name)
    plt.savefig(fig_path+'profile_Sum_{}.png'.format(part_name))
    plt.close('all')
    # time and energy component slice (x/y vs t/E)
    for dim in ['x','y']:
        pulse_sum.imshow_time_slice(im_name, dim=dim)
        plt.savefig(fig_path+'tilt_{}_Sum_{}.png'.format(dim,part_name))
        pulse_sum.imshow_energy_slice(im_name, dim=dim)
        plt.savefig(fig_path+'spectrum_{}_Sum_{}.png'.format(dim, part_name))
        plt.close('all')
    pulse_sum = 0

