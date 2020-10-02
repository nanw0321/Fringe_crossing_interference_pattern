import numpy as np
import matplotlib.pyplot as plt
# OE thickness functions
def Flat_Mirror(x,y):
    z = np.zeros_like(x)
    return z

def Square_Aperture(x,y,dx,dy,xoff=0.,yoff=0.):
    z = np.ones(x.shape)*1e30
    window = (np.abs(x-xoff)<= dx/2) * (np.abs(y-yoff)<=dy/2)
    z[window] = 0.
    return z

def Perfect_lens(x,y,n,f,xoff=0,yoff=0):
    delta_d = f - np.sqrt(f**2-np.square(x-xoff)-np.square(y-yoff))
    z = -delta_d/(n-1)
    return z

def Perfect_lens_array(x,y,n,f,nlx,nly):
    nx,ny = x.shape
    z = np.zeros((nx,ny))
    npts = int(nx/nlx)
    dl = (x.max()-x.min())/(nx-1) * npts    # length of each micro lens
    if npts != nx/nlx:
        print('need integer multiples of small lens dimension')
    else:
        for il in range(nlx):
            xi = il * npts
            xf = (il+1)*npts
            for jl in range(nly):
                yi = jl * npts
                yf = (jl+1)*npts
                xl = x[xi:xf,yi:yf]
                yl = y[xi:xf,yi:yf]
                xlcent = xl.mean()
                ylcent = yl.mean()
                z[xi:xf,yi:yf] = Perfect_lens(xl,yl,n,f,xlcent,ylcent)
    return z

def Pinhole(x,y,r_hole,xoff=0.,yoff=0.):
    z = np.ones(x.shape)*1e30
    window = (np.square(x-xoff)+np.square(y-yoff))<=r_hole**2
    z[window]=0.
    return z

def Pinhole_array(x,y,r_hole,nhx,nhy):
    nx,ny = x.shape
    z = np.ones((nx,ny))*1e30
    npts = int(nx/nhx)
    dl = (x.max()-x.min())/(nx-1) * npts
    if npts != nx/nhx:
        print('need integer multiples of small pinhole dimension')
    else:
        for il in range(nhx):
            xi = il * npts
            xf = (il+1)*npts
            for jl in range(nhy):
                yi = jl * npts
                yf = (jl+1)*npts
                xl = x[xi:xf,yi:yf]
                yl = y[xi:xf,yi:yf]
                xlcent = xl.mean()
                ylcent = yl.mean()
                z[xi:xf,yi:yf] = Pinhole(xl,yl,r_hole,xlcent,ylcent)
    return z


def Prism(x,y,n,f):
    nx,ny = x.shape
    dl = x.max()-x.min()                    # width of the whole prism
    d_hole = dl/3
    #print('slope', slope)
    # For prism in x:
    z1 = np.zeros((nx,ny))                  # initialize prism
    index1 = x[0]>=d_hole/2.                # one edge
    nn = index1.sum()                       # get indicies of window boundary
    delta_d = (x[:,-nn:]-d_hole/2) * d_hole/f     # OPD needed for phase shift
    z1[:,-nn:] = -delta_d/(n-1)             # assign height
    z1 = z1 + np.fliplr(z1)
    # For prism in y:
    z2 = z1.T
    z = z1+z2
    return z
'''

def Prism(x,y,n,f):
    nx,ny = x.shape
    dl = (x.max()-x.min())/2   # width of the whole prism
    d_hole = dl*0.3
    hmax = -np.sqrt((f**2*n**2)/((2*n-n**2)**2) - ((dl-d_hole)**2)/(2*n-n**2)) + f*n/(2*n - n**2)   # maximum height needed
#     print('maximum height', hmax)
    slope = - hmax/(d_hole - dl)
    b = d_hole*hmax/(d_hole-dl)
#     print('slope', slope)
#     print('b',b)
    
    # For prism in x:
    z1 = np.zeros((nx,ny))                  # initialize prism
    index1 = (x[0]) > d_hole          # one edge
    nn = index1.sum()   # get indicies of window boundary

    z1[:,-nn:] = slope * (x[:,-nn:]) + b # assign height
    z1 = z1 + np.fliplr(z1)
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     ax.contour3D(xx, yy, z1, 50, cmap='binary')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     plt.figure()
#     plt.plot(z1[1,:])
    # For prism in y:
    z2 = z1.T
    z = z1+z2
    return z
'''

def Prism_array(x,y,n,f,npx,npy):
    # input npx, npy: # of individual prisms in each dimension.
    # output z: height profile of the entire optical element.
    nx,ny = x.shape                         # dimension of the input 
    z = np.ones((nx,ny))*1e30               # initialize height profile
    npts = int(nx/npx)                      # dimension of a single prism
    dl = (x.max()-x.min())/(nx-1) * npts    # width of a single prism
    if npts != nx/npx:
        print('need integer multiples of small prism dimension')
    else:
        for il in range(npx):
            xi = il * npts              # indices for a single prism
            xf = (il+1)*npts
            for jl in range(npy):
                yi = jl * npts
                yf = (jl+1)*npts
                xl = x[xi:xf,yi:yf]     # x&y axis for a single prism
                yl = y[xi:xf,yi:yf]
                xl = xl - xl.mean()
                yl = yl - yl.mean()
                z[xi:xf,yi:yf] = Prism(xl,yl,n,f)
    return z

def Double_slit(x,y,wid,sep,xoff=0.,yoff=0.):
    nx,ny = x.shape
    z = np.ones((nx,ny))*1e30
    for i in range(nx):
        xi = x[0,i]
        if (xi-xoff>=-sep/2-wid) and (xi-xoff<=-sep/2+wid):
            z[:,i] = 0
        elif (xi-xoff>=sep/2-wid) and (xi-xoff<=sep/2+wid):
            z[:,i] = 0
    return z

# function to calculate optical path differnece and amplitude transmission
def Calc_OPD_and_AmpTr(srwTr, thicknessProfData, n, d_abs, nlx=1, nly=1, roix=1, roiy=1):
    N = len(thicknessProfData[0])
    auxMesh = srwTr.mesh
    nx = auxMesh.nx
    ny = auxMesh.ny
    if thicknessProfData.shape==(nx,ny):
        # amplitude transmission
        Tr = np.exp(-thicknessProfData/d_abs)
        # OPD
        OPD = thicknessProfData*(n-1)
        if thicknessProfData.max() >= 1e30:
            # pinhole
            Tr[thicknessProfData>=1e30] = 0.
            OPD[thicknessProfData>=1e30] = 0.
        #for iy in range(ny):
        #    for ix in range(nx):
        err = 0
        for iy in (np.arange(int(ny*roiy/nly))+int(ny*(nly-roiy)/nly/2)):
            for ix in (np.arange(int(nx*roix/nlx))+int(nx*(nlx-roix)/nlx/2)):
                try:
                    ofst = 2*ix + 2*nx*iy
                    srwTr.arTr[ofst]=Tr[iy,ix]
                    #srwTr.arTr[ofst] = 1.0
                    srwTr.arTr[ofst+1]=OPD[iy,ix]
                except:
                    err+=1
        if err > 0:
            print('lens roi > lens array by {} pixels'.format(err))
    else:
        print('OE shape not matched')

def Calc_OPD_and_AmpTr_Mirror(srwTr, heightProfData, theta):
    N = len(heightProfData)
    auxMesh = srwTr.mesh
    nx = auxMesh.nx
    ny = auxMesh.ny
    if heightProfData.shape==(nx,ny):
        for iy in range(ny):
            for ix in range(nx):
                ofst = 2*ix + 2*nx*iy
                srwTr.arTr[ofst] = 1.
                srwTr.arTr[ofst+1] = 2. * heightProfData[iy,ix] * np.sin(theta)
    else:
        print('OE shape not matched')

def Single_Prism(x,y,n,f,ori='x'):
    nx,ny = x.shape
    dl = x.max()-x.min()                    # width of the whole prism
    #print('slope', slope)
    # For prism in x:
    z1 = np.zeros((nx,ny))                  # initialize prism
    index1 = x[0]>=0
    nn = index1.sum()
    delta_d = x[:,-nn:] * (y.max()-y.min())/2/f  # OPD needed for phase shift
    
    z1[:,-nn:] = -delta_d/(n-1)             # assign height
    z1 = z1 + np.fliplr(z1)
    # For prism in y
    z2 = 0
    if ori == 'y':
        z2 = z1.T
        z1 = 0
    z = z1+z2
    return z
