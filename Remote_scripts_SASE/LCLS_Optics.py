import numpy as np

class Prism:
    """
    Class to represent a hard X-ray prism.

    Attributes
    ----------
    name: str
        Name of the device (e.g. PRM1)
    x_width: float
        horizontal size of prism
    y_width: float
        vertical size of prism
    slope: float
        thickness gradient (dimensionless)
    orientation: int
        defined as 0: thicker to +x; 1: thicker to +y; 2: thicker to -x; 3: thicker to -y
    material: str
        default to Beryllium
    z: float
        location along beamline
    dx: float
        horizontal de-centering relative to beam axis
    dy: float
        vertical de-centering relative to beam axis
    """

    def __init__(self, name, x_width=100e-6, y_width=100e-6, slope=100e-6, material='Be', z=0, dx=0, dy=0, orientation=0):
        """
        Method to create a prism
        Parameters
        ----------
        name: str
            Name of the device (e.g. PRM1)
        x_width: float
            horizontal size of prism
        y_width: float
            vertical size of prism
        slope: float
            thickness gradient (dimensionless)
        material: str
            default to Beryllium
        z: float
            location along beamline
        dx: float
            horizontal de-centering relative to beam axis
        dy: float
            vertical de-centering relative to beam axis
        orientation: int
            defined as 0: thicker to +x; 1: thicker to +y; 2: thicker to -x; 3: thicker to -y
        """

        # set some attributes
        self.name = name
        self.x_width = x_width
        self.y_width = y_width
        self.slope = slope
        self.material = material
        self.dx = dx
        self.dy = dy
        self.z = z
        self.orientation = orientation

        # load in CXRO data
        cxro_data = np.genfromtxt('{}.csv'.format(material), delimiter=',')
        self.energy = cxro_data[:, 0]
        self.delta = cxro_data[:, 1]
        self.beta = cxro_data[:, 2]

    def multiply(self, beam):

        # interpolate to find index of refraction at beam's energy
        delta = np.interp(beam.photonEnergy, self.energy, self.delta)
        beta = np.interp(beam.photonEnergy, self.energy, self.beta)

        # prism aperture
        aperture_x = (np.abs(beam.x - self.dx) < self.x_width / 2).astype(float)
        aperture_y = (np.abs(beam.y - self.dy) < self.y_width / 2).astype(float)

        thickness = np.zeros_like(beam.x)
        p1_x = 0
        p1_y = 0

        if self.orientation == 0:
            thickness = self.slope * (beam.x - self.dx + self.x_width / 2)
            p1_x = -delta * self.slope
            aperture = aperture_x
        elif self.orientation == 1:
            thickness = self.slope * (beam.y - self.dy + self.y_width / 2)
            p1_y = -delta * self.slope
            aperture = aperture_y
        elif self.orientation == 2:
            thickness = -self.slope * (beam.x - self.dx - self.x_width / 2)
            p1_x = delta * self.slope
            aperture = aperture_x
        elif self.orientation == 3:
            thickness = -self.slope * (beam.y - self.dy - self.y_width / 2)
            p1_y = delta * self.slope
            aperture = aperture_y

        # prism transmission based on beta and thickness profile
        transmission = np.exp(-beam.k0 * beta * thickness) * aperture

        # multiply by transmission
        if self.orientation == 0:
            beam.wavex *= transmission
        elif self.orientation == 1:
            beam.wavey *= transmission
        elif self.orientation == 2:
            beam.wavex *= transmission
        elif self.orientation == 3:
            beam.wavey *= transmission
        
        # adjust beam direction
        beam.rotate_beam(delta_ax=p1_x, delta_ay=p1_y)

    def propagate(self, beam):
        """
        Method to propagate beam through prism. Calls multiply.
        :param beam: Beam
            Beam object to propagate through prism. Beam is modified by this method.
        :return: None
        """
        self.multiply(beam)

class CRL_no_abs:
    """
    Class to represent parabolic compound refractive lenses (CRLs). This is a 1D implementation so the CRLs are square.

    Attributes
    ----------
    name: str
        Name of the device (e.g. CRL1)
    diameter: float
        Diameter beyond which the lenses absorb all photons. (meters)
    roc: float
        Lens radius of curvature. Lenses are actually parabolic but are labeled this way. (meters)
    E0: float or None
        photon energy in eV for calculating radius of curvature for a given focal length
    f: float or None
        focal length in meters for calculating radius of curvature for a given energy
    material: str
        Lens material. Currently only Be is implemented but may add CVD diamond in the future.
        Looks up downloaded data from CXRO.
    dx: float
        Lens de-centering along beam's x-axis.
    dy: float
        Lens de-centering along beam's y-axis.
    z: float
        z location of lenses along beamline.
    energy: (N,) ndarray
        List of photon energies from CXRO file (eV).
    delta: (N,) ndarray
        Real part of index of refraction. n = 1 - delta + 1j * beta
    beta: (N,) ndarray
        Imaginary part of index of refraction. n = 1 - delta + 1j * beta
    """

    def __init__(self, name, diameter=300e-6, roc=50e-6, E0=None, f=None, material='Be', z=0, dx=0, orientation=0):
        """
        Method to create a CRL object.
        :param name: str
            Name of the device (e.g. CRL1)
        :param diameter: float
            Diameter beyond which the lenses absorb all photons. (meters)
        :param roc: float
            Lens radius of curvature. Lenses are actually parabolic but are labeled this way. (meters)
        :param E0: float
            photon energy for calculating radius of curvature for a given focal length (eV)
        :param f: float
            focal length for calculating radius of curvature for a given energy (meters)
        :param material: str
            Lens material. Currently only Be is implemented but may add CVD diamond in the future.
        Looks up downloaded data from CXRO.
        :param z: float
            z location of lenses along beamline.
        :param dx: float
            Lens de-centering along beam's x-axis.
        :param orientation: int
            Whether or not this is a horizontal or vertical lens (0 for horizontal, 1 for vertical).
        """

        # set some attributes
        self.name = name
        self.diameter = diameter
        self.roc = roc
        self.E0 = E0
        self.f = f
        self.material = material
        self.dx = dx
        self.z = z
        self.global_x = 0
        self.global_y = 0
        self.orientation = orientation

        # load in CXRO data
        cxro_data = np.genfromtxt('{}.csv'.format(material), delimiter=',')
        self.energy = cxro_data[:, 0]
        self.delta = cxro_data[:, 1]
        self.beta = cxro_data[:, 2]

        # if these arguments are given then override default roc or even roc argument
        if self.f is not None and self.E0 is not None:
            # interpolate to find index of refraction at beam's energy
            delta = np.interp(self.E0, self.energy, self.delta)
            # calculate radius of curvature based on f and delta
            self.roc = 2 * delta * self.f

        elif self.E0 is not None:
            # interpolate to find index of refraction at beam's energy
            delta = np.interp(self.E0, self.energy, self.delta)
            self.f = self.roc/2/delta


    def multiply(self, beam):
        """
        Method to propagate beam through CRL
        :param beam: Beam
            Beam object to propagate through CRL. Beam is modified by this method.
        :return: None
        """

        if self.orientation == 0:
            beamx = beam.x
            beamz = beam.zx
            beamc = beam.cx
        else:
            beamx = beam.y
            beamz = beam.zy
            beamc = beam.cy

        # interpolate to find index of refraction at beam's energy
        delta = np.interp(beam.photonEnergy, self.energy, self.delta)
        beta = np.interp(beam.photonEnergy, self.energy, self.beta)

        # CRL thickness (for now assuming perfect lenses but might add aberrations later)
        # thickness = 2 * self.roc * (1 / 2 * ((beam.x - self.dx) ** 2 + (beam.y - self.dy) ** 2) / self.roc ** 2)
        thickness = 2 * self.roc * (1 / 2 * ((beamx - self.dx) ** 2) / self.roc ** 2)

        # lens aperture
        mask = (((beamx - self.dx) ** 2) < (self.diameter / 2) ** 2).astype(float)

        # subtract 2nd order and linear terms
        phase = -beam.k0 * delta * (thickness - 2 / 2 / self.roc * ((beamx - self.dx) ** 2))

        # 2nd order
        p2 = -beam.k0 * delta * 2 / 2 / self.roc
        # 1st order
        p1_x = p2 * 2 * (beamc - self.dx)

        # lens transmission based on beta and thickness profile
        transmission = np.exp(1j * phase) * mask

        # adjust beam properties
        new_zx = 1 / (1 / beamz + p2 * beam.lambda0 / np.pi)

        if self.orientation == 0:
            beam.change_z(new_zx=new_zx)
            delta_ax = p1_x * beam.lambda0 / 2 / np.pi
            beam.rotate_beam(delta_ax=delta_ax)
            # beam.ax += p1_x * beam.lambda0 / 2 / np.pi
            # multiply beam by CRL transmission function and any high order phase
            beam.wavex *= transmission * np.exp(1j * phase)
        else:
            beam.change_z(new_zy=new_zx)
            delta_ay = p1_x * beam.lambda0 / 2 / np.pi
            beam.rotate_beam(delta_ay=delta_ay)
            # beam.ay += p1_x * beam.lambda0 / 2 / np.pi
            # multiply beam by CRL transmission function and any high order phase
            beam.wavey *= transmission * np.exp(1j * phase)

        print('focal length: %.2f' % (-1/(p2*beam.lambda0/np.pi)))

    def propagate(self, beam):
        """
        Method to propagate beam through CRL. Calls multiply.
        :param beam: Beam
            Beam object to propagate through CRL. Beam is modified by this method.
        :return: None
        """
        self.multiply(beam)


class Force_tilt:
    """
    Class to represent an arbitrary tilt in beam.

    Attributes
    ----------
    name: str
        Name of the device (e.g. PRM1)
    delta_a: float
        change in beam angle (rad)
    orientation: int
        defined as 0: to +x; 1: to +y
    z: float
        location along beamline
    """
    def __init__(self, name, delta_a=100e-6, z=0, orientation=0):
        # set some attributes
        self.name = name
        self.delta_a = delta_a
        self.z = z
        self.orientation = orientation

    def multiply(self, beam):
        p1_x = 0
        p1_y = 0
        if self.orientation == 0:
            p1_x = self.delta_a
        if self.orientation == 1:
            p1_y = self.delta_a

        # adjust beam direction
        beam.rotate_beam(delta_ax=p1_x, delta_ay=p1_y)

    def propagate(self, beam):
        """
        Method to propagate beam through tilt. Calls multiply.
        :param beam: Beam
            Beam object to propagate through prism. Beam is modified by this method.
        :return: None
        """
        self.multiply(beam)

class Force_shift:
    """
    Class to represent an arbitrary tilt in beam.

    Attributes
    ----------
    name: str
        Name of the device (e.g. PRM1)
    x_offset: float
        change in beam position in x (m)
    y_offset: float
        change in beam position in y (m)
    z: float
        location along beamline
    """
    def __init__(self, name, x_offset=0, y_offset=0, z=0):
        # set some attributes
        self.name = name
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z = z

    def shift(self, beam):
        # offset beam
        beam.beam_offset(x_offset = self.x_offset, y_offset = self.y_offset)

    def propagate(self, beam):
        """
        Method to propagate beam through manual offset. Calls shift.
        :param beam: Beam
            Beam object to propagate through prism. Beam is modified by this method.
        :return: None
        """
        self.shift(beam)