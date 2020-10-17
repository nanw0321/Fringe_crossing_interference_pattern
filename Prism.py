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
        beam.ax += p1_x
        beam.ay += p1_y

    def propagate(self, beam):
        """
        Method to propagate beam through prism. Calls multiply.
        :param beam: Beam
            Beam object to propagate through prism. Beam is modified by this method.
        :return: None
        """
        self.multiply(beam)