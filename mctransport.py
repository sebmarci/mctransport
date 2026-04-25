import numpy as np
from utils import *
import configparser

# --- UNIT CONVENTION ---
# Energy:                         keV
# Length (height, radius, etc):   cm
# Macroscopic cross section:      1/cm
# Density:                        g/cm^3

# Constants for selecting collision events
EVENT_SCATTER = 1
EVENT_ABSORPTION = 2
EVENT_PAIRPROD = 3

class Simulation:
    
    def __init__(self, input_file = 'input.ini'):
        config = configparser.ConfigParser()
        config.read(input_file)
        
        self.n_photons = int(config['Simulation']['N_source_photons'])
        
        detector_r = float(config['Detector']['radius'])
        detector_h = float(config['Detector']['height'])
        detector_dens = float(config['Detector']['density'])
        detector_fwhm = float(config['Detector']['fwhm'])
        source_energy = float(config['Source']['energy'])
        source_pos = np.array([float(d) for d in config['Source']['position'].split(',')])
        
        self.statistics = {
            'Misses': 0,
            'Hits': 0,
            'Escapes': 0,
            'Scatters': 0,
            'Absorptions': 0,
            'Pair productions': 0
        }
        
        self.detector = Detector(detector_r, detector_h, detector_fwhm)
        self.source = Source(source_energy, source_pos, detector_r, detector_h)
        self.cross_sections = CrossSections(detector_dens)
        
    def simulate_single_photon(self, photon):
                    
        while True:
        
            free_path = self.cross_sections.get_free_path(photon.energy)
            
            if free_path > self.detector.intersect_in(photon):
                self.statistics['Escapes'] += 1
                break
            
            photon.propagate(free_path)
            event = self.cross_sections.get_collision_event(photon.energy)
            
            if event == EVENT_SCATTER:
                self.detector.absorb_energy(photon.scatter())
                self.statistics['Scatters'] += 1
                
            elif event == EVENT_ABSORPTION:
                self.detector.absorb_energy(photon.energy)
                self.statistics['Absorptions'] += 1
                break
            
            else:
                e_absorb, p1, p2 = photon.pair_production()
                
                self.statistics['Pair productions'] += 1
                self.detector.absorb_energy(e_absorb)
                self.simulate_single_photon(p1)
                self.simulate_single_photon(p2)
                break
                
    def run_simulation(self):
        
        for _ in range(self.n_photons):
            
            photon = self.source.emit()
            d_intersect = self.detector.intersect_out(photon)
            
            if d_intersect == np.inf:
                self.statistics['Misses'] += 1
                continue
            
            self.statistics['Hits'] += 1
            
            self.simulate_single_photon(photon)
            self.detector.register_energy_sum()
            
class Photon:
    
    def __init__(self, position, direction, energy):
        self.position = position
        self.direction = direction
        self.energy = energy
        
    def propagate(self, distance):
        self.position += distance * self.direction
    
    def scatter(self):
        energy_out, direction = compton_scatter(self.energy, self.direction)
        absorbed_energy = self.energy - energy_out
        
        self.direction = direction
        self.energy = energy_out
        
        return absorbed_energy
    
    def pair_production(self):
        assert self.energy >= 1022, f'Photon cannot undergo pair production, E = {self.energy} keV < 1022 keV'
        
        absorbed_energy = self.energy - 1022
        pair_direction = isotropic_direction()
        
        photon1 = Photon(
            self.position.copy(),
            pair_direction.copy(),
            511
        )
        photon2 = Photon(
            self.position.copy(),
            -pair_direction.copy(),
            511
        )
        
        return (absorbed_energy, photon1, photon2)
    
class Detector:
    
    def __init__(self, radius, height, fwhm):
        self.radius = radius
        self.height = height
        self.sigma = fwhm/(2*np.sqrt(2*np.log(2)))
        self.energy_buffer = []
        self.registered_energies = []
    
    def intersect_in(self, photon: Photon):
        return intersect_cylinder_in(
            photon.position,
            photon.direction,
            self.height/2,
            -self.height/2,
            self.radius
        )
    
    def intersect_out(self, photon: Photon):
        return intersect_cylinder_out(
            photon.position,
            photon.direction,
            self.height/2,
            -self.height/2,
            self.radius
        )

    def absorb_energy(self, energy: float):
        self.energy_buffer.append(energy)

    def register_energy_sum(self):
        energy_sum = sum(self.energy_buffer)
        self.energy_buffer = []
        
        self.registered_energies.append(
            np.random.normal(energy_sum, self.sigma)
        )
        
class Source:
    
    def __init__(self, energy, position, det_radius, det_height):
        self.energy = energy
        self.position = position
        
        rg = np.sqrt(det_radius*det_radius + det_height*det_height/4)
        dist = np.linalg.norm(position)

        self.cone_angle = np.asin(rg/dist)
        self.det_direction = -position / np.linalg.norm(position)
        
    def emit(self) -> Photon:
        n = isotropic_direction_in_angle(self.cone_angle)
        n_transformed = transform_direction(n, self.det_direction)

        return Photon(
            self.position.copy(),
            n_transformed,
            self.energy
        )
        
class CrossSections:
    
    def __init__(self, density: float, file_path: str = 'xs_unique.txt'):
        xsdata = np.loadtxt(file_path)
        xsdata[:, 1:] *= density
        
        self.energies = xsdata[:, 0]
        
        self.xs_total = xsdata[:, 1:].sum(axis = 1)
        self.p_scatter = xsdata[:, 1] / self.xs_total
        self.p_absorption = xsdata[:, 2] / self.xs_total
    
    def get_free_path(self, energy) -> float:
        
        xs_total = np.interp(energy, self.energies, self.xs_total)
        return -1 / xs_total * np.log(np.random.rand())
    
    def get_collision_event(self, energy):
        
        ps = np.interp(energy, self.energies, self.p_scatter)
        pa = np.interp(energy, self.energies, self.p_absorption)
        
        r = rand()
        
        if r < ps:
            return EVENT_SCATTER
        elif r < ps + pa:
            return EVENT_ABSORPTION
        else:
            return EVENT_PAIRPROD