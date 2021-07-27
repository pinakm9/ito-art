import numpy as np
from numpy.lib.npyio import save
import tables
import os
import matplotlib.pyplot as plt
import utility as ut
import shutil, copy

class SDE:
    """
    Description:
        Implementation of a an SDE of the form dX_t = mu(t, X_t)dt + sigma(t, X_t)dB_t
    Args:
        space_dim: space dimension of the SDE
        mu: a function of time and space mu(t, X_t)
        sigma: a function of time and space sigma(t, X_t)
        name: name of the SDE
    """
    def __init__(self, space_dim, mu, sigma,  name='GenericSDE'):
        self.space_dim = space_dim
        self.mu = mu
        self.sigma = sigma
        self.name = name

    @ut.timer
    def evolve(self, initial_ensemble, record_path, begin_specs, end_specs):
        """
        Description:
            evolves an initial ensemble according to the SDE dynamics
        Args:
            initial_ensemble: the ensemble that starts the evolution
            record_path: file path ending with .h5 describing where to record the ensemble evolution
            begin_specs: 3-tuple specifying how to move forward and save in the beginning (num_frames, time_step, save_gap)
            end_specs: 3-tuple specifying how to move forward and save in the end (num_frames, time_step, save_gap)
        """
        self.num_particles = len(initial_ensemble) 
        hdf5 = tables.open_file(record_path, 'w')
        ens_folder = hdf5.create_group(hdf5.root, 'ensemble')
        new_ensemble = initial_ensemble
        frames = [0]
        # record the initial ensemble
        hdf5.create_array(ens_folder, 'time_0', initial_ensemble)
        num_steps, time_step, save_gap = begin_specs
        noise_std = np.sqrt(time_step)
        noise = np.random.normal(loc=0.0, scale=noise_std, size=(num_steps, self.num_particles, self.space_dim))
        for step in range(num_steps):
        # evolve ensemble with Euler-Maruyama according to begin_specs
            new_ensemble += self.mu(step*time_step, new_ensemble)*time_step + self.sigma(step*time_step, new_ensemble) * noise[step, :, :]
            # record the new ensemble is necessary
            if step % save_gap == 0:
                hdf5.create_array(ens_folder, 'time_' + str(step + 1), new_ensemble)
        frames = frames + [step + 1 for step in range(num_steps)]
        # add some extra useful information to the evolution file
        class Config(tables.IsDescription):
            num_steps = tables.Int32Col(pos=0)
            time_step = tables.Float32Col(pos=1)
            ensemble_size = tables.Int32Col(pos=2)
            dimension = tables.Int32Col(pos=3)
        tbl = hdf5.create_table(hdf5.root, 'config', Config)
        config = tbl.row
        config['num_steps'] = num_steps
        config['time_step'] = time_step
        config['ensemble_size'] = len(new_ensemble)
        config['dimension'] = new_ensemble.shape[-1]
        config.append()
        tbl.flush()

        # evolvbe according to end_specs
        num_steps, time_step, save_gap = end_specs
        noise_std = np.sqrt(time_step)
        noise = np.random.normal(loc=0.0, scale=noise_std, size=(num_steps, self.num_particles, self.space_dim))
        for step in range(num_steps):
        # evolve ensemble with Euler-Maruyama
            new_ensemble += self.mu(step*time_step, new_ensemble)*time_step + self.sigma(step*time_step, new_ensemble) * noise[step, :, :]
            # record the new ensemble is necessary
            if step % save_gap == 0:
                hdf5.create_array(ens_folder, 'time_' + str(step + 1 + begin_specs[0]), new_ensemble)

        config = tbl.row
        config['num_steps'] = num_steps
        config['time_step'] = time_step
        config['ensemble_size'] = len(new_ensemble)
        config['dimension'] = new_ensemble.shape[-1]
        config.append()
        tbl.flush()


        hdf5.close()
        return frames + [step + 1 + begin_specs[0] for step in range(num_steps)]