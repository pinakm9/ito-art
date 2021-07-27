# add modules to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
root = str(script_dir.parent.parent)
print(root)
sys.path.insert(0, root + '/modules')

# import required modules
import sde
import numpy as np


# set parameters
np_dtype = np.float64
beta = 100.0
s = np.sqrt(2.0/beta)


# create an SDE object
def mu(t, X_t):
    x, y = X_t[:, 0], X_t[:, 1]
    z = 4.0 * (1.0 - x*x - y*y)
    return np.array([x*z, y*z], dtype=np_dtype).T

def sigma(t, X_t):
    return s 

def get_sde():
    return sde.SDE(space_dim=2, mu=mu, sigma=sigma, name='circle')