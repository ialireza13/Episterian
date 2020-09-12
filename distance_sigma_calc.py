import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from matplotlib import gridspec
#from simulate import simulate
from simulate_distance import simulate
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib
import matplotlib.cm as cm
import multiprocessing as mp
import pandas as pd

N = 100
N_ill = 1
Lx = Ly = 30
stepSize = 0.5
infection_rate = 0.01
tile_infection_rate = pollution_rate = 0.005
flow_rate = 0
tMax = 200 #Basically the run number right now
shuffled_pollution_activate = False
animatable_output = True
centralized_infectious = False
state_after_infection = 1 #1 for SEI, 2 for SI
opening_duration = 0 #flash_forward every ...
sigma_1 = sigma_2 = 0.3
n_sigma_2 = 0


args = N, N_ill, Lx, Ly, stepSize, infection_rate, pollution_rate\
, tile_infection_rate, flow_rate, tMax,\
shuffled_pollution_activate, animatable_output,\
centralized_infectious, state_after_infection,\
opening_duration, sigma_1, sigma_2, n_sigma_2


jobs = []
sigma_range = np.arange( 0, 2, 0.2 )
sigma_range[0] += 0.00001
#sigma_range = [0.01,0.5,1]

for sigma_ind ,sigma_1 in enumerate(sigma_range):

    args = (N, N_ill, Lx, Ly, stepSize, infection_rate, pollution_rate\
    , tile_infection_rate, flow_rate, tMax,\
    shuffled_pollution_activate, animatable_output,\
    centralized_infectious, state_after_infection,\
    opening_duration, sigma_1, sigma_2, n_sigma_2)

    jobs.append( ( (np.random.randint(10000),)+args ) )

with mp.Pool(mp.cpu_count()) as pool:
    p_r = pool.map_async(simulate, jobs)
    res = p_r.get()


res = np.array(res)
results = pd.DataFrame( np.zeros_like((sigma_range), float) );
results.columns = ['sigma']
results['sigma'] = sigma_range
results['mean'] = res.mean(1).mean(1)
results['std'] = res.mean(2).std(1)
results['err'] = results['std'] / np.sqrt(tMax)
file_string = 'N=' + str(N) + '-L=' + str(Lx) + '-sigma-distance.csv'
results.to_csv(file_string, index=False)
