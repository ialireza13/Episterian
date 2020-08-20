import numpy as np
from simulate import simulate
import multiprocessing as mp

if __name__ ==  '__main__': 

    N = 100
    N_ill = 1
    Lx = Ly = 30
    stepSize = 0.5
    infection_rate = 0.06
    tile_infection_rate = pollution_rate = 0.01
    flow_rate = 0
    shuffled_pollution_activate = False
    animatable_output = False
    centralized_infectious = False
    state_after_infection = 1 #1 for SEI, 2 for SI
    opening_duration = 0 #flash_forward every ...
    # sigma_1 = 4.0
    sigma_2 = 0
    n_sigma_2 = 0
    
    realisations = 10000
    tMax = 6000

    sigma = (0.00001, 0.05, 6.00001)
    
    sigma = np.arange(start=sigma[0], step=sigma[1], stop=sigma[2])
    rand_id = str(np.random.randint(100000000))
    for i in range(len(sigma)):

        args = N, N_ill, Lx, Ly, stepSize, infection_rate, pollution_rate\
                , tile_infection_rate, flow_rate, tMax,\
                shuffled_pollution_activate, animatable_output,\
                centralized_infectious, state_after_infection,\
                opening_duration, sigma[i], sigma_2, n_sigma_2
                
        with mp.Pool(mp.cpu_count()) as pool:
            p_r = pool.map_async(simulate, [(np.random.randint(10000000),)+args for i in range(realisations)])
            res = p_r.get()
        for j in range(len(res)):
            res[j] = (res[j]['from_per'].cumsum(), res[j]['from_env'].cumsum())
        
        id_string = 'sigma=' + str(sigma[i]) + ', ' + rand_id
    
        np.save(id_string, res)
    
    with open("info sweep " + rand_id, "w") as f: 
        f.write( 'parameter sweep:\n\n' )
        
        f.write( str(args)[1:-1] )
