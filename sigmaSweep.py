import numpy as np
from simulate import simulate
import multiprocessing as mp

if __name__ ==  '__main__': 

    N = 100
    N_ill = 1
    Lx = Ly = 10
    stepSize = 0.5
    infection_rate = 0.01
    tile_infection_rate = pollution_rate = 0.005
    flow_rate = 0
    shuffled_pollution_activate = False
    animatable_output = False
    centralized_infectious = False
    state_after_infection = 1 #1 for SEI, 2 for SI
    opening_duration = 0 #flash_forward every ...
    # sigma_1 = 4.0
    sigma_2 = 0
    n_sigma_2 = 0
    
    realisations = 7000
    tMax = 6000
    
    # sigma = (0.0001, 0.5, 10.10001)
    # sigma = np.arange(start=sigma[0], step=sigma[1], stop=sigma[2])
    sigma = [5.3]
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
        output = np.zeros((2,2,tMax))
        output[0] = np.mean(res,0)
        output[1] = np.std(res,0)
        id_string = 'sigma=' + str(sigma[i]) + ', ' + rand_id

        np.save(id_string, output)
    
    with open("info sweep " + rand_id, "w") as f: 
        f.write( 'parameter sweep:\n\n' )
        
        f.write( str(args)[1:-1] )
