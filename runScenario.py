import numpy as np
import pandas as pd
from simulate import simulate
import multiprocessing as mp

shuffled_pollution_activate = False
centralized_infectious = False


if __name__ ==  '__main__':
    

    jobs = np.zeros((3), dtype=[('N', int), ('N_ill', int), ('Lx', int), ('Ly',int), ('step size',float), ('infection rate',float), ('pollution rate',float)\
                    , ('tile infection rate',float), ('flow rate',float), ('time',int)\
                    ,('shuffled_pollution',bool), ('animation',bool), ('infectious_center', bool),\
                        ('state_after_infection',int), ('opening_duration',int), ('sigma_1', float),\
                            ('sigma_2', float),('n_sigma_2',int)
                     ] )

    N = 100
    N_ill = 1
    Lx = Ly = 30
    stepSize = 0.5
    infection_rate = 0.1
    tile_infection_rate = pollution_rate = 0.02
    flow_rate = 0
    shuffled_pollution_activate = False
    animatable_output = False
    centralized_infectious = False
    state_after_infection = 1 #1 for SEI, 2 for SI
    opening_duration = 0 #flash_forward every ...
    sigma_1 = 4.0
    sigma_2 = 0
    n_sigma_2 = 0
    
    realisations = 5
    tMax = 20
    
    args1 = N, N_ill, Lx, Ly, stepSize, infection_rate, pollution_rate\
        , tile_infection_rate, flow_rate, tMax,\
        shuffled_pollution_activate, animatable_output,\
        centralized_infectious, state_after_infection,\
        opening_duration, sigma_1, sigma_2, n_sigma_2
    
    N = 100
    N_ill = 1
    Lx = Ly = 30
    stepSize = 0.5
    infection_rate = 0.1
    tile_infection_rate = pollution_rate = 0.02
    flow_rate = 0
    shuffled_pollution_activate = False
    animatable_output = False
    centralized_infectious = False
    state_after_infection = 1 #1 for SEI, 2 for SI
    opening_duration = 0 #flash_forward every ...
    sigma_1 = 0.001
    sigma_2 = 0
    n_sigma_2 = 0
    
    args2 = N, N_ill, Lx, Ly, stepSize, infection_rate, pollution_rate\
        , tile_infection_rate, flow_rate, tMax,\
        shuffled_pollution_activate, animatable_output,\
        centralized_infectious, state_after_infection,\
        opening_duration, sigma_1, sigma_2, n_sigma_2
    #---------------N---N_ill--Lx--Ly----step---inf rate--poll rate--tile inf rate--flow rate--time -- 

    N = 100
    N_ill = 1
    Lx = Ly = 30
    stepSize = 0.5
    infection_rate = 0.1
    tile_infection_rate = pollution_rate = 0.02
    flow_rate = 0
    shuffled_pollution_activate = False
    animatable_output = False
    centralized_infectious = False
    state_after_infection = 1 #1 for SEI, 2 for SI
    opening_duration = 0 #flash_forward every ...
    sigma_1 = 0.001
    sigma_2 = 4.0
    n_sigma_2 = 50
    
    args3 = N, N_ill, Lx, Ly, stepSize, infection_rate, pollution_rate\
        , tile_infection_rate, flow_rate, tMax,\
        shuffled_pollution_activate, animatable_output,\
        centralized_infectious, state_after_infection,\
        opening_duration, sigma_1, sigma_2, n_sigma_2

    jobs[0] = tuple(args1)
    jobs[1] = tuple(args2)
    jobs[2] = tuple(args3)


    for job in jobs:
        works = [job for i in range(realisations)]
        
        with mp.Pool(mp.cpu_count()) as pool:
            p_r = pool.map_async(simulate, works)
            results = p_r.get()
            
        for j in range(len(results)):
            results[j] = (results[j]['from_per'].cumsum(), results[j]['from_env'].cumsum())
            
        ts = np.mean(results, axis=0)
        errors = np.std(results, axis=0)
        
        rand_string = str(np.random.randint(100000000))
        id_string = 'i_r='+ str(int(job['infection rate']*100)) + ', t_r' + str(int(job['tile infection rate']*100)) + ', ' + rand_string
        
        np.save(id_string, results)
        
        #with open("info "+rand_string, "w") as f: 
            #f.write( str(job)[1:-1] )
        
        with open("info "+rand_string, "w") as f: 
            f.write( 'scenario:\n\n' )
            #if shuffled_pollution_activate:
                #f.write('shuffled\n')
            #else:
                #f.write('not shuffled\n')
    
            # f.write( 'shuffled_pollution_activate=' + str( shuffled_pollution_activate ) + '\n' )
            # f.write( 'centralized_infectious=' + str( centralized_infectious ) + '\n' )
            # f.write( 'infection_rate=' + str( infection_rate ) + '\n' )
            # f.write( 'tile_rate=' + str( tile_rate ) + '\n' )
            f.write( str(job)[1:-1] )

        
        #np.save( id_string , ts)
        #np.save(str(int(job['infection rate']*100)) + "-" + str(int(job['tile infection rate']*100))+"-err", errors)
