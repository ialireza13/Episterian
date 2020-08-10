import numpy as np
import pandas as pd
from simulate import simulate
import multiprocessing as mp

shuffled_pollution_activate = False
centralized_infectious = False


if __name__ ==  '__main__':
    

    jobs = np.zeros((1), dtype=[('N', int), ('N_ill', int), ('Lx', int), ('Ly',int), ('step size',float), ('infection rate',float), ('pollution rate',float)\
                    , ('tile infection rate',float), ('flow rate',float), ('time',int)\
                    ,shuffled_pollution_activate, False, centralized_infectious
                     ] )

    Lx = 30
    Ly = 30
    stepSize = 0.5
    N = 100
    N_ill = 1

    realisations = 20000
    tMax = 1000
    #---------------N---N_ill--Lx--Ly----step---inf rate--poll rate--tile inf rate--flow rate--time -- 

    jobs[0] = tuple([N, N_ill, Lx, Ly, stepSize, 0.85       , 0.05,        0.05,       0,    tMax,shuffled_pollution_activate, False, centralized_infectious])
    #jobs[1] = tuple([N, N_ill, Lx, Ly, stepSize, 0.65       , 0.1,        0.1,       0,    tMax])
    #jobs[2] = tuple([N, N_ill, Lx, Ly, stepSize, 0.4       , 0.2,        0.2,       0    ,tMax])


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
    
            f.write( 'shuffled_pollution_activate=' + str( shuffled_pollution_activate ) + '\n' )
            f.write( 'centralized_infectious=' + str( centralized_infectious ) + '\n' )
            f.write( 'infection_rate=' + str( infection_rate ) + '\n' )
            f.write( 'tile_rate=' + str( tile_rate ) + '\n' )
            f.write( str(job)[1:-1] )

        
        #np.save( id_string , ts)
        #np.save(str(int(job['infection rate']*100)) + "-" + str(int(job['tile infection rate']*100))+"-err", errors)
