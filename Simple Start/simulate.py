import numpy as np
from scipy.spatial.distance import pdist, squareform
#from functions import *
import pathlib
from functions_for_distance import *


dt = 0.1

def simulate(args):
    shuffled_pollution_activate = False
    animatable_output = False
    centralized_infectious = False
    state_after_infection = 1 #1 for E, 2 for I
    opening_duration = 0 # 0 indicates no flash_forward
    sigma_1 = 2
    sigma_2 = 0
    n_sigma_2 = 0
    if len(args) == 11:
        random_seed, N, N_ill, Lx, Ly, stepSize, infection_rate, pollution_rate\
        , tile_infection_rate, flow_rate, tMax = args
        
    elif len(args) == 12:

        random_seed, N, N_ill, Lx, Ly, stepSize, infection_rate, pollution_rate\
        , tile_infection_rate, flow_rate, tMax, shuffled_pollution_activate = args
        
    elif len(args) == 13:

        random_seed, N, N_ill, Lx, Ly, stepSize, infection_rate, pollution_rate\
        , tile_infection_rate, flow_rate, tMax,\
        shuffled_pollution_activate, animatable_output = args
        
    elif len(args) == 14:

        random_seed, N, N_ill, Lx, Ly, stepSize, infection_rate, pollution_rate\
        , tile_infection_rate, flow_rate, tMax,\
        shuffled_pollution_activate, animatable_output, centralized_infectious = args

    elif len(args) == 15:

        random_seed, N, N_ill, Lx, Ly, stepSize, infection_rate, pollution_rate\
        , tile_infection_rate, flow_rate, tMax,\
        shuffled_pollution_activate, animatable_output,\
        centralized_infectious, state_after_infection = args
        
    elif len(args) == 16:

        random_seed, N, N_ill, Lx, Ly, stepSize, infection_rate, pollution_rate\
        , tile_infection_rate, flow_rate, tMax,\
        shuffled_pollution_activate, animatable_output,\
        centralized_infectious, state_after_infection,\
        opening_duration = args

    elif len(args) == 19:

        random_seed, N, N_ill, Lx, Ly, stepSize, infection_rate, pollution_rate\
        , tile_infection_rate, flow_rate, tMax,\
        shuffled_pollution_activate, animatable_output,\
        centralized_infectious, state_after_infection,\
        opening_duration, sigma_1, sigma_2, n_sigma_2 = args


    elif len(args) == 21:

        random_seed, N, N_ill, Lx, Ly, stepSize, infection_rate, pollution_rate\
        , tile_infection_rate, flow_rate, tMax,\
        shuffled_pollution_activate, animatable_output,\
        centralized_infectious, state_after_infection,\
        opening_duration, sigma_1, sigma_2, n_sigma_2, algorithm, dt_ratio = args

        

    else:
        print("Number of arguments don't match for simulate.")
    
    #dt_ratio = 1
    dt = 0.1 / dt_ratio
    tMax *= dt_ratio
 
    
    # set random seed for each process
    np.random.seed(random_seed)
    #np.random.seed(0)
    
    #predestination:
    predicted_destin_arrivals = 1000
    predestination_x = np.random.randint( 1, Lx - 2, (N, predicted_destin_arrivals) )
    predestination_y = np.random.randint( 1, Ly - 2, (N, predicted_destin_arrivals) )
    
    arrival_num = np.zeros(N, 'int')    

    
    
    # print('rnd',str(np.random.randint(10000)))
    tile_x_num = Lx-1
    tile_y_num = Ly-1
    
    tile_x_size = Lx / tile_x_num
    tile_y_size = Ly / tile_y_num

    disease_timeline = np.zeros( ( tMax ), dtype=[ ('from_per',int), ('from_env',int)] )

    agents = np.zeros((N), dtype=[('tile_x',int), ('tile_y',int), ('health',int), ('sigma', float)] )
    
    agents['sigma'] = sigma_1 # temp
    agents['sigma'][:n_sigma_2] = sigma_2
    
    positions = np.zeros((N, 4)) #x, y, vx, vy, sigma(interaction constant)
    distances = squareform(pdist(positions[:, :2]))
    destinations = np.zeros((N, 2), int)
    
    
    pollution = np.zeros( (tile_x_num, tile_y_num),float )

    force = np.zeros((N, 2))

    
    if shuffled_pollution_activate:
        shuffled_x = np.arange(tile_x_num)
        shuffled_y = np.arange(tile_y_num)
        np.random.shuffle(shuffled_x)
        np.random.shuffle(shuffled_y)
        
        fake_pollution = np.zeros( (tile_x_num, tile_y_num),float )
    
    if animatable_output:
        pollution_history = np.zeros( (tMax, tile_x_num, tile_y_num),float )
        #destin_anim = np.zeros_like( pollution, float )
        destin_anim_pos = np.zeros((tMax,2))
        agents_history = np.zeros((tMax, N), dtype=[('x',float), ('y',float), ('health',int)])


        
    #disease_timeline = np.zeros( tMax ,dtype="int" )
    
    t = 0
    init(agents, positions, destinations, distances, force, N, N_ill, Lx, Ly, centralized_infectious, tile_x_size, tile_y_size, \
         dt, predestination_x, predestination_y, arrival_num, t, algorithm, dt_ratio)
    
    if flow_rate>=1:
        for t in range(tMax):
            active_walk(agents, positions, destinations, distances, force, N, Lx, Ly, tile_x_size, tile_y_size, dt\
               , predestination_x, predestination_y, arrival_num, t, algorithm, dt_ratio)
            
            #walk(agents, positions, N, stepSize, Lx, Ly, tile_x_size, tile_y_size)
            #update_tile(agents, positions, tile_x_size, tile_y_size)
            if shuffled_pollution_activate:
                shuffled_pollute(agents, pollution, fake_pollution, shuffled_x,\
                                 shuffled_y, pollution_rate, tile_infection_rate)
            else:
                pollute(agents, pollution, pollution_rate, tile_infection_rate)
            if t%flow_rate == 0:
                flow(agents, N_ill/N)
            
            if (t % dt_ratio) == 0:
                disease_timeline[t]['from_per'], disease_timeline[t]['from_env'] = \
                get_infected(agents, pollution, distances, state_after_infection, infection_rate)
            
            
            if animatable_output:                
                pollution_history[t] = pollution
                
                destin_anim_pos[t] = get_destin_anim_pos( destinations, tile_x_size, tile_y_size)
                
                agents_history[t]['x'] = positions[:, 0]
                agents_history[t]['y'] = positions[:, 1]
                agents_history[t]['health'] = agents['health']
            

                
            
    else:
        for t in range(tMax):
            
            active_walk(agents, positions, destinations, distances, force, N, Lx, Ly, tile_x_size, tile_y_size, dt\
                , predestination_x, predestination_y, arrival_num, t, algorithm, dt_ratio)

#temp shuffled pollution deativated for now
#            if shuffled_pollution_activate:
#                shuffled_pollute(agents, pollution, fake_pollution, shuffled_x\
#                                 , shuffled_y, pollution_rate, tile_infection_rate)
#            else:
#                pollute(agents, pollution, pollution_rate, tile_infection_rate)
                
            if (t % dt_ratio) == 0:
                disease_timeline[t]['from_per'], disease_timeline[t]['from_env'] \
                = get_infected(agents, pollution, distances, state_after_infection, infection_rate)
                
                pollute(agents, pollution, pollution_rate, tile_infection_rate)
            
            if opening_duration: #if flash_forward is happening
                if (t % opening_duration == 0):
                    flash_forward(agents, positions, destinations, distances, N, pollution, Lx, Ly, tile_x_size, tile_y_size)
            #print(t, pollution.sum())


            
            if animatable_output:
                #get_destin_anim(destinations, destin_anim, tile_infection_rate)
                #pollution_history[t] = pollution + destin_anim
                pollution_history[t] = pollution
                
                destin_anim_pos[t] = get_destin_anim_pos( destinations, tile_x_size, tile_y_size)
                
                agents_history[t]['x'] = positions[:, 0]
                agents_history[t]['y'] = positions[:, 1]
                agents_history[t]['health'] = agents['health']

                
    if animatable_output:
        path = pathlib.Path('Results')
        path.mkdir(parents=True, exist_ok=True)
        
        dt_str = algorithm + '-' + str(dt)
        np.save('Results/pollution_history', pollution_history)
        np.save('Results/destination_history', destin_anim_pos)
        np.save('Results/agents_history', agents_history)
        #if shuffled_pollution_activate:
         #we can keep a record of the fake polluted tiles.   

    return disease_timeline
