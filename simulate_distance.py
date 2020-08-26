import numpy as np
from scipy.spatial.distance import pdist, squareform
from functions import *

def simulate(args):
    shuffled_pollution_activate = False
    animatable_output = False
    centralized_infectious = False
    distance_output = True
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

        

    else:
        print("Number of arguments don't match for simulate.")
    
    # set random seed for each process
    np.random.seed(random_seed)
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

    
    if shuffled_pollution_activate:
        shuffled_x = np.arange(tile_x_num)
        shuffled_y = np.arange(tile_y_num)
        np.random.shuffle(shuffled_x)
        np.random.shuffle(shuffled_y)
        
        fake_pollution = np.zeros( (tile_x_num, tile_y_num),float )
    
    if animatable_output:
        pollution_history = np.zeros( (tMax, tile_x_num, tile_y_num),float )
        destin_anim = np.zeros_like( pollution, float )
        #agents_history = np.zeros((tMax, N), dtype=[('x', 'float'), ('y', 'float'), ('tile_x',int), ('tile_y',int), ('health',int)] )
        #agents_history = np.zeros((tMax, N), dtype=[('x', 'float'), ('y', 'float'), ('tile_x',int), ('tile_y',int), ('health',int)])
        #agents_history = np.zeros((tMax, N), dtype=[('tile_x',int), ('tile_y',int), ('health',int)])
        agents_history = np.zeros((tMax, N), dtype=[('x',float), ('y',float), ('health',int)])
        
    if distance_output:
        distance_history = np.zeros((tMax, N))


        
    #disease_timeline = np.zeros( tMax ,dtype="int" )
    
    init(agents, positions, destinations, distances, N, N_ill, Lx, Ly, centralized_infectious, tile_x_size, tile_y_size)
    
    if flow_rate>=1:
        for t in range(tMax):
            active_walk(agents, positions, destinations, distances, N, Lx, Ly, tile_x_size, tile_y_size)
            #walk(agents, positions, N, stepSize, Lx, Ly, tile_x_size, tile_y_size)
            #update_tile(agents, positions, tile_x_size, tile_y_size)
            if shuffled_pollution_activate:
                shuffled_pollute(agents, pollution, fake_pollution, shuffled_x,\
                                 shuffled_y, pollution_rate, tile_infection_rate)
            else:
                pollute(agents, pollution, pollution_rate, tile_infection_rate)
            if t%flow_rate == 0:
                flow(agents, N_ill/N)
            disease_timeline[t]['from_per'], disease_timeline[t]['from_env'] = get_infected(agents, pollution, distances, state_after_infection, infection_rate)
            
            
            if animatable_output:
                get_destin_anim(destinations, destin_anim, tile_infection_rate)
                pollution_history[t] = pollution + destin_anim
                agents_history[t]['x'] = positions[:, 0]
                agents_history[t]['y'] = positions[:, 1]
                agents_history[t]['health'] = agents['health']
            

                
            
    else:
        for t in range(tMax):
            
            active_walk(agents, positions, destinations, distances, N, Lx, Ly, tile_x_size, tile_y_size)
            relax_agents(agents, positions, destinations, distances, N, Lx, Ly, tile_x_size, tile_y_size, number_of_steps = 40)
            min_dists = get_neighbor_dists(distances)
            distance_history[t][:] = min_dists
            #print(t)
            #if (t % 100 == 0 and t != 0):
#                print(t)
                

            #walk(agents, positions, N, stepSize, Lx, Ly, tile_x_size, tile_y_size)
            #update_tile(agents, positions, tile_x_size, tile_y_size)
            #if shuffled_pollution_activate:
#                shuffled_pollute(agents, pollution, fake_pollution, shuffled_x\
#                                 , shuffled_y, pollution_rate, tile_infection_rate)
#            else:
#                pollute(agents, pollution, pollution_rate, tile_infection_rate)
                
            disease_timeline[t]['from_per'], disease_timeline[t]['from_env'] \
            = get_infected(agents, pollution, distances, state_after_infection, infection_rate)
            if opening_duration: #if flash_forward is happening
                if (t % opening_duration == 0):
                    flash_forward(agents, positions, destinations, distances, N, pollution, Lx, Ly, tile_x_size, tile_y_size)
            #print(t, pollution.sum())


            
            if animatable_output:
                get_destin_anim(destinations, destin_anim, tile_infection_rate)
                pollution_history[t] = pollution + destin_anim
                agents_history[t]['x'] = positions[:, 0]
                agents_history[t]['y'] = positions[:, 1]
                agents_history[t]['health'] = agents['health']
                
    if animatable_output:
        np.save('Results/pollution_history', pollution_history)
        np.save('Results/agents_history', agents_history)
        #if shuffled_pollution_activate:
         #we can keep a record of the fake polluted tiles.   

    return disease_timeline, distance_history
