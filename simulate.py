import numpy as np
from func import *
def simulate(args):
    shuffled_pollution_activate = False
    animatable_output = False
    centralized_infectious = False
    state_after_infection = 2 #1 for E, 2 for I
    opening_duration = 0 # 0 indicates no flash_forward
    if len(args) == 10:
        N, N_ill, Lx, Ly, stepSize, infection_rate, pollution_rate\
        , tile_infection_rate, flow_rate, tMax = args
        
    elif len(args) == 11:

        N, N_ill, Lx, Ly, stepSize, infection_rate, pollution_rate\
        , tile_infection_rate, flow_rate, tMax, shuffled_pollution_activate = args
        
    elif len(args) == 12:

        N, N_ill, Lx, Ly, stepSize, infection_rate, pollution_rate\
        , tile_infection_rate, flow_rate, tMax,\
        shuffled_pollution_activate, animatable_output = args
        
    elif len(args) == 13:

        N, N_ill, Lx, Ly, stepSize, infection_rate, pollution_rate\
        , tile_infection_rate, flow_rate, tMax,\
        shuffled_pollution_activate, animatable_output, centralized_infectious = args

    elif len(args) == 14:

        N, N_ill, Lx, Ly, stepSize, infection_rate, pollution_rate\
        , tile_infection_rate, flow_rate, tMax,\
        shuffled_pollution_activate, animatable_output,\
        centralized_infectious, state_after_infection = args
        
    elif len(args) == 15:

        N, N_ill, Lx, Ly, stepSize, infection_rate, pollution_rate\
        , tile_infection_rate, flow_rate, tMax,\
        shuffled_pollution_activate, animatable_output,\
        centralized_infectious, state_after_infection,\
        opening_duration = args



    else:
        print("Number of arguments don't match for simulate.")
    
    tile_x_num = Lx-1
    tile_y_num = Ly-1
    
    tile_x_size = Lx / tile_x_num
    tile_y_size = Ly / tile_y_num
    
    
    disease_timeline = np.zeros( ( tMax ), dtype=[ ('from_per',int), ('from_env',int)] )

    agents = np.zeros((N), dtype=[('x', 'float'), ('y', 'float'), ('tile_x',int), ('tile_y',int), ('health',int)] )
    pollution = np.zeros( (tile_x_num, tile_y_num),float )

    
    if shuffled_pollution_activate:
        shuffled_x = np.arange(tile_x_num)
        shuffled_y = np.arange(tile_y_num)
        np.random.shuffle(shuffled_x)
        np.random.shuffle(shuffled_y)
        
        fake_pollution = np.zeros( (tile_x_num, tile_y_num),float )
    
    if animatable_output:
        pollution_history = np.zeros( (tMax, tile_x_num, tile_y_num),float )
        agents_history = np.zeros((tMax, N), dtype=[('x', 'float'), ('y', 'float'), ('tile_x',int), ('tile_y',int), ('health',int)] )

        
    #disease_timeline = np.zeros( tMax ,dtype="int" )
    
    init(agents, N, N_ill, Lx, Ly, centralized_infectious, tile_x_size, tile_y_size)

    if flow_rate>=1:
        for t in range(tMax):
            walk(agents, N, stepSize, Lx, Ly)
            update_tile(agents, tile_x_size, tile_y_size)
            if shuffled_pollution_activate:
                shuffled_pollute(agents, pollution, pollution_rate, tile_infection_rate)
            else:
                pollute(agents, pollution, pollution_rate, tile_infection_rate)
            if t%flow_rate == 0:
                flow(agents, N_ill/N)
            disease_timeline[t]['from_per'], disease_timeline[t]['from_env'] = get_infected(agents, pollution, state_after_infection, infection_rate)
            
            
            if animatable_output:
                pollution_history[t] = pollution
                agents_history[t] = agents
            

                
            
    else:
        for t in range(tMax):
            walk(agents, N, stepSize, Lx, Ly)
            update_tile(agents, tile_x_size, tile_y_size)
            if shuffled_pollution_activate:
                shuffled_pollute(agents, pollution, pollution_rate, tile_infection_rate)
            else:
                pollute(agents, pollution, pollution_rate, tile_infection_rate)
                
            disease_timeline[t]['from_per'], disease_timeline[t]['from_env'] = get_infected(agents, pollution, state_after_infection, infection_rate)
            if opening_duration: #if flash_forward is happening
                if (t % opening_duration == 0):
                    flash_forward(agents, N, pollution, Lx, Ly, tile_x_size, tile_y_size)
            print(t, pollution.sum())


            
            if animatable_output:
                pollution_history[t] = pollution
                agents_history[t] = agents
            #print(pollution)
    if animatable_output:
        np.save('Results/pollution_history', pollution_history)
        np.save('Results/agents_history', agents_history)
        #if shuffled_pollution_activate:
         #we can keep a record of the fake polluted tiles.   

    return disease_timeline
