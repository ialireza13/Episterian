import numpy as np

def simulate(args):
    shuffled_pollution_activate = False
    animatable_output = False
    centralized_infectious = False
    state_after_infection = 2 #1 for E, 2 for I
    
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

    def walk(agents):
        step = np.random.random(N) * 2.*np.pi
        dx = stepSize * np.cos(step)
        dy = stepSize * np.sin(step)
        agents['x'] += dx
        agents['y'] += dy
        passedRight = ((agents['x']+dx) > Lx)
        passedLeft = ((agents['x']+dx) < 0)
        passedTop = ((agents['y']+dy) > Ly)
        passedBottom = ((agents['y']+dy) < 0)
        agents['x'][passedRight] = 2.*Lx - dx[passedRight] - agents['x'][passedRight]
        agents['x'][passedLeft] = -dx[passedLeft] - agents['x'][passedLeft]
        agents['y'][passedTop] = 2.*Ly - dy[passedTop] - agents['y'][passedTop]
        agents['y'][passedBottom] = -dy[passedBottom] - agents['y'][passedBottom]
        return agents


    def update_tile(agents):
        agents['tile_x'] = agents['x'] / tile_x_size
        agents['tile_y'] = agents['y'] / tile_y_size
        return agents

    def pollute(agents, pollution):
        
        polluted_x = agents['tile_x'][agents['health'] == 2]
        polluted_y = agents['tile_y'][agents['health'] == 2]
        rnd_list = np.random.random(len(polluted_x))
        pollution[ polluted_x, polluted_y ] = (rnd_list < pollution_rate) * tile_infection_rate + (rnd_list >= pollution_rate) * pollution[ polluted_x, polluted_y ]

        return pollution
    
    def shuffled_pollute(agents, pollution):
        
        polluted_x = agents['tile_x'][agents['health'] == 2]
        polluted_y = agents['tile_y'][agents['health'] == 2]
        rnd_list = np.random.random(len(polluted_x))
        fake_pollution[ polluted_x, polluted_y ] = (rnd_list < pollution_rate) * tile_infection_rate + (rnd_list >= pollution_rate) * fake_pollution[ polluted_x, polluted_y ]
        fake_pollution_num = np.sum(fake_pollution != 0)
        #print(fake_pollution_num)
        
        pollution[ shuffled_x[ :fake_pollution_num ], shuffled_y[ :fake_pollution_num ] ] = tile_infection_rate

        return pollution


    def get_infected(agents, pollution):
        susceptibles = agents['health'] == 0
        susceptibles_num = np.sum(susceptibles)
        
        #from environment infection
        from_env_inf = np.random.random( susceptibles_num ) < pollution[ agents['tile_x'][susceptibles], agents['tile_y'][susceptibles] ]
        agents['health'][susceptibles] = from_env_inf * state_after_infection
        from_env_num = np.sum(from_env_inf)

        infectors = agents[ agents['health'] == 2 ]

        tiles = [(infector['tile_x'], infector['tile_y']) for infector in infectors]
        on_tile = [all([(agent['tile_x'], agent['tile_y']) in tiles, agent['health']==0]) for agent in agents]

        on_tile_num = np.sum(on_tile)

        from_per_inf = np.random.random(on_tile_num) < infection_rate
        agents['health'][on_tile] = from_per_inf * state_after_infection
        from_per_num = np.sum(from_per_inf)

        return from_per_num, from_env_num
    
    def flow(agents, init_infection_prob):
        leaver = np.random.randint(len( agents ))
#         agents[leaver]['health'] = not(np.random.random() < init_infection_prob)
#         agents[leaver]['health'] *= 2
        agents[leaver]['health'] = np.random.choice( [0,2], p=[1-init_infection_prob, init_infection_prob] )
        agents[leaver]['x'] = np.random.random() * Lx
        agents[leaver]['y'] = np.random.random() * Ly
    
    def init(agents, centralized_infectious):
        
        agents['x'] = np.random.random(N) * Lx
        agents['y'] = np.random.random(N) * Ly

        if not centralized_infectious:
            infection_seed = np.random.randint(N, size=N_ill)

            agents['health'][infection_seed] = 2
        else: #centralized infectious seed
            agents['x'][0], agents['y'][0] = Lx/2, Ly/2
            agents['health'][0] = 2
            
        agents = update_tile(agents)            
        
    #disease_timeline = np.zeros( tMax ,dtype="int" )
    
    init(agents, centralized_infectious)

    if flow_rate>=1:
        for t in range(tMax):
            walk(agents)
            update_tile(agents)
            if shuffled_pollution_activate:
                shuffled_pollute(agents, pollution)
            else:
                pollute(agents, pollution)
            if t%flow_rate == 0:
                flow(agents, N_ill/N)
            disease_timeline[t]['from_per'], disease_timeline[t]['from_env'] = get_infected(agents, pollution)
            
            
            if animatable_output:
                pollution_history[t] = pollution
                agents_history[t] = agents

                
            
    else:
        for t in range(tMax):
            walk(agents)
            update_tile(agents)
            if shuffled_pollution_activate:
                shuffled_pollute(agents, pollution)
            else:
                pollute(agents, pollution)
                
            disease_timeline[t]['from_per'], disease_timeline[t]['from_env'] = get_infected(agents, pollution)
            
            if animatable_output:
                pollution_history[t] = pollution
                agents_history[t] = agents

    if animatable_output:
        np.save('Results/pollution_history', pollution_history)
        np.save('Results/agents_history', agents_history)
        #if shuffled_pollution_activate:
         #we can keep a record of the fake polluted tiles.   

    return disease_timeline
