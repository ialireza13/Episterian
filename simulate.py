def simulate(args):
    N, N_ill, Lx, Ly, stepSize, infection_rate, pollution_rate, tile_infection_rate, flow_rate, tMax = args
    import numpy as np
    
    tile_x_num = Lx-1
    tile_y_num = Ly-1
    
    tile_x_size = Lx / tile_x_num
    tile_y_size = Ly / tile_y_num

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

    def get_infetced(agents, pollution):
        susceptibles = agents['health'] == 0
        susceptibles_num = np.sum(susceptibles)
        #from environment infection
        from_env_inf = np.random.random( susceptibles_num ) < pollution[ agents['tile_x'][susceptibles], agents['tile_y'][susceptibles] ]
        agents['health'][susceptibles] = from_env_inf
        from_env_num = np.sum(from_env_inf)

        infectors = agents[ agents['health'] == 2 ]
#         on_each_tile = np.zeros((len(infectors), N),'bool') #those who are neighbors with each infector.
#         for infector_ind, infector in enumerate(infectors):
#             tx = infector['tile_x']
#             ty = infector['tile_y']
#         #     on_tile.append([all([agent['tile_x']==tx, agent['tile_y']==ty, agent['health']==0]) for agent in agents])
#             on_each_tile[infector_ind] = [all([agent['tile_x']==tx, agent['tile_y']==ty, agent['health']==0]) for agent in agents]
#         on_tile = np.any(on_each_tile,0)

        tiles = [(infector['tile_x'], infector['tile_y']) for infector in infectors]
        on_tile = [all([(agent['tile_x'], agent['tile_y']) in tiles, agent['health']==0]) for agent in agents]

        on_tile_num = np.sum(on_tile)

        from_per_inf = np.random.random(on_tile_num) < infection_rate
        agents['health'][on_tile] = from_per_inf
        from_per_num = np.sum(from_per_inf)

        return from_per_num, from_env_num
    
    def flow(agents, init_infection_prob):
        leaver = np.random.randint(len( agents ))
#         agents[leaver]['health'] = not(np.random.random() < init_infection_prob)
#         agents[leaver]['health'] *= 2
        agents[leaver]['health'] = np.random.choice( [0,2], p=[1-init_infection_prob, init_infection_prob] )
        agents[leaver]['x'] = np.random.random() * Lx
        agents[leaver]['y'] = np.random.random() * Ly

        
    #disease_timeline = np.zeros( tMax ,dtype="int" )
    disease_timeline = np.zeros( ( tMax ), dtype=[ ('from_per',int), ('from_env',int)] )

    agents = np.zeros((N), dtype=[('x', 'float'), ('y', 'float'), ('tile_x',int), ('tile_y',int), ('health',int)] )
    agents['x'] = np.random.random(N) * Lx
    agents['y'] = np.random.random(N) * Ly
    pollution = np.zeros( (tile_x_num, tile_y_num),float )
    ###initialize
    infection_seed = np.random.randint(N, size=N_ill)
    agents = update_tile(agents)
    agents['health'][infection_seed] = 2
    pollution = pollute(agents, pollution)
    if flow_rate>=1:
        for t in range(tMax):
            walk(agents)
            update_tile(agents)
            pollute(agents, pollution)
            if t%flow_rate == 0:
                flow(agents, N_ill/N)
            disease_timeline[t]['from_per'], disease_timeline[t]['from_env'] = get_infetced(agents, pollution)
    else:
        for t in range(tMax):
            walk(agents)
            update_tile(agents)
            pollute(agents, pollution)
            disease_timeline[t]['from_per'], disease_timeline[t]['from_env'] = get_infetced(agents, pollution)
        
    return disease_timeline