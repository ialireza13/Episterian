def simulate(args):
    N, Lx, Ly, stepSize, infection_rate, pollution_rate, decay_rate, tMax = args
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
        pollution *= decay_rate

        polluted_x = agents['tile_x'][agents['health'] == 2]
        polluted_y = agents['tile_y'][agents['health'] == 2]
        pollution[ polluted_x, polluted_y ] = np.random.random(len(polluted_x)) < pollution_rate

        return pollution

    def get_infetced(agents, pollution):
        susceptibles = agents['health'] == 0
        susceptibles_num = np.sum(susceptibles)

        agents['health'][susceptibles] = np.random.random( susceptibles_num ) < pollution[ agents['tile_x'][susceptibles], agents['tile_y'][susceptibles] ]

        tx = agents[agents['health']==2]['tile_x'][0]
        ty = agents[agents['health']==2]['tile_y'][0]
        on_tile = [all([agent['tile_x']==tx, agent['tile_y']==ty, agent['health']==0]) for agent in agents]
        on_tile_num = np.sum(on_tile)
        agents['health'][on_tile] = np.random.random(on_tile_num) < infection_rate

        return susceptibles_num
        
    disease_timeline = np.zeros( tMax ,dtype="int" )
    agents = np.zeros((N), dtype=[('x', 'float'), ('y', 'float'), ('tile_x',int), ('tile_y',int), ('health',int)] )
    agents['x'] = np.random.random(N) * Lx
    agents['y'] = np.random.random(N) * Ly
    pollution = np.zeros( (tile_x_num, tile_y_num),float )
    ###initialize
    infection_seed = np.random.randint(N)
    agents = update_tile(agents)
    agents['health'][infection_seed] = 2
    pollution = pollute(agents, pollution)
    for t in range(tMax):
        walk(agents)
        update_tile(agents)
        pollute(agents, pollution)
        disease_timeline[t] = N - get_infetced(agents, pollution)
    return disease_timeline