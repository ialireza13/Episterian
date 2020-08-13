import numpy as np

def walk(agents, N, stepSize, Lx, Ly):
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


def update_tile(agents, tile_x_size, tile_y_size):
    agents['tile_x'] = agents['x'] / tile_x_size
    agents['tile_y'] = agents['y'] / tile_y_size
    return agents

def pollute(agents, pollution, pollution_rate, tile_infection_rate):
    
    polluted_x = agents['tile_x'][agents['health'] == 2]
    polluted_y = agents['tile_y'][agents['health'] == 2]
    rnd_list = np.random.random(len(polluted_x))
    pollution[ polluted_x, polluted_y ] = (rnd_list < pollution_rate) * tile_infection_rate + (rnd_list >= pollution_rate) * pollution[ polluted_x, polluted_y ]

    return pollution

def shuffled_pollute(agents, pollution, pollution_rate, tile_infection_rate):
    
    polluted_x = agents['tile_x'][agents['health'] == 2]
    polluted_y = agents['tile_y'][agents['health'] == 2]
    rnd_list = np.random.random(len(polluted_x))
    fake_pollution[ polluted_x, polluted_y ] = (rnd_list < pollution_rate) * tile_infection_rate + (rnd_list >= pollution_rate) * fake_pollution[ polluted_x, polluted_y ]
    fake_pollution_num = np.sum(fake_pollution != 0)
    #print(fake_pollution_num)
    
    pollution[ shuffled_x[ :fake_pollution_num ], shuffled_y[ :fake_pollution_num ] ] = tile_infection_rate

    return pollution


def get_infected(agents, pollution, state_after_infection, infection_rate):
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

def flash_forward(agents, N, pollution, Lx, Ly, tile_x_size, tile_y_size):

    (agents['health'][ agents['health'] == 1 ]) = 2
    agents['x'] = np.random.random(N) * Lx
    agents['y'] = np.random.random(N) * Ly
    
    agents = update_tile(agents, tile_x_size, tile_y_size)
    
    pollution[:] = 0



def init(agents, N, N_ill, Lx, Ly, centralized_infectious, tile_x_size, tile_y_size):
    
    agents['x'] = np.random.random(N) * Lx
    agents['y'] = np.random.random(N) * Ly

    if not centralized_infectious:
        infection_seed = np.random.randint(N, size=N_ill)

        agents['health'][infection_seed] = 2
    else: #centralized infectious seed
        agents['x'][0], agents['y'][0] = Lx/2, Ly/2
        agents['health'][0] = 2
        
    agents = update_tile(agents, tile_x_size, tile_y_size)        


