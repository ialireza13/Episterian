import numpy as np
from scipy.spatial.distance import pdist, squareform

def walk(agents, positions, N, stepSize, Lx, Ly, tile_x_size, tile_y_size):
    step = np.random.random(N) * 2.*np.pi
    dx = stepSize * np.cos(step)
    dy = stepSize * np.sin(step)
    positions[:, 0] += dx
    positions[:, 1] += dy
    passedRight = ((positions[:, 0]+dx) > Lx)
    passedLeft = ((positions[:, 0]+dx) < 0)
    passedTop = ((positions[:, 1]+dy) > Ly)
    passedBottom = ((positions[:, 1]+dy) < 0)
    positions[:, 0][passedRight] = 2.*Lx - dx[passedRight] - positions[:, 0][passedRight]
    positions[:, 0][passedLeft] = -dx[passedLeft] - positions[:, 0][passedLeft]
    positions[:, 1][passedTop] = 2.*Ly - dy[passedTop] - positions[:, 1][passedTop]
    positions[:, 1][passedBottom] = -dy[passedBottom] - positions[:, 1][passedBottom]
    update_tile(agents, positions, tile_x_size, tile_y_size)

    return agents

def active_walk(agents, positions, destinations, distances, force, N, Lx, Ly, tile_x_size, tile_y_size, dt\
               , predestination_x, predestination_y, arrival_num, t, algorithm, dt_ratio):
        
    #print( dt )
    x = positions[:, 0]
    y = positions[:, 1]
    vx = positions[:, 2]
    vy = positions[:, 3]


    if algorithm == 'euler':
        update_force( agents, positions, distances, force, dt, Lx, Ly, destinations\
                     ,tile_x_size, tile_y_size, tau_inv = 2)

        vx[:] += force[:, 0] * dt
        vy[:] += force[:, 1] * dt

        x[:] += vx * dt
        y[:] += vy * dt

    elif algorithm == 'velocity_verlet':
        x[:] += (vx * dt + (0.5 * force[:, 0] * (dt ** 2)) )
        y[:] += (vy * dt + (0.5 * force[:, 1] * (dt ** 2)) )

        vx[:] += force[:, 0] * 0.5 * dt
        vy[:] += force[:, 1] * 0.5 * dt

        update_force( agents, positions, distances, force, dt, Lx, Ly, destinations\
             ,tile_x_size, tile_y_size, tau_inv = 2)

        vx[:] += force[:, 0] * 0.5 * dt
        vy[:] += force[:, 1] * 0.5 * dt

    elif algorithm == 'runge_kutta':
        m1x = dt * vx
        m1y = dt * vy

        update_force( agents, positions, distances, force, dt, Lx, Ly, destinations\
             ,tile_x_size, tile_y_size, tau_inv = 2)
        k1x = dt * force[:, 0]
        k1y = dt * force[:, 1]

        m2x = dt*(vx + 0.5*k1x)
        m2y = dt*(vy + 0.5*k1y)

        t_positions = np.copy(positions)
        t_positions[:, 0] += 0.5 * m1x
        t_positions[:, 1] += 0.5 * m1y

        update_force( agents, t_positions, distances, force, dt, Lx, Ly, destinations\
             ,tile_x_size, tile_y_size, tau_inv = 2)

        k2x = dt * force[:, 0]
        k2y = dt * force[:, 1]


        m3x = dt*(vx + 0.5*k2x)
        m3y = dt*(vy + 0.5*k2y)

        t_positions[:] = positions
        t_positions[:, 0] += 0.5 * m2x
        t_positions[:, 1] += 0.5 * m2y

        update_force( agents, t_positions, distances, force, dt, Lx, Ly, destinations\
             ,tile_x_size, tile_y_size, tau_inv = 2)

        k3x = dt * force[:, 0]
        k3y = dt * force[:, 1]


        m4x = dt*(vx + k3x)
        m4y = dt*(vy + k3y)

        t_positions[:] = positions
        t_positions[:, 0] += m3x
        t_positions[:, 1] += m3y

        update_force( agents, t_positions, distances, force, dt, Lx, Ly, destinations\
             ,tile_x_size, tile_y_size, tau_inv = 2)


        k4x = dt * force[:, 0]
        k4y = dt * force[:, 1]

        x[:] += (m1x + 2*m2x + 2*m3x + m4x)/6
        y[:] += (m1y + 2*m2y + 2*m3y + m4y)/6

        vx[:] += (k1x + 2*k2x + 2*k3x + k4x)/6
        vy[:] += (k1y + 2*k2y + 2*k3y + k4y)/6

    else:
        print('algorithm not correctly specified')
    
    positions[:, 0] %= Lx
    positions[:, 1] %= Ly
    
    if (t % dt_ratio) == 0:
        #print(t)
        update_tile(agents, positions, tile_x_size, tile_y_size)
        update_arrived_destinations( agents, destinations, Lx, Ly, predestination_x, predestination_y, arrival_num)
    limit_velocity( positions, vMax = 2 )

    return agents

def update_force( agents, positions, distances, force, dt, Lx, Ly, destinations\
                 ,tile_x_size, tile_y_size, tau_inv = 2):
    
    force[:] = 0
    agent_agent_interaction(agents, positions, distances, force, dt = dt)
    agent_wall_interaction(positions, Lx, Ly, force, dt = dt)
    agent_self_adjustment(positions, destinations, Lx, Ly, tile_x_size, tile_y_size, force\
                          , dt = dt, tau_inv = 2)

def agent_agent_interaction(agents, positions, distances, force, dt, cut_off = 3):
    distances[:] = squareform(pdist(positions[:, :2]))
    ind1, ind2 = np.where(distances < cut_off)

    unique = (ind1 < ind2)
    ind1 = ind1[unique]
    ind2 = ind2[unique]

    for i1, i2 in zip(ind1, ind2):
        #print(i1, i2)
        # location vector
        r1 = positions[i1, :2]
        r2 = positions[i2, :2]
        # velocity vector
        v1 = positions[i1, 2:4]
        v2 = positions[i2, 2:4]
        # relative location & velocity vectors
        #r_rel = r1 - r2

        distance = distances[i1, i2]
        r_norm = (r1 - r2) / distance
        sigma_1 = agents['sigma'][i1]
        sigma_2 = agents['sigma'][i2]

        force_mag_1 = 7.0 * np.exp( -distance / sigma_1 )
        force_mag_2 = 7.0 * np.exp( -distance / sigma_2 )

        
        force_1 = force_mag_1 * r_norm
        force_2 = force_mag_2 * r_norm

        #positions[i1, 2:4] = positions[i1, 2:4] + force_1 *dt
        #positions[i2, 2:4] = positions[i2, 2:4] - force_2 *dt
        
        force[i1] += force_1
        force[i2] -= force_2
        



def agent_wall_interaction(positions, Lx, Ly, force, dt, cut_off = 1, wall_sigma = 5):

    r_indices = [0, 0, 1, 1]
    impact_directions = [1, -1, 1, -1]
    border_positions = [0, Lx, 0, Ly]
    for r_index, impact_direction, border_position in zip(r_indices, impact_directions, border_positions):

        r = positions[:, r_index ]
        distance = np.abs( r - border_position )
        on_border = np.where( distance < cut_off)

        force_mag = wall_sigma * np.exp( -distance[on_border] / wall_sigma )

        force_vector = force_mag * impact_direction
        #print(id(force))
        force[ on_border, r_index ] += force_vector
        #print(id(force))
        
        
        #v_index = r_index + 2 #x, y, vx, vy
        #positions[on_border, v_index] = positions[on_border, v_index] + force_vector * dt


def agent_self_adjustment(positions, destinations, Lx, Ly, tile_x_size, tile_y_size, force, dt, prefer_speed = 1.3 ,tau_inv = 2):

    r = positions[:, :2]
    v = positions[:, 2:4]
    tile_coords = np.array([tile_x_size, tile_y_size])
    float_destinations = (destinations * tile_coords) + (tile_coords / 2)
    directions = float_destinations - r
    directions /= np.linalg.norm(directions, axis = 1)[:, None]
    
    prefer_v = directions * prefer_speed
    adj_force = ( prefer_v - v ) * tau_inv
    #positions[:, 2:4] = v + adj_force * dt
    #v[:] = v[:] + adj_force * dt
    force[:] += adj_force
    
def update_tile(agents, positions, tile_x_size, tile_y_size):
    agents['tile_x'] = positions[:, 0] / tile_x_size
    agents['tile_y'] = positions[:, 1] / tile_y_size
    return agents


        
def update_arrived_destinations( agents, destinations, Lx, Ly\
    , predestination_x, predestination_y, arrival_num):

    arrived = np.where( np.all([ destinations[:, 0] == agents['tile_x'], destinations[:, 1] == agents['tile_y'] ], 0) )[0]

    #destinations[arrived, 0] = np.random.randint( 1, Lx - 2, len(arrived) )
    #destinations[arrived, 1] = np.random.randint( 1, Ly - 2, len(arrived) )
    
    arrival_num[ arrived ] += 1
    
    
    destinations[arrived, 0] = predestination_x[ arrived, arrival_num[ arrived ] ]
    destinations[arrived, 1] = predestination_y[ arrived, arrival_num[ arrived ] ]


    #print( 'arrived: ', arrived, ' state: ', np.random.get_state()[2] )

def limit_velocity( positions, vMax = 2 ):
    v = positions[:, 2:4]
    #np.linalg.norm(v, axis = 1)[:, None]
    velocity_size = np.linalg.norm(v, axis = 1) 
    v_exceeded = np.where( velocity_size > vMax)[0]
    lowering_factor = velocity_size[ v_exceeded ] / vMax
    
    positions[v_exceeded, 2:4] /= lowering_factor[:, None]

        
        
def pollute(agents, pollution, pollution_rate, tile_infection_rate):
    
    polluted_x = agents['tile_x'][agents['health'] == 2]
    polluted_y = agents['tile_y'][agents['health'] == 2]
    rnd_list = np.random.random(len(polluted_x))
    pollution[ polluted_x, polluted_y ] = (rnd_list < pollution_rate) * tile_infection_rate + (rnd_list >= pollution_rate) * pollution[ polluted_x, polluted_y ]

    return pollution

def shuffled_pollute(agents, pollution, fake_pollution, shuffled_x, shuffled_y, pollution_rate, tile_infection_rate):
    
    polluted_x = agents['tile_x'][agents['health'] == 2]
    polluted_y = agents['tile_y'][agents['health'] == 2]
    rnd_list = np.random.random(len(polluted_x))
    fake_pollution[ polluted_x, polluted_y ] = (rnd_list < pollution_rate) * tile_infection_rate + (rnd_list >= pollution_rate) * fake_pollution[ polluted_x, polluted_y ]
    fake_pollution_num = np.sum(fake_pollution != 0)
    #print(fake_pollution_num)
    
    pollution[ shuffled_x[ :fake_pollution_num ], shuffled_y[ :fake_pollution_num ] ] = tile_infection_rate

    return pollution


def get_infected(agents, pollution, distances, state_after_infection, infection_rate, infection_cut_off = 1):
    susceptibles = (agents['health'] == 0)
    susceptibles_num = np.sum(susceptibles)
    
    ##from environment infection
    from_env_inf = np.random.random( susceptibles_num ) < \
    pollution[ agents['tile_x'][susceptibles], agents['tile_y'][susceptibles] ]
    
    agents['health'][susceptibles] = from_env_inf * state_after_infection
    from_env_num = np.sum(from_env_inf)

    ##from per infection

    infectors = np.where( agents['health'] == 2 )[0]
    
    from_per_num = 0
    for infector in infectors:
        close_enough = np.where(distances[infector] < infection_cut_off)[0] #close enough for infecton.
        for neighbor in close_enough:
            if agents['health'][neighbor] == 0: #if is susceptible
                if np.random.random() < infection_rate:
                    agents['health'][neighbor] = state_after_infection
                    #print(neighbor)
                    from_per_num += 1

    return from_per_num, from_env_num

def flow(agents, init_infection_prob):
    leaver = np.random.randint(len( agents ))
#         agents[leaver]['health'] = not(np.random.random() < init_infection_prob)
#         agents[leaver]['health'] *= 2
    agents[leaver]['health'] = np.random.choice( [0,2], p=[1-init_infection_prob, init_infection_prob] )
    agents[leaver]['x'] = np.random.random() * Lx
    agents[leaver]['y'] = np.random.random() * Ly

def flash_forward(agents, positions, destinations, distances, N, Lx, Ly, tile_x_size, tile_y_size, dt\
                  , predestination_x, predestination_y, arrival_num, t, algorithm, dt_ratio):

    init_movements(agents, positions, destinations, distances, N, Lx, Ly, tile_x_size, tile_y_size, dt\
                  , predestination_x, predestination_y, arrival_num, t, algorithm, dt_ratio)
    
    (agents['health'][ agents['health'] == 1 ]) = 2
    agents = update_tile(agents, positions, tile_x_size, tile_y_size)
    
    pollution[:] = 0


def init(agents, positions, destinations, distances, force, N, N_ill, Lx, Ly, centralized_infectious, tile_x_size, tile_y_size, dt\
        , predestination_x, predestination_y, arrival_num, t, algorithm, dt_ratio):
        
    init_movements(agents, positions, destinations, distances, force, N, Lx, Ly, tile_x_size, tile_y_size, dt\
                  , predestination_x, predestination_y, arrival_num, t, algorithm, dt_ratio)
    #print( np.random.get_state()[2] )

    
    if not centralized_infectious:
        
        infection_seed = np.random.randint(N, size=N_ill)
        #agents['health'][infection_seed] = 2
        agents['health'][0] = 2
        
        
        
    else: #centralized infectious seed
        positions[0, 0], agents[0, 1] = Lx/2, Ly/2
        agents['health'][0] = 2
        
    agents = update_tile(agents, positions, tile_x_size, tile_y_size)        


def init_movements(agents, positions, destinations, distances, force, N, Lx, Ly, tile_x_size, tile_y_size, dt\
                  , predestination_x, predestination_y, arrival_num, t, algorithm, dt_ratio):    
    #print( np.random.get_state()[2] )
    positions[:, 0] = np.random.random(N) * Lx
    positions[:, 1] = np.random.random(N) * Ly
    
    
    #destinations[:, 0] = np.random.randint(1, Lx - 2, N)
    #destinations[:, 1] = np.random.randint(1, Ly - 2, N)
    
    destinations[:, 0] = predestination_x[ :, 0 ]
    destinations[:, 1] = predestination_y[ :, 0 ]


    
    positions[:, 2] =  np.random.uniform(-1, 1, N)
    positions[:, 3] =  np.random.uniform(-1, 1, N)
    
    #print( 'after_position = ', np.random.get_state()[2] )

    
    #positions[0, :] = [ 15, 21, 0, 0]
    #positions[1, :] = [ 15, 20, 0, 0]
    
    relax_agents(agents, positions, destinations, distances, force, N, Lx, Ly, tile_x_size, tile_y_size, dt, predestination_x, predestination_y, arrival_num, t, algorithm, dt_ratio, number_of_steps = 60 * dt_ratio)

    
def relax_agents(agents, positions, destinations, distances, force, N, Lx, Ly, tile_x_size, tile_y_size, dt, predestination_x, predestination_y, arrival_num, t, algorithm, dt_ratio, number_of_steps):               
    for step in range(number_of_steps):
        active_walk(agents, positions, destinations, distances, force, N, Lx, Ly, tile_x_size, tile_y_size, dt\
               , predestination_x, predestination_y, arrival_num, step, algorithm, dt_ratio)

def get_destin_anim(destinations, destin_anim, tile_infection_rate):
    destin_anim[:] = 0
    
    #destin_anim[destinations[:, 0], destinations[:, 1]] = 0.05
    if tile_infection_rate:
        destin_anim[destinations[0, 0], destinations[0, 1]] = (tile_infection_rate / 2)
    else:
        destin_anim[destinations[0, 0], destinations[0, 1]] = 0.5

def get_destin_anim_pos( destinations, tile_x_size, tile_y_size):
    zeroth_destin = destinations[0]    
    x = zeroth_destin[0] * tile_x_size
    y = zeroth_destin[1] * tile_y_size
    return np.array([x, y])

        
def get_neighbor_dists(distances):
    min_dists = np.zeros( len(distances) )
    #print(distances.min())
    for i_row, row in enumerate(distances):
        min_dists[i_row] = row[(row > 0)].min()
    return min_dists

def get_velocities(positions):
    vx = positions[:, 2]
    vy = positions[:, 3]
    
    print (np.mean( np.sqrt( vx ** 2 + vy ** 2 ) ))
    return np.sqrt( vx ** 2 + vy ** 2 )