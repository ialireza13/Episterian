import numpy as np
import multiprocessing as mp

def returnPollution(args):
    N_ill, Lx, Ly, stepSize, pollution_rate, tMax = args
    
    tile_x_num = Lx-1
    tile_y_num = Ly-1
    
    tile_x_size = Lx / tile_x_num
    tile_y_size = Ly / tile_y_num

    def walk(agents):
        step = np.random.random(N_ill) * 2.*np.pi
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
        pollution[ polluted_x, polluted_y ] = (rnd_list < pollution_rate) + pollution[polluted_x, polluted_y]*(rnd_list >= pollution_rate)

        return pollution

    agents = np.zeros((N_ill), dtype=[('x', 'float'), ('y', 'float'), ('tile_x',int), ('tile_y',int), ('health',int)] )
    agents['x'] = np.random.random(N_ill) * Lx
    agents['y'] = np.random.random(N_ill) * Ly
    pollution = np.zeros( (tile_x_num, tile_y_num),float )
    ###initialize
    agents = update_tile(agents)
    agents['health'] = 2
    pollution = pollute(agents, pollution)
    poll_dens = np.zeros(tMax)
    for t in range(tMax):
        walk(agents)
        update_tile(agents)
        pollute(agents, pollution)
        poll_dens[t] = np.sum(pollution)/(Lx*Ly)
    
    return poll_dens

if __name__ ==  '__main__':

    jobs = np.zeros((1), dtype=[('N_ill', int), ('Lx', int), ('Ly',int), ('step size',float), ('pollution rate',float), ('time',int)] )

    Lx = 30
    Ly = 30
    stepSize = 0.5
    N_ill = 1

    realisations = 5
    tMax = 1000
    #----------------N_ill--Lx--Ly----step----poll rate--time  
    jobs[0] = tuple([N_ill, Lx, Ly, stepSize,      0.1,    tMax])

    works = [jobs[0] for i in range(realisations)]
    
    with mp.Pool(mp.cpu_count()) as pool:
        p_r = pool.map_async(returnPollution, works)
        results = p_r.get()
    
    np.save(str(jobs[0]), results)