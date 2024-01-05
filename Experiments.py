import random
from Simulator import Simulator
import numpy as np

def trajectories():
    setup = {
        'N': 30,
        'TILE_SIZE': 25,
        'object': True,
        'symbol': 'O',
        'target_shape': True,
        'show_tetromines': False,
        'show_tetromino_contour': True,

        'resolution': 3,
        'n_random_targets': 0,
        'shuffle_targets': False,

        'delay': False,
        'visualize': False,
        'save_data': True,
        'data_tiles': False,
        'data_objet_target': True,
        'file_name': 'file_name',
        'dead_tiles': 0,
        'save_animation': False,

    }

    symbols = ["O", "L","J", "S", "Z"]

    for i, symbol in enumerate(symbols):
        setup['symbol'] = symbol
        setup['N'] = 20
        setup['resolution'] = 2
        setup['file_name'] = f'Trajectories/symbol_{symbol}'
        seed = random.randint(0, 1000000)
        random.seed(seed)

        simulator = Simulator(setup)
        simulator.run_simulation()

def diffusion_mechanism():
    setup = {
        'N' : 25,
        'TILE_SIZE' : 20,
        'object': True,
        'symbol': 'T',
        'target_shape': True,
        'show_tetromines' : False,
        'show_tetromino_contour' : True,
        
        'resolution': 2,
        'n_random_targets' : 0,
        'shuffle_targets': False,
        
        'delay': False,
        'visualize': True,
        'save_data': True,
        'data_tiles': False,
        'data_objet_target': True,
        'file_name': 'defaultname',

        'dead_tiles': 0,
        'save_animation': False,
        }

    N = 10
    setup['N'] = N
    setup['file_name'] = f'Diffusion/data_diffusion'
    simulator = Simulator(setup)
    seed = random.randint(0, 1000000)

    # target_tiles = np.array([(7, 3), (1, 2), (3, 8)], dtype=np.int32)*N//10
    target_tiles = np.array([(N//2, N//2)], dtype=np.int32)
    for tile in target_tiles:
        t = simulator.board.get_tile(tile[0], tile[1])
        t.set_as_target()
        t.target_angle = 180
    random.seed(seed)
    print('seed: ', seed)

    simulator.run_simulation()

def resolution_influence():
    setup = {
        'N' : 25,
        'TILE_SIZE' : 20,
        'object': True,
        'symbol': 'T',
        'target_shape': True,
        'show_tetromines' : False,
        'show_tetromino_contour' : True,
        
        'resolution': 2,

        'n_random_targets' : 0,
        'shuffle_targets': False,
        
        'delay': False,
        'visualize': False,

        'save_data': True,
        'data_tiles': False,
        'data_objet_target': True,
        'file_name': 'defaultname',

        'dead_tiles': 0,
        'save_animation': False,
    }

    symbols = ["I", "O", "T", "J", "L", "S", "Z"]
    # This is the setup used in the paper:
    runs_per_m = 20
    resolutions = [0.5, 0.75, 1, 2, 3, 4, 5]
    
    # This is a tiny experiment:    
    #runs_per_m = 5
    #resolutions = [3,4,5]

    for resolution in resolutions:
        print(f'Resolution: {resolution}')
        for run in range(runs_per_m):
            setup['symbol'] = random.choice(symbols)
            setup['resolution'] = resolution
            setup['file_name'] = f'Resolution/res_{resolution}_run_{run}'
            seed = random.randint(0, 1000000)
            random.seed(seed)

            simulator = Simulator(setup)
            simulator.run_simulation()

def fault_tolerance():

    setup = {
    'N' : 25,
    'TILE_SIZE' : 20,
    'object': True,
    'symbol': 'T',
    'target_shape': True,
    'show_tetromines' : False,
    'show_tetromino_contour' : True,
    
    'resolution': 2,
    'n_random_targets' : 0,
    'shuffle_targets': False,
    
    'delay': False,
    'visualize': False,
    'save_data': True,
    'data_tiles': False,
    'data_objet_target': True,
    'file_name': 'defaultname',

    'dead_tiles': 0,
    'save_animation': False,
    }

    symbols = ["I", "O", "T", "J", "L", "S", "Z"]  
    
    # This is the setup used in the paper:
    
    runs_per_percent = 20
    percentages = [i/10 for i in range(0, 10)]
    resolutions = [2, 3, 4, 5]
    
    # This is a tiny experiment:
    #runs_per_percent = 10
    #percentages = [i/10 for i in range(0, 10)]
    #resolutions = [0.5, 0.75, 1, 2, 3, 4, 5]

    for res in resolutions:
        for percent in percentages:
            print(f'Dead percent: {percent}')
            setup['dead_tiles'] = percent
            for run in range(runs_per_percent):
                runid = run
                setup['resolution'] = res
                setup['symbol'] = random.choice(symbols)
                setup['file_name'] = f'Faulty/res_{res}_fau_{percent}_run_{runid}'
                seed = random.randint(0, 1000000)
                random.seed(seed)

                simulator = Simulator(setup)
                simulator.run_simulation()

if __name__ == "__main__":
    #diffusion_mechanism()
    #trajectories()
    #resolution_influence()
    fault_tolerance()
    pass