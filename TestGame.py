import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import random
from Simulator import Simulator
import random

if __name__ == "__main__":
    setup = {
        'N': 20,
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
        'visualize': True,
        'save_data': False,
        'data_tiles': False,
        'data_objet_target': True,
        'file_name': 'file_name',
        'dead_tiles': 0,
        'save_animation': False,
        }

    setup['symbol'] = random.choice(["O", "L","J", "S", "Z"])

    simulator = Simulator(setup)
    simulator.run_simulation()