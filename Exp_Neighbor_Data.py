from loguru import logger

import _config
import _folders
from NCA import NCA_CenterFinder
from StateStructure import StateStructure
from Training import Trainer_CenterFinder
from Visuals import simulate_center_finder, visualize_from_file


def run_block(state_structure:StateStructure, files_name:str, seed = None, visualize=False):
    if seed is not None:
        from Util import set_seed
        set_seed(seed)

    model = NCA_CenterFinder(state_structure=state_structure)
    trained_model = Trainer_CenterFinder().train_center_finder(model=model, state_structure=state_structure, experiment_name=files_name)
    _folders.save_model(trained_model=trained_model, experiment_name=files_name)
    logger.success(f"Model {files_name} trained and saved")

    if visualize:
        model = _folders.load_model(files_name)
        for board_shape in [_config.BOARD_SHAPE[0]]:
            simulation_path = f"{_folders.SIMULATIONS_PATH}/{files_name}.npy"            
            
            simulate_center_finder(model = model, 
                                   state_structure = state_structure, 
                                   board_shape = [board_shape,board_shape], 
                                   batch_size=2, 
                                   n_movements=20, 
                                   output_file_name=simulation_path, 
                                   num_steps=15)
            
            animation_path = f"{_folders.VISUALIZATIONS_PATH}/{files_name}.gif"
            visualize_from_file(simulation_path=simulation_path, animation_path=animation_path, state_structure=state_structure)

if __name__ == '__main__':
    import os

    experiment_name=f'Neighbor_Hidden_Size'
    _folders.set_experiment_folders(experiment_name)
    
    experiment_files = [f for f in os.listdir(_folders.RESULTS_PATH) if f.endswith('.json') and 'Training' in f]
    print(f"Found {len(experiment_files)} experiments")
        
    all_hs = range(11)
    all_runs = range(0,6)
    all_neighborhoods = ['Chebyshev', 'Manhattan']

    args = []
    
    for hs in all_hs:
        for run in all_runs:
            for neighborhood in all_neighborhoods:
                if f'Training_{neighborhood}_HS_{hs}_Run_{run}.json' in experiment_files:
                    #print(f"Skipping {neighborhood} HS {hs} Run {run}")
                    continue
                else:
                    args.append((neighborhood, hs, run))
                    print(f"Missing training {neighborhood} HS {hs} Run {run}")

    print(f"Training {len(args)} models")
    
    for i, arg in enumerate(args):
        logger.success(f"Training {i+1} out of {len(args)}")
        neighborhood, hs, run = arg
        logger.success(f"Training {neighborhood} HS {hs} Run {run}")
        _config.set_parameters({
            'BOARD_SHAPE' :     [8,8],
            'TRAINING_STEPS' :  1000,
            'NEIGHBORHOOD' :    neighborhood,
        })

        state_structure = StateStructure(
                            estimation_dim  = 2,    
                            constant_dim    = 2,   
                            sensor_dim      = 1,   
                            hidden_dim      = hs)
        files_name = f'{_config.NEIGHBORHOOD}_HS_{hs}_Run_{run}'
        run_block(state_structure=state_structure, files_name= files_name, seed=None, visualize=False)
