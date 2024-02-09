import json
import os
import random
import numpy as np
import torch

import _config
import Environment.Shapes as Shapes
import _folders
import Util
from Environment.ContactBoard import ContactBoard
from Environment.Tetromino import Tetromino
from StateStructure import StateStructure


def _moving_contact_masks(shapes: list[np.array]) -> (torch.Tensor, dict):
    '''
    Returns a tensor of shape [NUM_MOVEMENTS, HEIGHT, WIDTH]
    This method also saves the tetrominos used to generate the masks.
    Their positions in each number.
    '''
    height,width = _config.BOARD_SHAPE
    contact_masks = torch.zeros(_config.NUM_MOVEMENTS, height, width)
    board = ContactBoard(shape=[height, width])
    min_size_board = min(height, width)    

    shape_index =   random.randint(0, len(shapes)-1)
    shape       =   shapes[shape_index]
    name_shape  =   Shapes.get_name(shape)

    max_side_lengths = Util._calculate_max_distance(shape)
    max_scaler = min_size_board / max_side_lengths
    scaler = np.random.uniform(1, max_scaler)
    scaler = 2

    tetro = Tetromino(constructor_vertices=shape, scaler=scaler)
    tetro.center = np.array([width / 2, height / 2])
    
    shapes_data = {
        'alive_percent':0,
        'name_shape': name_shape,
        'scaler': scaler,
        'movements': [],
    }

    for movement_index in range(_config.NUM_MOVEMENTS):
        while True:
            displacement = np.random.uniform(-1, 1, size=2)
            if board.has_point_inside(displacement + tetro.center):
                break
        tetro.translate(displacement)
        tetro.rotate(np.random.uniform(0, 20))
        contact_i = board.get_contact_mask(tetro)
        contact_masks[movement_index] = torch.from_numpy(contact_i)
        shapes_data['movements'].append({
            'angle': tetro.angle,
            'true_center': tetro.center.tolist()
        })
    
    
    return contact_masks, shapes_data


def _get_target_tensor(constant_states: torch.Tensor, contact_mask:torch.Tensor) -> (torch.Tensor, list):
    
    D, H, W = constant_states.shape
    target_tensor = torch.zeros(1, 2, H, W)
    mass_center = [0,0]
    
    total_weight = torch.sum(contact_mask)
    if total_weight == 0:
        return target_tensor, mass_center
    
    for x in range(W):
        for y in range(H):
            if contact_mask[y, x] > 0:
                mass_center[0] += constant_states[0][y, x] * contact_mask[y, x]
                mass_center[1] += constant_states[1][y, x] * contact_mask[y, x]
    
    mass_center[0] /= total_weight
    mass_center[1] /= total_weight
    
    mass_center = [float(mass_center[0]), float(mass_center[1])]
    
    target_tensor[0, 0] = mass_center[0] * contact_mask
    target_tensor[0, 1] = mass_center[1] * contact_mask
    target_tensor = target_tensor.squeeze(0)

    return target_tensor, mass_center


def _calculate_distance(estimation_states: torch.Tensor, 
                        target_tensor: torch.Tensor, 
                        contact_mask: torch.Tensor,
                        dead_mask:torch.Tensor = None) -> (np.array, np.array):
    
    update_steps = estimation_states.shape[0]
    means = []
    stds = []

    for update_step in range(update_steps):
        estimation = estimation_states[update_step]

        Cs = contact_mask.detach().cpu().numpy().flatten()
        if dead_mask is not None:
            Dm = dead_mask.detach().cpu().numpy().flatten()
            Cs = Cs * Dm

        Xs, Ys = estimation[0].detach().cpu().numpy().flatten()[Cs > 0], estimation[1].detach().cpu().numpy().flatten()[Cs > 0]        
        Xt, Yt = target_tensor[0].detach().cpu().numpy().flatten()[Cs > 0], target_tensor[1].detach().cpu().numpy().flatten()[Cs > 0]
        
        distance = np.sqrt((Xs - Xt)**2 + (Ys - Yt)**2)
        mean_distance, std_distance = float(np.mean(distance)), float(np.std(distance))

        #print(f'update_step: {update_step}, mean: {mean_distance}, std: {std_distance}')
        means.append(mean_distance)
        stds.append(std_distance)
    
    return means, stds


def _create_dead_mask(alive_percent: float, device: torch.device) -> torch.Tensor:
    dead_mask = torch.zeros(_config.BOARD_SHAPE, device=device)
    dead_mask = dead_mask.flatten()
    dead_mask[:int(dead_mask.shape[0] * alive_percent)] = 1
    dead_mask = dead_mask[torch.randperm(dead_mask.shape[0])]
    dead_mask = dead_mask.reshape(_config.BOARD_SHAPE)
    return dead_mask

if __name__ == '__main__':
    '''
    Do not run this, it will overwrite the data. And it is a rather slow experiment.
    '''
    import random

    from Util import set_seed
    seed = random.randint(0, 100000)
    set_seed(seed)

    experiment_name=f'Robustness'
    _folders.set_experiment_folders(experiment_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_names = ['Chebyshev','Chebyshev_Robust']


    for model_name in model_names:
        model = _folders.load_model(model_name).to(device).eval()
        _config.set_parameters({
            'BOARD_SHAPE' :     [8,8],
            'BATCH_SIZE' :      10,
            'NUM_MOVEMENTS' :   10,
            'NEIGHBORHOOD' :    'Chebyshev',
            })

        state_structure = StateStructure(
            estimation_dim  = 2,    
            constant_dim    = 2,   
            sensor_dim      = 1,
            hidden_dim      = 10)

        initial_states = Util.create_initial_states(_config.BATCH_SIZE, state_structure, _config.BOARD_SHAPE).to(device)
        experiment_data = []

        alive_percents = [i/10 for i in range(11)]

        shapes_groups = [Shapes.tetrominos, Shapes.unknown_shapes]
        shapes_names = ['tetrominos', 'unknown_shapes']

        for gorup, shape_name in zip(shapes_groups, shapes_names):        
            for batch in range(_config.BATCH_SIZE):
                for al in alive_percents:
                    print(f'batch: {batch}, alive_percent: {al}')
                    contact_masks, batch_data = _moving_contact_masks(gorup)
                    contact_masks = contact_masks.to(device)
                    
                    batch_data['alive_percent'] = al
                    dead_mask = _create_dead_mask(al, device)

                    for movement in range(_config.NUM_MOVEMENTS):
                        initial_state_movement = initial_states[batch]
                        initial_state_movement[..., state_structure.sensor_channels, :, :] = contact_masks[movement]
                        initial_states[batch] = initial_state_movement
                        
                        update_steps = 50
                        
                        with torch.no_grad():
                            output_states:torch.Tensor = model(initial_state_movement, update_steps, return_frames=True, dead_mask=dead_mask)

                        estimation_states = output_states[...,state_structure.estimation_channels, :, :]
                        constant_states = initial_state_movement[..., state_structure.constant_channels, :, :]
                        target_tensor, mc = _get_target_tensor(constant_states, contact_masks[movement])

                        means, stds = _calculate_distance(estimation_states, target_tensor, contact_masks[movement])
                        
                        batch_data['movements'][movement]['target_center'] = mc
                        batch_data['movements'][movement]['means'] = means
                        batch_data['movements'][movement]['stds'] = stds            
                        
                    experiment_data.append(batch_data)

            data_path = f'{_folders.RESULTS_PATH}/{model_name}_{shape_name}.json'
            with open(data_path, 'w') as f:
                json.dump(experiment_data, f)

            if os.path.getsize(data_path) > 100000000:
                import gzip
                with open(data_path, 'rb') as f_in:
                    with gzip.open(data_path + '.gz', 'wb') as f_out:
                        f_out.writelines(f_in)
                os.remove(data_path)

