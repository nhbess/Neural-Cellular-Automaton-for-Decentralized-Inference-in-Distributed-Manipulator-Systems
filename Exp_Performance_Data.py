import json
import random
import sys

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
        'name_shape': name_shape,
        'scaler': scaler,
        'movements': []
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


def _calculate_distance(estimation_states: torch.Tensor, target_tensor: torch.Tensor, contact_mask: torch.Tensor) -> (np.array, np.array):
    update_steps = estimation_states.shape[0]
    means = []
    stds = []

    for update_step in range(update_steps):
        estimation = estimation_states[update_step]

        Cs = contact_mask.detach().cpu().numpy().flatten()
        Xs, Ys = estimation[0].detach().cpu().numpy().flatten()[Cs > 0], estimation[1].detach().cpu().numpy().flatten()[Cs > 0]        
        Xt, Yt = target_tensor[0].detach().cpu().numpy().flatten()[Cs > 0], target_tensor[1].detach().cpu().numpy().flatten()[Cs > 0]
        
        distance = np.sqrt((Xs - Xt)**2 + (Ys - Yt)**2)
        mean_distance, std_distance = float(np.mean(distance)), float(np.std(distance))

        #print(f'update_step: {update_step}, mean: {mean_distance}, std: {std_distance}')
        means.append(mean_distance)
        stds.append(std_distance)
    
    return means, stds



if __name__ == '__main__':
    print('Running this experiment takes some time and it will overwrite previous results.')
    sys.exit()
    experiment_name=f'Performance'
    _folders.set_experiment_folders(experiment_name)

    shapes = [Shapes.tetrominos, Shapes.unknown_shapes]
    name_shapes = ['tetrominos', 'unknown_shapes']

    for shape_set, name_shape in zip(shapes, name_shapes):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'device: {device}')
        model_name = 'Chebyshev'
        model = _folders.load_model(model_name).to(device).eval()

        _config.set_parameters({
            'BOARD_SHAPE' :     [8,8],
            'BATCH_SIZE' :      500,
            'NUM_MOVEMENTS' :   50,
            'NEIGHBORHOOD' :    model_name,
            })

        state_structure = StateStructure(
            estimation_dim  = 2,    
            constant_dim    = 2,   
            sensor_dim      = 1,
            hidden_dim      = 10)

        initial_states = Util.create_initial_states(_config.BATCH_SIZE, state_structure, _config.BOARD_SHAPE).to(device)
        experiment_data = []

        for batch in range(_config.BATCH_SIZE):
            print(f'batch: {batch}')
            contact_masks, batch_data = _moving_contact_masks(shape_set)
            contact_masks = contact_masks.to(device)
            for movement in range(_config.NUM_MOVEMENTS):
                initial_state_movement = initial_states[batch]
                initial_state_movement[..., state_structure.sensor_channels, :, :] = contact_masks[movement]
                initial_states[batch] = initial_state_movement
                
                update_steps = np.random.randint(*_config.UPDATE_STEPS)
                update_steps = 50
                
                with torch.no_grad():
                    output_states:torch.Tensor = model(initial_state_movement, update_steps, return_frames=True)

                estimation_states = output_states[...,state_structure.estimation_channels, :, :]
                constant_states = initial_state_movement[..., state_structure.constant_channels, :, :]
                target_tensor, mc = _get_target_tensor(constant_states, contact_masks[movement])

                
                means, stds = _calculate_distance(estimation_states, target_tensor, contact_masks[movement])
                
                batch_data['movements'][movement]['target_center'] = mc
                batch_data['movements'][movement]['means'] = means
                batch_data['movements'][movement]['stds'] = stds            
                
            experiment_data.append(batch_data)

        data_path = f'{_folders.RESULTS_PATH}/{name_shape}.json'
        
        with open(data_path, 'w') as f:
            json.dump(experiment_data, f)
    