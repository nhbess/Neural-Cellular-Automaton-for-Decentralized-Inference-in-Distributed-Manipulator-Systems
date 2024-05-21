import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torch import nn

import _colors
import _config
import Environment.Shapes as Shapes
import _folders
from Environment.ContactBoard import ContactBoard
from Environment.Tetromino import Tetromino
from StateStructure import StateStructure
from Util import calculate_mass_center


_colors.FIG_SIZE = (6, 3)
SHAPE_NAME = 'T'

if SHAPE_NAME == 'U':

    INITIAL_POSITION = [3.4, 3.6]
    INITIAL_ANGLE = 188

elif SHAPE_NAME == 'T':
    INITIAL_POSITION = [3.1, 4.1]
    INITIAL_ANGLE = 45

else:
    INITIAL_POSITION = [3.1, 4.1]
    INITIAL_ANGLE = 45

palette = _colors.create_palette(5)
style = {
    'fig_width':  4,
    'fig_height': 4,
    'color_tile_contour': [0,0,0],
    'color_tile_contact': palette[2],
    'color_tile_no_contact': palette[1],
    'color_sensor': palette[0],
    
    'sensor_marker': 'x',
    'tetro_color': palette[3],
    'tetro_center_color': palette[4],
    'tetro_line_width': 2,
}

def get_contact_mask(n_movements: int, 
                     height: int, 
                     width: int,                                           
                     show:bool=False,
                     ) -> torch.Tensor:
    
    board = ContactBoard(shape=[height, width])
    contact_masks = torch.zeros(n_movements, 1, height, width)
    
    all_shapes = {**Shapes.tetros_dict, **Shapes.unknown_shapes_dict}
    shape = all_shapes[SHAPE_NAME] 
    
    scaler = 2
    tetro = Tetromino(constructor_vertices=shape, scaler=scaler)
    tetro.center = np.array(INITIAL_POSITION)
    tetro.rotate(INITIAL_ANGLE)
    
    for movement_index in range(n_movements):
        if movement_index != 0:
            while True:
                displacement = np.random.uniform(-1, 1, size=2)
                if board.has_point_inside(displacement + tetro.center):
                    break
            tetro.translate(displacement)
            tetro.rotate(np.random.uniform(0, 20))
        contact_i = board.get_contact_mask(tetro)
        contact_masks[movement_index, 0] = torch.from_numpy(contact_i)
        if show:
            import matplotlib.pyplot as plt
            plt.imshow(contact_i)
            plt.show()

    return contact_masks

def create_states_dynamic(batch_size:int, state_structure:StateStructure, board_shape:tuple, n_movements:int = 1):
    contact_mask = get_contact_mask(n_movements=n_movements, 
                                    height=board_shape[0],
                                    width=board_shape[1],
                                    show=False)
    
    contact_mask = contact_mask.unsqueeze(2)
    init_state = torch.zeros(n_movements,batch_size,state_structure.state_dim, *board_shape)
    
    lowx = 0
    lowy = 0
    highx = lowx + board_shape[0]-1
    highy = lowy + board_shape[1]-1

    x, y = torch.meshgrid(torch.linspace(lowx, highx, board_shape[0]), torch.linspace(lowy, highy, board_shape[1]), indexing='ij')
    
    init_state[..., state_structure.constant_channels, :, :] = torch.stack([x, y], dim=0)
    init_state[..., state_structure.estimation_channels, :, :] = torch.stack([x, y], dim=0) 
    init_state[..., state_structure.sensor_channels, :, :] = contact_mask

    return init_state

def _simulate_center_finder(model:nn.Module, state_structure: StateStructure, board_shape:tuple, batch_size:int, n_movements:int, output_file_name:str = None, num_steps = 50) -> torch.Tensor:
    logger.info(f'Simulating {batch_size}')
    init_states = create_states_dynamic(batch_size, state_structure, board_shape, n_movements)
    model = model.cpu()
    all_frames = []
    with torch.no_grad():
        for movement in range(n_movements):
            init_state = init_states[movement]            
            simulated_frames = model.forward(input_state=init_state, num_steps= num_steps, return_frames = True)
            simulated_frames = simulated_frames.detach().numpy()
            
            simulated_frames = simulated_frames[:-1, ...]
            simulated_frames = np.concatenate((init_state.unsqueeze(0).detach().numpy(), simulated_frames), axis=0)
            simulated_frames[0, ...] = init_state.detach().numpy()
            all_frames.append(simulated_frames)
    
    np.save(output_file_name, all_frames)

def _visualize_from_file(simulation_path: str, animation_path:str, state_structure: StateStructure):
    logger.info(f'Visualizing dynamic input frames from file: {simulation_path}')

    simulation = np.load(file=simulation_path)
    movements = simulation.shape[0]
    steps = simulation.shape[1]
    batches = simulation.shape[2]
    board_shape = simulation.shape[-2:]
    
    reshaped = simulation.reshape(movements*steps, batches, state_structure.state_dim, *board_shape)
    
    steps_to_show = 3
    index_to_show = np.linspace(0, steps-1, steps_to_show, endpoint=True, dtype=int)
    #TODO: careful here this is manually set
    index_to_show = [0, steps//4, steps-1]
    
    for b in range(batches):
        batch = reshaped[:,b, ...]
        sensible = batch[..., state_structure.estimation_channels, :, :]
        sensor = batch[..., state_structure.sensor_channels, :, :]
        constant = batch[..., state_structure.constant_channels, :, :]
        mass_centers = [calculate_mass_center(sensor[i, :, :, :], constant_state=constant[i, :, :, :]) for i in range(steps*movements)]    
        
        
        fig, ax = plt.subplots(1, steps_to_show, figsize=_colors.FIG_SIZE)
        print(index_to_show)

        scaler = 2
        all_shapes = {**Shapes.tetros_dict, **Shapes.unknown_shapes_dict}
        shape = all_shapes[SHAPE_NAME]
        tetro = Tetromino(constructor_vertices=shape, scaler=scaler)
        tetro.center = np.array([INITIAL_POSITION[1], INITIAL_POSITION[0]])
        tetro.rotate(180-INITIAL_ANGLE)
        

        for i in range(steps_to_show):
            step = index_to_show[i]
            print('i:', i, 'step: ', step)
            
            ax[i].clear()
            #ax[i].imshow(sensor[step, 0, :, :], cmap='gray', vmin=0, vmax=1, origin='lower')
            
            #Tiles            
            #draw grid
            for j in range(board_shape[0]+1):
                for k in range(board_shape[1]+1):
                    ax[i].plot([k-0.5,k-0.5],[0-0.5,board_shape[0]-0.5], color='black', zorder = 2, linewidth=1)
                    ax[i].plot([0-0.5,board_shape[1]-0.5],[k-0.5,k-0.5], color='black', zorder = 2, linewidth=1)


            for j in range(board_shape[0]):
                for k in range(board_shape[1]):
                    x = [k-0.5, k+0.5, k+0.5, k-0.5]
                    y = [j-0.5, j-0.5, j+0.5, j+0.5]
                    if sensor[step, 0, j, k] == 1:
                        ax[i].fill(x, y, color=style['color_tile_contact'], alpha = 1)

                    else:
                          ax[i].fill(x, y, color=style['color_tile_no_contact'], alpha = 1)
            
            #remove axis
            ax[i].axis('off')
            ax[i].set_xticks([])
            ax[i].set_yticks([])

            #plot sensor only if sensor is on
            x_channel = np.array(sensible[step, 0, :, :].tolist()).flatten()/board_shape[0]
            y_channel = np.array(sensible[step, 1, :, :].tolist()).flatten()/board_shape[1]
            ax[i].scatter(y_channel*(board_shape[0]), x_channel*(board_shape[1]), color=style['color_sensor'], marker=style['sensor_marker'], s=20, zorder = 3, label='Estimation')
            
            tetro.center = [mass_centers[step][1], mass_centers[step][0]]
            x_values, y_values = zip(*tetro.vertices)
            ax[i].plot(x_values, y_values, color=style['tetro_color'], linewidth=style['tetro_line_width'], )
            ax[i].plot(tetro.center[0], tetro.center[1], 'x', color=style['tetro_center_color'],markeredgewidth=3, markersize = 10,zorder = 10)
            ax[i].fill(x_values, y_values, color=style['tetro_color'], alpha = 0.5)
            
            #calculate error
            x_channel = np.array(sensible[step, 0, :, :].tolist()).flatten()
            y_channel = np.array(sensible[step, 1, :, :].tolist()).flatten()
            s_channel = np.array(sensor[step, 0, :, :].tolist()).flatten()
            x_channel = x_channel[s_channel == 1]
            y_channel = y_channel[s_channel == 1]
            
            distances = np.sqrt((x_channel - tetro.center[0])**2 + (y_channel - tetro.center[1])**2)
            error = np.mean(distances)
            std = np.std(distances)
            ax[i].set_title(f'Step: {step}\n$\mu_e={error:.2f}$  $\sigma_e={std:.2f}$')
            ax[i].set_aspect('equal', adjustable='box')
            plt.tight_layout()
            plt.subplots_adjust(wspace=-0.35, hspace=-0.35)
            #plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    #add legend
    plt.tight_layout()
    plt.savefig(animation_path, dpi=600, bbox_inches='tight', pad_inches=0.02)
    #plt.show()


def run_block(state_structure:StateStructure, seed = None, visualize=False):
    if seed is not None:
        from Util import set_seed
        set_seed(seed)

    model = _folders.load_model('Chebyshev')

    if visualize:
        simulation_path = f"{_folders.SIMULATIONS_PATH}/{'test'}.npy"

        _simulate_center_finder(model = model, 
                               state_structure = state_structure, 
                               board_shape = [*_config.BOARD_SHAPE], 
                               batch_size=1, 
                               n_movements=1, 
                               output_file_name=simulation_path, 
                               num_steps=15)
        
        animation_path = f"{_folders.VISUALIZATIONS_PATH}/convergence_{SHAPE_NAME}.png"
        _visualize_from_file(simulation_path=simulation_path, animation_path=animation_path, state_structure=state_structure)

if __name__ == '__main__':

    experiment_name=f'Performance'
    _folders.set_experiment_folders(experiment_name)
    
    _config.set_parameters({
        'BOARD_SHAPE' :     [8,8],
        'NEIGHBORHOOD' :    'Chebyshev',
    })
    state_structure = StateStructure(
                        estimation_dim  = 2,    
                        constant_dim    = 2,   
                        sensor_dim      = 1,   
                        hidden_dim      = 10)
    
    run_block(state_structure=state_structure, seed=None, visualize=True)
