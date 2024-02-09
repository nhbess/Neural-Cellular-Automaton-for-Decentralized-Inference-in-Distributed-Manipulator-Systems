import matplotlib.pyplot as plt

import _colors
import Environment.Shapes as Shapes
from Environment.ContactBoard import ContactBoard
from Environment.Tetromino import Tetromino
from Environment.Tile import Tile


def plot_DMS(board:ContactBoard, tetromino: Tetromino = None, save=False, name='test.png', plot_text=False, style = None) -> None:
    palette = _colors.create_palette(5)
    _colors.show_palette(palette, save=True)
    if style is None:
        style = {
            'fig_width':  2,
            'fig_height': 2,

            'color_tile_contour': [0,0,0],
            'color_tile_contact': palette[2],
            'color_tile_no_contact': palette[1],
            'color_sensor': palette[0],
            
            'sensor_marker': 'o',
            'tetro_color': palette[3],

            'tetro_center_color': palette[4],
            'tetro_line_width': 3,
        }
    
    plt.figure(figsize=(_colors.FIG_SIZE))  # Set the figure size
    contact_mask = board.get_contact_mask(tetromino).T
    
    for tile in board.tiles:
        exterior_coords = tile.polygon.exterior.coords.xy
        x,y = tile.matrix_position
        W, H = board.shape
        plt.plot(exterior_coords[0], exterior_coords[1], color=style['color_tile_contour'])
        
        if contact_mask[x][W-y-1] == 0: plt.fill(exterior_coords[0], exterior_coords[1], color=list(style['color_tile_no_contact']))
        else: plt.fill(exterior_coords[0], exterior_coords[1], color=list(style['color_tile_contact']))

        plt.plot(tile.sensor.x, tile.sensor.y, style['sensor_marker'], color=list(style['color_sensor']))
        
    if tetromino is not None:
        
        x_values, y_values = zip(*tetromino.vertices)
        plt.plot(x_values, y_values, color=style['tetro_color'], linewidth=3, label='Object')
        plt.plot(tetromino.center[0], tetromino.center[1], 'x', color=style['tetro_center_color'],linewidth=4,label='Object center')
        plt.fill(x_values, y_values, color=style['tetro_color'], alpha = 0.5)


    plt.plot([], [], 'o', label="Sensor", color=style['color_sensor'])
    plt.plot([], [], 's', label="Tile in contact", color=style['color_tile_contact'])
    plt.plot([], [], 's', label="Tile no contact", color=style['color_tile_no_contact'])
    

    plt.gca().set_aspect('equal', adjustable='box')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    #remove axis
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    #limit the image to the board size
    plt.xlim(-0.5, board.shape[1] - 0.5)
    plt.ylim(-0.5, board.shape[0] - 0.5)
    plt.margins(0)
    plt.legend(loc='upper right', framealpha=1)
    
    if save:
        plt.savefig(name, bbox_inches='tight', dpi=600)
    else:
        plt.show()
    plt.close()

def plot_NCA(board:ContactBoard, tetromino: Tetromino = None, save=False, name='test.png', plot_text=False, style = None) -> None:
    palette = _colors.create_palette(5)
    _colors.show_palette(palette, save=True)
    if style is None:
        style = {
            'fig_width':  3,
            'fig_height': 3,

            'color_tile_contour': [0,0,0],
            'color_tile_contact': palette[2],
            'color_tile_no_contact': palette[1],
            'color_sensor': palette[0],
            
            'sensor_marker': 'o',
            'tetro_color': palette[3],

            'tetro_center_color': palette[4],
            'tetro_line_width': 3,
        }
    
    
    plt.figure(figsize=(style['fig_width'], style['fig_height']))  # Set the figure size
    contact_mask = board.get_contact_mask(tetromino).T
    
    W, H = board.shape
    if False: #Chebishev
        for i in range(W-1):
                for j in range(H-1):
                    plt.plot([j, j + 1], [i, i + 1], color='black', zorder = 2)  # top-left to bottom-right diagonal
                    plt.plot([j + 1, j], [i, i + 1], color='black', zorder = 2)  # top-right to bottom-left diagonal

    for tile in board.tiles:
        exterior_coords = tile.polygon.exterior.coords.xy
        x,y = tile.matrix_position
        
        #draw a grid
        plt.plot([x,x],[0,W-1], color='black', zorder = 2)
        plt.plot([0,H-1],[y,y], color='black', zorder = 2)
                
        
        if False: #Maybe
            if contact_mask[x][W-y-1] == 0: plt.fill(exterior_coords[0], exterior_coords[1], color=list(style['color_tile_no_contact']), zorder = 1)
            else: plt.fill(exterior_coords[0], exterior_coords[1], color=list(style['color_tile_contact']), zorder = 1)

        if contact_mask[x][W-y-1] == 0: color = style['color_tile_no_contact']
        else: color = style['color_tile_contact']
        plt.plot(tile.sensor.x, tile.sensor.y, style['sensor_marker'], color=[0,0,0], zorder = 3, markersize=15)
        plt.plot(tile.sensor.x, tile.sensor.y, style['sensor_marker'], color=color, zorder = 4, markersize=10)
        

    plt.plot([], [], '-', label=f'Comm', color=style['color_sensor'])
    plt.plot([], [], 'o', label=f'$NN$', color=[0,0,0])
    plt.plot([], [], 'o', label="Contact", color=style['color_tile_contact'])
    plt.plot([], [], 'o', label="No contact", color=style['color_tile_no_contact'])
    

    plt.gca().set_aspect('equal', adjustable='box')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    #remove axis
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    #limit the image to the board size
    plt.xlim(-0.5, board.shape[1] - 0.5)
    plt.ylim(-0.5, board.shape[0] - 0.5)
    plt.margins(0)
    #plt.title('Contact Board')
    plt.legend(loc='upper right', framealpha=1)
    if save:
        plt.savefig(name, bbox_inches='tight')  # Save the figure without extra white space
    else:
        plt.show()
    plt.close()


    
W,H = 5, 6
board = ContactBoard(shape=[W,H])
tetro = Tetromino(Shapes.tetros_dict['T'], scaler=1.5)
tetro.center = (H/3, W/2.9)
tetro.rotate(-18)
contact = board.get_contact_mask(tetro)

plot_DMS(board, tetro, save=True, name='DMS.png', plot_text=False, style = None)
plot_NCA(board, tetro, save=True, name='NCA.png', plot_text=False, style = None)