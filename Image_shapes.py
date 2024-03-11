import numpy as np
from matplotlib import pyplot as plt

import Environment.Shapes

if __name__ == '__main__':
    from _colors import create_palette
    
    shapes_groups = [Environment.Shapes.tetros_dict, Environment.Shapes.unknown_shapes_dict]
    
    
    for n, group in enumerate(shapes_groups):
    
        previous_origin = -0.5
        x_max = 0
        y_max = 0

        shapes = list(group.values())
        names = list(group.keys())
        palette  = create_palette(len(shapes))


        tick_positions = []
        for i in range(len(shapes)):
            shape = shapes[i]

            shape[:, 1] = shape[:, 1] - min(shape[:, 1])
            shape[:, 0] = shape[:, 0] - min(shape[:, 0])

            shape = shape + [previous_origin + 0.5, 0]
            
            previous_origin = max(shape[:, 0])
            
            xs, ys = shape[:, 0], shape[:, 1]
            xs = [*xs, xs[0]]
            ys = [*ys, ys[0]]  

            x_max = max(x_max, max(xs))
            y_max = max(y_max, max(ys))

            shape_center = np.mean(shape, axis=0)
            tick_positions.append(shape_center[0])

            plt.plot(xs, ys, color='black', linewidth=1.5, zorder=2)
            plt.fill(xs, ys, color=palette[i], alpha=1, zorder=1)

        pad = 0.25
        plt.xlim(0 - pad, x_max + pad)
        plt.ylim(0 - pad, y_max + pad)
        
        plt.xticks(tick_positions, names, rotation=0)

        #plt.axis('off')
        #set y axis False
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().set_aspect('equal', adjustable='box')
        #ticks font size
        plt.xticks(fontsize=12)
        plt.tight_layout()
        path_image = f'__Images/shapes_group_{n}.png'
        plt.savefig(path_image, dpi=300, bbox_inches='tight', pad_inches=0, transparent=False)
        plt.show()
        plt.clf()