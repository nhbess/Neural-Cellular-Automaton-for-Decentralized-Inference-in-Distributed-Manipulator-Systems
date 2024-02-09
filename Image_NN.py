import copy

import matplotlib.pyplot as plt
import numpy as np
import shapely.affinity
from shapely.geometry import Point, Polygon

import _colors


class Object:
    def __init__(self, tile_size = 1, circle=False) -> None:
        self.tile_size = tile_size
        if circle:
            self.polygon = Point(0,0).buffer(self.tile_size)
        else:
            self.polygon = Polygon([(0, 0), (self.tile_size, 0), (self.tile_size, self.tile_size), (0, self.tile_size)])
        self._text = 'Text'
        self.color = 'black'

    @property
    def center(self) -> tuple:
        center = self.polygon.centroid.coords.xy
        return np.array([center[0][0], center[1][0]])
    
    @center.setter
    def center(self, new_center: tuple) -> None:
        translation_x, translation_y = np.subtract(new_center, self.center)
        self.polygon = shapely.affinity.translate(self.polygon, xoff=translation_x, yoff=translation_y, zoff=0.0)
    @property
    def text(self) -> str:
        return self._text
    @text.setter
    def text(self, text:str) -> None:
        self._text = text        
    
    @property
    def upper_middle(self) -> tuple:
        max_x = np.max([p[0] for p in self.polygon.exterior.coords])
        max_y = np.max([p[1] for p in self.polygon.exterior.coords])
        min_x = np.min([p[0] for p in self.polygon.exterior.coords])
        
        middle_x = (max_x + min_x) / 2
        middle_y = max_y + 0.2

        return (middle_x, middle_y)
    
    def plot(self, zorder:int = 0):
        x_values, y_values = zip(*self.polygon.exterior.coords)
        plt.plot(x_values, y_values, color='black', linewidth=1, label='Object', zorder=zorder)
        #plt.plot(self.center[0], self.center[1], 'x', color='black',linewidth=4,label='Object center')
        plt.fill(x_values, y_values, color=self.color, alpha = 0.6, zorder=zorder)
        plt.text(self.center[0], self.center[1], self._text, fontsize=17*self.tile_size, horizontalalignment='center', verticalalignment='center', zorder=zorder)
        

class Block:
    def __init__(self, objects:Object) -> None:
        self.objects = objects
        self._text = None

    @property
    def center(self) -> tuple:
        return np.mean([o.center for o in self.objects], axis=0)
    
    @center.setter
    def center(self, new_center: tuple) -> None:
        translation_x, translation_y = np.subtract(new_center, self.center)
        for o in self.objects:
            o.center = o.center + np.array([translation_x, translation_y])

    @property
    def upper_middle(self) -> tuple:
        max_x = np.max([o.center[0] for o in self.objects])
        max_y = np.max([o.center[1] for o in self.objects])
        min_x = np.min([o.center[0] for o in self.objects])
        
        pad = 0.25
        middle_x = (max_x + min_x) / 2
        middle_y = max_y + 0.5 + pad

        return (middle_x, middle_y)
    
    def plot(self):
        zorders = [i for i in range(len(self.objects))]
        zorders.reverse()
        
        if self._text is not None:
            plt.text(self.upper_middle[0], self.upper_middle[1] + 0.1, self._text, fontsize=15, horizontalalignment='center', verticalalignment='center')
        
        for zo,o in zip(zorders, self.objects):
            o.plot(zo)


if __name__ == '__main__':

    plt.figure(figsize=(5, 5))

    #State
    state = ['P','V','E','T','N']
    palette = _colors.create_palette(len(state))
    state_objects = []
    for i,t in enumerate(state):
        obj = Object()
        obj.center = (i*1,0)
        obj.text = f'{t}'
        obj.color = palette[i]
        state_objects.append(obj)

    state_block = Block(state_objects)
    state_block._text = '$State_{t}$'
    state_block.plot()

    #Neighborhood
    objects_ni = []
    for i,t in enumerate(['P','V','E','T']):
        obj = Object(tile_size=0.75)
        obj.center = (i*0.25,i*0.25)
        obj.text = f'${t}_0$'
        obj.color = palette[i]
        objects_ni.append(obj)

    ni = Block(objects_ni)
    ni.center = (2,2)
    
    objects_nm = []
    for i,t in enumerate(['P','V','E','T']):
        obj = Object(tile_size=0.75)
        obj.center = (i*0.25,i*0.25)
        obj.text = f'${t}_n$'
        obj.color = palette[i]
        objects_nm.append(obj)
    nm = Block(objects_nm)
    nm.center = (3.6,2)
    
    n_block = Block([*objects_ni, *objects_nm])
    n_block._text = '$N = [{N_0}, {\dots}, {N_n}]$'
    n_block.center = (4,2)
    n_block.plot()


    nn_object = Object(tile_size=1.25)
    nn_object.center = state_block.upper_middle - np.array([0,2.5])
    nn_object.text = '$NN$'
    nn_object.color = 'black'
    nn_object.plot()

    out_state = copy.deepcopy(state_objects)
    for obj in out_state:
        if obj.text not in ['E', 'T']:
            obj.text = ''
            obj.color = 'white'


    out_block = Block(out_state)
    out_block.center = nn_object.upper_middle - np.array([0,2.5])
    out_block.plot()

    plus = Object(tile_size=0.5, circle=True)
    plus.center = out_block.upper_middle - np.array([0,2.5])
    plus.text = '$+$'
    plus.color = 'black'
    plus.plot()

    out_state = copy.deepcopy(state_block)
    out_state._text = '$State_{t+1}$'
    out_state.center = plus.upper_middle - np.array([0,2.5])
    out_state.plot()

    #### Arrows
    #arrow = mpatches.Arrow(state_block.upper_middle[0], state_block.upper_middle[1] - 1.25, 0,-0.6, width=0.25, color = 'black')
    #plt.gca().add_patch(arrow)
    #arrow = mpatches.Arrow(nn_object.upper_middle[0], nn_object.upper_middle[1] - 1.5, 0,-0.5, width=0.25, color = 'black')
    #plt.gca().add_patch(arrow)
    #arrow = mpatches.Arrow(out_block.upper_middle[0], out_block.upper_middle[1] - 1.25, 0,-0.75, width=0.25, color = 'black')
    #plt.gca().add_patch(arrow)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.tight_layout()
    #plt.savefig('test.png', bbox_inches='tight')  # Save the figure without extra white space
    plt.show()