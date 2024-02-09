from shapely.geometry import Point, Polygon
import shapely.affinity
import numpy as np

class Tile:
    def __init__(self, tile_size:float) -> None:
        self.tile_size = tile_size
        self.polygon = Polygon([(0, 0), (self.tile_size, 0), (self.tile_size, self.tile_size), (0, self.tile_size)])
        self.sensor = Point(np.array([self.polygon.centroid.coords.xy[0][0], self.polygon.centroid.coords.xy[1][0]]))
        self.matrix_position = None

    @property
    def center(self) -> tuple:
        center = self.polygon.centroid.coords.xy
        return np.array([center[0][0], center[1][0]])
    
    @center.setter
    def center(self, new_center: tuple) -> None:
        translation_x, translation_y = np.subtract(new_center, self.center)
        self.polygon = shapely.affinity.translate(self.polygon, xoff=translation_x, yoff=translation_y, zoff=0.0)
        self.sensor = shapely.affinity.translate(self.sensor, xoff=translation_x, yoff=translation_y, zoff=0.0)
    
    def __str__(self) -> str:
        return (self.center, self.matrix_position)
    
if __name__ == '__main__':
    pass