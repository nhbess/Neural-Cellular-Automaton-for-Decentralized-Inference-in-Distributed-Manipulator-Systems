from Environment.Tile import Tile
from Environment.Tetromino import Tetromino
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point

class ContactBoard:
    def __init__(self, shape: [int, int]) -> None:
        self.shape = shape
        self.tiles = self._create_tiles()
        self.contour = self._create_contour()

    def _create_contour(self) -> np.array:
        contour_vertices = [(0,0), (0, self.shape[0]), (self.shape[1], self.shape[0]), (self.shape[1], 0)]
        contour_vertices = [(point[0] - 0.5, point[1] - 0.5) for point in contour_vertices]
        contour = Polygon(contour_vertices)
        return contour
    
    def _create_tiles(self) -> None:
        tiles = []
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                tile = Tile(1)
                tile.center = np.array([j, i])
                tile.matrix_position = np.array([j, i])
                tiles.append(tile)
        return tiles

    def has_tetromino_inside(self, tetromino: Tetromino) -> bool:
            return self.contour.contains(tetromino.polygon)
    
    def has_point_inside(self, point: tuple) -> bool:
        return self.contour.contains(Point(point))
    
    def get_contact_mask(self, tetromino: Tetromino) -> np.array:
        contact_mask = np.zeros(self.shape, dtype=int)
        for tile in self.tiles:
            if tetromino.polygon.buffer(1e-6).contains(tile.sensor):
                row_index = self.shape[0] - tile.matrix_position[1] - 1
                col_index = tile.matrix_position[0]
                contact_mask[row_index, col_index] = 1
        return contact_mask


if __name__ == '__main__':
    pass