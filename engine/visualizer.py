from engine.category import Category
from numpy.typing import NDArray
import numpy as np
import math


class IndexMapVisualizer:
    def __init__(self, categories: list[Category]) -> None:
        self.categories = categories

    def _visualize_one(self, index_map: NDArray) -> NDArray:
        """Visualize a single index map with given palatte."""
        w, h = index_map.shape
        visualization = np.zeros((w, h, 3))
        for cat in self.categories:
            visualization[index_map == cat.id] = (cat.b, cat.g, cat.r)
        return visualization

    def make_grid(self, vis: list[np.ndarray], nrow: int = 8) -> np.ndarray:
        """Arrange the visualizations into grid."""
        m = len(vis) % nrow
        if m != 0:
            for _ in range(m):
                vis.append(np.zeros(vis[0].shape))
        b = len(vis)
        ncol = math.ceil(b / nrow)
        output = []
        for nc in range(ncol):
            row = np.concatenate(vis[nc * nrow : (nc + 1) * nrow], 1)
            output.append(row)
        return np.concatenate(output, 0)

    def visualize(self, index_map: np.ndarray, nrow: int = 8) -> np.ndarray:
        """Visualize a batch of index map (NxHxW) or a single index map (HxW) to a picture (HxWx3) with given palatte (categories)."""
        assert (
            len(index_map.shape) == 2 or len(index_map.shape) == 3
        ), "The shape of index map should be NxWxH or WxH."

        if len(index_map.shape) == 2:
            index_map = np.expand_dims(index_map, 0)

        return self.make_grid([self._visualize_one(im) for im in index_map], nrow)
