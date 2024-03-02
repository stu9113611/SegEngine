import math

import numpy as np
from engine.category import Category
from numpy.typing import NDArray
import torch
from typing import Union


class IdMapVisualizer:
    def __init__(self, categories: list[Category]) -> None:
        self.categories = categories

    def _visualize_one(self, id_map: Union[torch.Tensor, NDArray]) -> NDArray[np.uint8]:
        """Visualize an id map (BGR) with given categories.

        Args:
            id_map (Union[torch.Tensor, NDArray]): A HxW map that consists of category ids.

        Returns:
            NDArray[np.uint8]: A visualized 8-bit BGR image (HxWx3) given categories.
        """

        w, h = id_map.shape
        vis = np.zeros((w, h, 3), dtype=np.uint8)
        for cat in self.categories:
            vis[id_map == cat.id] = (cat.b, cat.g, cat.r)
        return vis

    def _make_grid(
        self, vis_list: list[np.ndarray], nrow: int = 8
    ) -> NDArray[np.uint8]:
        """Re-arrange a list of visualizations into grid.

        Args:
            vis_list (list[np.ndarray]): A list of visualizations
            nrow (int, optional): Number of visualizations in a row. Defaults to 8.

        Returns:
            NDArray[np.uint8]: A grid of visualizations.
        """
        n = len(vis_list)  # batch size
        m = n % nrow  # number of rows
        ncol = math.ceil(n / nrow)  # number of columns
        if n > nrow and m:
            for _ in range(m):
                vis_list.append(np.zeros(vis_list[0].shape))
        return np.concatenate(
            [
                np.concatenate(vis_list[nc * nrow : (nc + 1) * nrow], 1)
                for nc in range(ncol)
            ],
            0,
        )

        # """Visualize a batch of index map (NxHxW) or a single index map (HxW) to a picture (HxWx3) with given palatte (categories)."""

    def visualize(
        self, id_map: Union[torch.Tensor, NDArray], nrow: int = 8
    ) -> NDArray[np.uint8]:
        """Visualize a batch of id map (NxHxW) or one index map (HxW) to a grid of BGR image (HxWx3) with given categories.

        Args:
            id_map (Union[torch.Tensor, NDArray]): A HxW or NxHxW map that consists of category ids.
            nrow (int, optional): Number of visualizations in a row. Defaults to 8.

        Returns:
            NDArray[np.uint8]: A grid visualization of id map with given categories.
        """
        assert (
            len(id_map.shape) == 2 or len(id_map.shape) == 3
        ), "The shape of index map should be NxWxH or WxH."

        if len(id_map.shape) == 2:
            id_map = np.expand_dims(id_map, 0)

        if id_map.shape[0] < nrow:
            nrow = id_map.shape[0]

        return self._make_grid([self._visualize_one(im) for im in id_map], nrow)
