from loader import Loader
from filtering import Filter
from data_transformer import DataTransformer
from interface import Interface
from vizualizer import Vizualizer
from renderer import Renderer

import torch
import numpy as np


class MainLoader(Loader, Filter, DataTransformer, Interface, Vizualizer, Renderer):
    def __init__(self, config_path):
        super().__init__(config_path)


if __name__ == '__main__':
    main_loader = MainLoader()

    '''
    main_loader.get_merged_car_mask(10)
    main_loader.render_by_all_bounding_boxes(torch.tensor([[10., 5., -1., 3.8, 2., 1.5, np.pi],
                                                           [10., -5., -1., 4.5, 1.8, 1.5, np.pi/2]],
                                                          device="cuda", dtype=torch.float32,
                                                          requires_grad=True))
    '''
