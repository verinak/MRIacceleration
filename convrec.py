import torch

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    print("Using GPU:", torch.cuda.get_device_name(device))
else:
    device = torch.device('cpu')
    print("Using CPU")

import fastmri
from fastmri.data import transforms as T
import h5py
import numpy as np
import sys
sys.path.insert(0, 'src/')

from ucnnreco import CartesianScampi
from pathlib import Path
from utils.plot_utils import disp
from utils.params import RecoParams
from utils.cartesian.sampling import *

def crop_kspace(kspace_data, target_width=640, target_height=320):
    """
    Crop k-space data from (4, 768, 392) to the target size (4, 640, 320).

    Parameters:
        kspace_data (numpy.ndarray): The k-space data to be cropped. Expected shape (4, 768, 392).
        target_width (int): The target width of the cropped k-space data (default is 640).
        target_height (int): The target height of the cropped k-space data (default is 320).

    Returns:
        numpy.ndarray: The cropped k-space data with shape (4, 640, 320).
    """
    kspace_shape = kspace_data.shape

    # Ensure the target size is smaller than or equal to the k-space dimensions
    assert target_width <= kspace_shape[1] and target_height <= kspace_shape[2], \
        "Target size must be smaller than or equal to k-space dimensions."

    # Calculate cropping ranges for the center of the k-space
    start_x = (kspace_shape[1] - target_width) // 2  # for width (768 -> 640)
    start_y = (kspace_shape[2] - target_height) // 2  # for height (392 -> 320)

    # Crop the k-space data (keeping the coil dimension intact)
    cropped_kspace = kspace_data[:, start_x:start_x + target_width, start_y:start_y + target_height]

    return cropped_kspace

def center_crop(img_tensor, crop_size=320):
    h, w = img_tensor.shape
    startx = w // 2 - crop_size // 2
    starty = h // 2 - crop_size // 2
    return img_tensor[starty:starty+crop_size, startx:startx+crop_size]


def conv_rec(slice_ksp, acq):
    if slice_ksp.shape[2] > 320:
      slice_ksp = crop_kspace(slice_ksp, target_width=640, target_height=320)
    kdata = torch.from_numpy(slice_ksp)

    if acq == "CORPD_FBK" or acq == "CORPDFS_FBK":
      mtype = 'knee'
    else:
      mtype = 'brain'

    # generate mask
    mask_gen = Sampling()
    mask = mask_gen.generate_2d_mask(
        img_shape=slice_ksp[0].T.shape,
        accel=4.0,
        rsampling='gaussian',
        save=False,
        scale=0.5,           # required for 'gaussian'
        calibregion=32       # optional, number of central lines to always include
    )
    mask = torch.from_numpy(mask)
    print(slice_ksp.shape)
    print(mask.shape)

    # save data
    if not os.path.exists("/content/img_data/"):
        os.makedirs("/content/img_data/")
    torch.save(kdata, '/content/img_data/slice_kspace.pt')
    torch.save(mask, '/content/img_data/mask.pt')

    # load params
    if mtype == 'knee':
      params_path = 'src/config/CScampi.json'
    else:
      params_path = 'src/config/CScampi2.json'

    cartesian_params = RecoParams()
    cartesian_params.from_json(params_path)
    # print(cartesian_params)

    # initialize model
    reco = CartesianScampi(recopars=cartesian_params, device=device)
    reco.set_data_path(full_kspace_path=Path('/content/img_data/slice_kspace')
                    , mask_path=Path('/content/img_data/mask'))
    reco.prep_data()
    reco.prep_model(mtype=mtype)
    gt = reco.get_gt()

    # perform reconstruction
    res = reco()

    # crop images
    RESC = center_crop(res.squeeze(0), crop_size=slice_ksp.shape[2])
    k=torch.abs(RESC).cpu().numpy()

    RESC2 = center_crop(gt.squeeze(0), crop_size=slice_ksp.shape[2])
    k2=torch.abs(RESC2).cpu().numpy()

    return k2, k

