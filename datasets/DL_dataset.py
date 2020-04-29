import numpy as np
import PIL.Image as pil
import os
import random
import torch
import torch.utils.data as data
from torchvision import transforms

from .mono_dataset import MonoDataset

SAMPLES = 126
CAMS = 6
train_VIDS = 120
val_VIDS = 14


img_wt = 320
img_ht = 256

image_size = [img_wt, img_ht]

# This code is valid only if num_scales = 1. i.e. scales = [0,]

class DL_dataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        # Img_ext: jpeg
        super(DL_dataset, self).__init__(*args, **kwargs)

        self.K = [np.array([[879.03824732, 0, 613.17597314, 0],
                            [0, 879.03824732, 524.14407205, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32),
                  np.array([[882.61644117, 0, 621.63358525, 0],
                            [0, 882.61644117, 524.38397862, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32),
                  np.array([[880.41134027, 0, 618.9494972, 0],
                            [0, 880.41134027, 521.38918482, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32),
                  np.array([[881.28264688, 0, 612.29732111, 0],
                            [0, 881.28264688, 521.77447199, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32),
                  np.array([[882.93018422, 0, 616.45479905, 0],
                            [0, 882.93018422, 528.27123027, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32),
                  np.array([[881.63835671, 0, 607.66308183, 0],
                            [0, 881.63835671, 525.6185326, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32)]

        # Factor of 4 used because images have been resized
        for i in range(len(self.K)):
            self.K[i][0, :] /= 4
            self.K[i][1, :] /= 4

        self.inv_K = [np.linalg.pinv(x) for x in self.K]

        self.K = [torch.from_numpy(x) for x in self.K]
        self.inv_K = [torch.from_numpy(x) for x in self.inv_K]
        

        

        assert len(range(self.num_scales)) == 1
        if self.load_depth:
            raise ValueError("We do not have ground truth depth")
        if "s" in self.frame_idxs:
            raise ValueError("We do not use stereo pairs")


        self.cam_list = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

        self.num_vids = train_VIDS if self.is_train else val_VIDS

    def check_depth(self):
        return False
    
    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.
        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:
            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.
        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.
        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        # Extract indices for files
        ind = index

        cam_index = ind%CAMS
        ind /= CAMS

        samp_index = ind%(SAMPLES-2)+1
        ind /= SAMPLES-2

        scene_index = ind%num_vids
        if self.is_train:
            scene += train_VIDS

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)


        for i in self.frame_idxs:
            im = self.get_color(scene_index, samp_index + i, self.cam_list[cam_index], do_flip)
            inputs[("color", i, 0)] = self.to_tensor(im.copy())
            inputs[("color_aug", i, 0)] = self.to_tensor(color_aug(im))
            

        inputs[("K", 0)] = self.K[cam_index].copy()
        inputs[("inv_K", 0)] = self.inv_K[cam_index].copy()

        return inputs

    def len():
        return self.num_vids*(SAMPLES-2)*CAMS # Do not take first and last frame of each scene

    def get_image_path(self, scene_index, samp_index, cam):
        file_name = f"scene_{scene_index}/sample_{samp_index}/{cam}.jpeg"
        image_path = os.path.join(self.data_path, file_name)
        return image_path

    def get_color(self, scene_index, samp_index, cam, do_flip):
        color = self.loader(self.get_image_path(scene_index, samp_index, cam)).resize(256, 320)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        
        return color
