import noisebase
from model import Model
import hydra
import torch
import os
import glob
import re
import exr
import numpy as np

import image_loader

SAMPLES = 2
# SCENES = ['BistroExterior2', 'BistroExteriorDynamic', 'EmeraldSquare2', 'staircase']
if SAMPLES == 2:
    INPUT_DIR = f'/home/hchoi/nas/dataset_newscene_2spp/'
    SCENES = ['BistroExterior2', 'BistroExteriorDynamic', 'staircase']
    TMP_DIR = 'scenes_2spp'
elif SAMPLES == 4:
    INPUT_DIR = f'/home/hchoi/nas/dataset_newscene_4spp_nppd/'
    SCENES = ['EmeraldSquare2']
    TMP_DIR = 'scenes_4spp'
OUTPUT_DIR = '/home/hchoi/nas/nppd'

@hydra.main(version_base=None, config_path="../conf", config_name="small_2_spp")
def main(cfg):
    output_folder = os.path.join('outputs', cfg['name'])
    ckpt_folder = os.path.join(output_folder, 'ckpt_epoch')
    ckpt_files = glob.glob(os.path.join(ckpt_folder, "*.ckpt"))

    def extract_val_loss(filename):
        name = os.path.basename(filename)
        # Matches 'val_loss=' followed digits, a decimal point, and more digits (e.g., 'val_loss=0.123')
        return float(re.search(r'val_loss=(\d+.\d+)', name).group(1))

    best_model_path = min(ckpt_files, key=lambda x: extract_val_loss(x))

    test_set = hydra.utils.instantiate(cfg['test_data'])
    model = Model.load_from_checkpoint(**cfg['model'], checkpoint_path=best_model_path)

    if SAMPLES == 2:
        read_types = ['current'] # default
        read_types += ['current2'] # another color
        read_types += ['position'] # World-space position (no multi-sampled version)
        read_types += ['mvec'] # screen-space motion vector
        read_types += ['albedo_multi', 'normal_multi'] # Multi-sample G-buf
        read_types += ['emissive_multi']
        read_types += ['ref']
    elif SAMPLES == 4:
        read_types = ['color1', 'normal1', 'position1', 'albedo1', 'emissive1']
        read_types += ['color2', 'normal2', 'position2', 'albedo2', 'emissive2']
        read_types += ['color3', 'normal3', 'position3', 'albedo3', 'emissive3']
        read_types += ['color4', 'normal4', 'position4', 'albedo4', 'emissive4']
        read_types += ['mvec']
        read_types += ['ref']

    for SCENE in SCENES:
        loader = image_loader.NpyDataset(os.path.join(INPUT_DIR, SCENE), read_types, samples=SAMPLES, max_num_frames=301, scene_dir=TMP_DIR)
        out_dir = os.path.join(OUTPUT_DIR, SCENE)
        os.makedirs(out_dir, exist_ok=True)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        first = True
        for frame in loader:
            frame.to_torch()
            frame_index = frame.frame_index.cpu().item()
            torch.cuda.synchronize()

            start.record()
            if first:
                first = False
                model.temporal = model.temporal_init(frame)

            with torch.no_grad():
                output = model.test_step(frame)
            end.record()
            end.synchronize()
            time = start.elapsed_time(end)
            print('Frame', frame_index, 'Elapsed time:', time, 'ms')
            img = np.transpose(output.cpu().numpy()[0], [1, 2, 0])
            # exr.write(os.path.join(out_dir, f'nppd_{frame_index:04d}.exr'), img, compression=exr.ZIP_COMPRESSION)

if __name__ == '__main__':
    main()