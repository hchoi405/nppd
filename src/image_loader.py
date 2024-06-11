import os
import numpy as np
import exr
import glob
import multiprocessing as mp
import pathlib
from pathlib import Path
import tqdm
import json

import torch
from torch.utils.data import IterableDataset


# An empty wrapper class for images
class Var:
    def __init__(self):
        pass

    # Operator [] overloading
    def __getitem__(self, key):
        return self.__dict__[key]

    def to_torch(self, device='cuda'):
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                self.__dict__[key] = torch.from_numpy(value).to(device)
            elif isinstance(value, Var):
                value.to_torch(device)
        return self

def load_exr(path):
    numpy_img = exr.read_all(path)["default"]
    # Re-order the channels from RGB to BGR
    numpy_img = np.transpose(numpy_img, (2, 0, 1))
    return numpy_img

def save_as_npy(path):
    npy_path = path.replace('.exr', '.npy')
    if os.path.exists(npy_path):
        return
    numpy_img = load_exr(path)

    # Invalid value handling
    if np.isnan(numpy_img).any():
        print('There is NaN in', npy_path, 'Set it to zero for training.')
        numpy_img = np.nan_to_num(numpy_img, copy=False)
    if np.isposinf(numpy_img).any() or np.isneginf(numpy_img).any():
        print("There is INF in", npy_path, 'Set it to zero for training.')
        numpy_img[numpy_img == np.inf] = 0
        numpy_img[numpy_img == -np.inf] = 0

    np.save(npy_path, numpy_img)

# Make symolic link to files in orig_dir in new_dir using exr_dict
def make_symbolic(orig_dir, new_dir, exr_dict):
    print('Start to make symbolic links from', orig_dir, 'to', new_dir)

    new_dict = {}

    # Make symbolic links
    for key, files in exr_dict.items():
        for file in files:
            basename = os.path.basename(file)
            orig_path = os.path.join(orig_dir, file)
            new_file = os.path.join(new_dir, basename)
            # Make symbolic link
            if os.path.exists(new_file):
                os.remove(new_file)
            os.symlink(orig_path, new_file)

def extract_name_frame(filename):
    # Split the path into parts using the appropriate separator
    if os.path.sep == "\\":
        parts = filename.split("\\")
    else:
        parts = filename.split("/")
    # Extract the name from the last part
    name_and_number, ext = parts[-1].split(".", 1)
    name, frame = name_and_number.rsplit("_", 1)
    return name, int(frame)

def extract_filenames(filenames):
    # Extract the names from the file paths
    names = []
    # Dicts for each type
    files_dict = {}
    for filename in filenames:
        name, frame = extract_name_frame(filename)

        # Add the name to the list if it is not already there
        if name not in names:
            names.append(name)

        # Add the file to the dict
        if name not in files_dict:
            files_dict[name] = []
        files_dict[name].append(filename)
    return names, files_dict

# Function to check if all types of files have the same number of files and equal frame indices
def check_files(files_dict):
    # Check if all files have the same number of files
    num_files = {key: len(files) for key, files in files_dict.items()}
    all_same = len(set(num_files.values())) == 1
    if not all_same:
        raise Exception(f'Error: different number of files\n{num_files}')
    
    # Check if all files have equal frame indices
    frames_dict = {}
    for key, files in files_dict.items():
        frames = set()
        for file in files:
            filename = os.path.basename(file)
            name, frame = os.path.splitext(filename)[0].rsplit('_', 1)
            frames.add(int(frame))
        frames_dict[key] = frames
        if not 'frames_compare' in locals():
            frames_compare = frames
            continue
        if frames != frames_compare:
            raise Exception(f"Error: {key} has different frame indices than other types of files\n\t{frames}\n\t{frames_compare}")

    # Check if frame_dict is empty
    if not frames_dict:
        raise Exception('frames_dict is empty')

    # Check if any type of frames_dict has different frame indices than other types
    for key, frames in frames_dict.items():
        for key2, frames2 in frames_dict.items():
            if key != key2 and frames != frames2:
                raise(f"Error: {key} has different frame indices than {key2}")

class NpyDataset(IterableDataset):
    def __init__(self, directory, types, samples, max_num_frames=101, scene_dir='./scenes', *args, **kwargs):
        super(NpyDataset, self).__init__(*args, **kwargs)
        self.samples = samples

        # List files in format "{type}_{frame:04d}.exr"
        img_list = sorted(glob.glob(os.path.join(directory, "*.exr")))
        img_list = [os.path.basename(x) for x in img_list]
        img_list = [x for x in img_list if x.rsplit("_", 1)[0] in types]
        assert len(img_list) > 0, directory # Check emtpy

        # Remove excess frames of all types from the list
        if max_num_frames is not None:
            img_list = [x for x in img_list if int(x.rsplit("_", 1)[1].split(".")[0]) < max_num_frames]

        # Parse file type
        unique_types, exr_dict = extract_filenames(img_list)
        check_files(exr_dict)

        # Check if all types are given
        if set(types) != set(unique_types):
            raise Exception(f"Error: {set(types) - set(unique_types)} is not given in directory: {directory}")

        # Make a new directory
        parent_dir = os.path.dirname(directory)
        if not os.path.exists(scene_dir):
            os.makedirs(scene_dir)
            # Write parent directory of input to directory.txt
            with open(os.path.join(scene_dir, 'directory.txt'), 'w') as f:
                f.write(parent_dir)

        # Check if given directory is same as the one in directory.txt
        with open(os.path.join(scene_dir, 'directory.txt'), 'r') as f:
            orig_dir = f.read().replace('\n', '')
            if orig_dir != parent_dir:
                raise Exception(f"Error:\n\tloadded: {parent_dir}\n\tstored: {orig_dir}")

        new_dir = os.path.join(scene_dir, pathlib.PurePath(directory).name)
        # Make a data directory
        if not Path(new_dir).exists():
            print(f'Making a directory: {new_dir}')
            os.makedirs(new_dir)
            
        # Make a symbolic link of files in orig_dir in new_dir if not exist
        make_symbolic(directory, new_dir, exr_dict)

        # Check if the first and last saved npy of all types are same as exr
        npy_paths = [os.path.join(new_dir, f"{t}_{0:04d}.npy") for t in unique_types]
        npy_paths += [os.path.join(new_dir, f"{t}_{max_num_frames-1:04d}.npy") for t in unique_types]
        exr_paths = [os.path.join(directory, f"{t}_{0:04d}.exr") for t in unique_types]
        exr_paths += [os.path.join(directory, f"{t}_{max_num_frames-1:04d}.exr") for t in unique_types]

        for npy_path, exr_path in zip(npy_paths, exr_paths):
            if os.path.exists(npy_path):
                exr_img = load_exr(exr_path)
                exr_img = np.nan_to_num(exr_img, copy=False)
                npy_img = np.load(npy_path)
                assert np.allclose(exr_img, npy_img), f"Error: {exr_path} and {npy_path} are different"

        # Save exr images as npy
        fullpath_list = [os.path.join(new_dir, x) for x in img_list]
        print('Making npy files for faster loading... ')
        with mp.Pool(24) as p:
            list(tqdm.tqdm(p.imap(save_as_npy, fullpath_list), total=len(fullpath_list)))
        print('Done')
        
        # Change extension to npy
        img_list = [x.replace('.exr', '.npy') for x in img_list]

        # Get unique types
        unique_types = set([x.rsplit("_", 1)[0] for x in img_list])

        # Check each type has the same number of files
        num_frames = len(img_list) // len(unique_types)
        for t in unique_types:
            num_type = len([x for x in img_list if x.rsplit("_", 1)[0] == t])
            assert num_type == num_frames

        # Check each type has the same start number of frame
        start_frame = min([int(x.rsplit("_", 1)[1].split(".")[0]) for x in img_list])
        for t in unique_types:
            frame = min(
                [
                    int(x.rsplit("_", 1)[1].split(".")[0])
                    for x in img_list
                    if x.rsplit("_", 1)[0] == t
                ]
            )
            assert frame == start_frame

        # Check each type has the same max number of frame
        max_frame = max([int(x.rsplit("_", 1)[1].split(".")[0]) for x in img_list])
        for t in unique_types:
            frame = max(
                [
                    int(x.rsplit("_", 1)[1].split(".")[0])
                    for x in img_list
                    if x.rsplit("_", 1)[0] == t
                ]
            )
            assert frame == max_frame

        # Set for later use
        self.start_frame = start_frame
        self.num_frames = num_frames
        self.directory = new_dir
        self.unique_types = unique_types

        print(f"Directory '{directory}' has types: \n\t{unique_types}\nwith {num_frames} frames")

        # Load camera_info.json
        camera_info_path = os.path.join(directory, 'camera_info.json')
        with open(camera_info_path) as f:
            self.camera_info = json.load(f)

    def preprocess_camera_info(self, camera_info):
        view_proj_mat = np.array(camera_info['viewProjMatrix'], dtype=np.float32).reshape([4, 4])
        proj_mat = np.array(camera_info['projMatrix'], dtype=np.float32).reshape([4, 4])
        camera_position = np.array(camera_info['position'], dtype=np.float32)
        camera_target = np.array(camera_info['target'], dtype=np.float32)
        camera_up = np.array(camera_info['upVector'], dtype=np.float32)

        W = camera_target - camera_position # forward
        W = W / np.linalg.norm(W)
        U = np.cross(W, camera_up)
        U = U / np.linalg.norm(U)
        V = np.cross(U, W)
        V = V / np.linalg.norm(V)

        ret = {
            'view_proj_mat': view_proj_mat,
            'camera_position': camera_position,
            'camera_forward': W,
            'camera_up': V,
            'camera_left': U,
            'crop_offset': np.array([28, 0], dtype=np.int32),
        }

        return ret

    def screen_space_normal(self, w_normal, W, V, U):
        """Transforms per-sample world-space normals to screen-space / relative to camera direction

        Args:
            w_normal (ndarray, 3HWS): per-sample world-space normals
            W (ndarray, size (3)): vector in world-space that points forward in screen-space
            V (ndarray, size (3)): vector in world-space that points up in screen-space
            U (ndarray, size (3)): vector in world-space that points right in screen-space
        
        Returns:
            normal (ndarray, 3HWS): per-sample screen-space normals
        """
        # TODO: support any number of extra dimensions like apply_array
        return np.einsum('ij, ihws -> jhws', np.stack([W, U, V], axis=1), w_normal) # column vectors

    def log_depth(self, w_position, pos):
        """Computes per-sample compressed depth (disparity-ish)

        Args:
            w_position (ndarray, 3HWS): per-sample world-space positions
            pos (ndarray, size (3)): the camera's position in world-space
        
        Returns:
            motion (ndarray, 1HWS): per-sample compressed depth
        """
        # TODO: support any number of extra dimensions like apply_array
        d = np.linalg.norm(w_position - np.reshape(pos, (3, 1, 1, 1)), axis=0, keepdims=True)
        return np.log(1 + 1/d)

    def screen_space_position(self, w_position, pv, height, width):
        """Projects per-sample world-space positions to screen-space (pixel coordinates)

        Args:
            w_normal (ndarray, 3HWS): per-sample world-space positions
            pv (ndarray, size (4,4)): camera view-projection matrix
            height (int): height of the camera resolution (in pixels)
            width (int): width of the camera resolution (in pixels)
        
        Returns:
            projected (ndarray, 2HWS): Per-sample screen-space position (pixel coordinates).
                IJ INDEXING! for gather ops and consistency, 
                see backproject_pixel_centers in noisebase.torch.projective for use with grid_sample.
                Degenerate positions give inf.
        """
        # TODO: support any number of extra dimensions like apply_array
        homogeneous = np.concatenate(( # Pad to homogeneous coordinates
            w_position,
            np.ones_like(w_position)[0:1]
        ))
        print("homogeneous.shape:", homogeneous.shape)

        # ROW VECTOR ALERT!
        # DirectX uses row vectors...
        projected = np.einsum('ij, ihws -> jhws', pv, homogeneous)
        projected = np.divide(
            projected[0:2], projected[3], 
            out = np.zeros_like(projected[0:2]),
            where = projected[3] != 0
        )

        # directx pixel coordinate fluff
        projected = projected * np.reshape([0.5 * width, -0.5 * height], (2, 1, 1, 1)).astype(np.float32) \
            + np.reshape([width / 2, height / 2], (2, 1, 1, 1)).astype(np.float32)

        projected = np.flip(projected, 0) #height, width; ij indexing

        return projected

    def motion_vectors(self, w_position, w_motion, pv, prev_pv, height, width):
        """Computes per-sample screen-space motion vectors (in pixels)

        Args:
            w_position (ndarray, 3HWS): per-sample world-space positions
            w_motion (ndarray, 3HWS): per-sample world-space positions
            pv (ndarray, size (4,4)): camera view-projection matrix
            prev_pv (ndarray, size (4,4)): camera view-projection matrix from previous frame
            height (int): height of the camera resolution (in pixels)
            width (int): width of the camera resolution (in pixels)
        
        Returns:
            motion (ndarray, 2HWS): Per-sample screen-space motion vectors (in pixels).
                IJ INDEXING! for gather ops and consistency, 
                see backproject_pixel_centers in noisebase.torch.projective for use with grid_sample.
                Degenerate positions give inf.
        """
        # TODO: support any number of extra dimensions like apply_array (only the docstring here)
        print("w_position.shape:", w_position.shape)
        print("pv.shape:", pv.shape)
        print('height:', height)
        print('width:', width)

        current = self.screen_space_position(w_position, pv, height, width)
        prev = self.screen_space_position(w_position+w_motion, prev_pv, height, width)

        motion = prev-current

        return motion

    def unpack(self, frame, imgs, camera_info):
        unpacked = Var()

        # Unpack images into a dictionary
        for i, type in enumerate(self.unique_types):
            unpacked.__dict__[type] = imgs[i]


        # Make as multi-samples
        if self.samples == 2:
            # Stack
            color1 = unpacked.current + unpacked.emissive_multi # Include emissive
            color2 = unpacked.current2 + unpacked.emissive_multi # Include emissive
            color = np.stack([color1, color2], axis=-1)
            normal = np.stack([unpacked.normal_multi, unpacked.normal_multi], axis=-1)
            position = np.stack([unpacked.position, unpacked.position], axis=-1)
            motion = np.stack([unpacked.mvec[:2,...], unpacked.mvec[:2,...]], axis=-1)
            if frame == 0:
                motion = np.zeros_like(motion)
            diffuse = np.stack([unpacked.albedo_multi, unpacked.albedo_multi], axis=-1)
        elif self.samples == 4:
            # Collect samples
            colors = [unpacked.color1 + unpacked.emissive1, unpacked.color2 + unpacked.emissive2, unpacked.color3 + unpacked.emissive3, unpacked.color4 + unpacked.emissive4]
            normals = [unpacked.normal1, unpacked.normal2, unpacked.normal3, unpacked.normal4]
            positions = [unpacked.position1, unpacked.position2, unpacked.position3, unpacked.position4]
            diffuse = [unpacked.albedo1, unpacked.albedo2, unpacked.albedo3, unpacked.albedo4]
            # Stack
            color = np.stack(colors, axis=-1)
            normal = np.stack(normals, axis=-1)
            position = np.stack(positions, axis=-1)
            motion = np.stack([unpacked.mvec[:2,...] for _ in range(4)], axis=-1)
            if frame == 0:
                motion = np.zeros_like(motion)
            diffuse = np.stack(diffuse, axis=-1)
        else:
            raise Exception(f"Error: samples={self.samples} is not supported")
        reference = unpacked.ref

        # Crop
        color = color[..., 28:-28, :, :]
        normal = normal[..., 28:-28, :, :]
        position = position[..., 28:-28, :, :]
        motion = motion[..., 28:-28, :, :]
        diffuse = diffuse[..., 28:-28, :, :]
        reference = reference[:, 28:-28, :]

        # Reset __dict__ to exclude unnecessary types
        unpacked.__dict__ = {}
        unpacked.frame_index = np.array((frame,), dtype=np.int32)

        # Unpack camera_info into a dictionary
        ret = self.preprocess_camera_info(camera_info[frame])
        unpacked.__dict__.update(ret)
        
        # Make prev_camera
        unpacked.prev_camera = Var()
        if frame > 0:
            prev_ret = self.preprocess_camera_info(camera_info[frame-1])
            unpacked.prev_camera.__dict__.update(prev_ret)
        else:
            unpacked.prev_camera.__dict__.update(ret)
        
        # Transform
        unpacked.color = color
        unpacked.normal = self.screen_space_normal(normal, unpacked.camera_forward, unpacked.camera_up, unpacked.camera_left).astype(np.float32)
        unpacked.depth = self.log_depth(position, unpacked.camera_position)
        unpacked.diffuse = diffuse
        unpacked.reference = reference
        # Multiply by height and width
        height = normal.shape[1]
        width = normal.shape[2]
        motion *= np.array([width, height], dtype=np.float32).reshape([2, 1, 1, 1])
        ## HJ: this may not be necessary since motion is already in pixel coordinates
        # motion = self.motion_vectors(
        #     position, motion, 
        #     unpacked.view_proj_mat, unpacked.prev_camera.view_proj_mat,
        #     height, width
        # )
        unpacked.motion = np.clip(motion, -5e3, 5e3)

        # Add batch dimension
        for key, value in unpacked.__dict__.items():
            if isinstance(value, np.ndarray):
                unpacked.__dict__[key] = np.expand_dims(value, axis=0)
            elif isinstance(value, Var):
                for key2, value2 in value.__dict__.items():
                    if isinstance(value2, np.ndarray):
                        value.__dict__[key2] = np.expand_dims(value2, axis=0)
        
        return unpacked

    def __iter__(self):
        # Make filename using directory, unique_types and frame
        for frame in range(self.start_frame, self.start_frame + self.num_frames):
            filenames = [
                os.path.join(self.directory, f"{t}_{frame:04d}.npy")
                for t in self.unique_types
            ]
            ext_imgs = list(map(np.load, filenames))
            ret = self.unpack(frame, ext_imgs, self.camera_info)

            yield ret

    def __len__(self):
        return self.num_frames
