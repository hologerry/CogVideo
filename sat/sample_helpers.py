import math
import os

from typing import List, Union

import cv2
import imageio
import numpy as np
import torch
import torchvision.transforms as TT
import torchvision.transforms.functional as TF

from einops import rearrange
from omegaconf import ListConfig
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from tqdm import tqdm, trange


def read_from_cli():
    cnt = 0
    try:
        while True:
            x = input("Please input English text (Ctrl-D quit): ")
            yield x.strip(), cnt
            cnt += 1
    except EOFError as e:
        pass


def read_from_file(p, rank=0, world_size=1):
    with open(p, "r") as fin:
        cnt = -1
        for l in fin:
            cnt += 1
            if cnt % world_size != rank:
                continue
            yield l.strip(), cnt


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = np.repeat([value_dict["prompt"]], repeats=math.prod(N)).reshape(N).tolist()
            batch_uc["txt"] = np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N)).reshape(N).tolist()
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str, fps: int = 5, args=None, key=None):
    os.makedirs(save_path, exist_ok=True)

    for i, vid in enumerate(video_batch):
        gif_frames = []
        for frame in vid:
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            gif_frames.append(frame)
        now_save_path = os.path.join(save_path, f"{i:06d}.mp4")
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)


def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr


def save_video(tensor, video_path, fps=30):
    """
    Saves the video frames to a video file.

    Parameters:
        tensor (torch.Tensor): The video frames tensor.
        video_path (str): The path to save the output video.
    """
    tensor = tensor.float()
    frames = tensor[0].contiguous()  # drop the batch dimension

    writer = imageio.get_writer(video_path, fps=fps)
    for frame in frames:
        frame = rearrange(frame, "c h w -> h w c")
        frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
        writer.append_data(frame)
    writer.close()


def save_frames(frames, output_frames_path):
    for i, frame in tqdm(enumerate(frames), desc="Saving frames", total=len(frames)):
        if isinstance(frame, torch.Tensor):
            frame = TF.to_pil_image(frame)
        elif isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        frame.save(f"{output_frames_path}/{i:03d}.png")


def load_frames(frame_dir, start_frame_idx=90, num_frames=49, view_idx=0, fps=8, ignore_fps=False):
    frames = []
    frame_step = 1  # if ignore_fps else 30 // fps
    for i in trange(start_frame_idx, start_frame_idx + num_frames * frame_step, frame_step, desc="Loading frames"):
        frame_path = os.path.join(frame_dir, f"render_frame{i:03d}_train{view_idx:02d}_last.png")
        assert os.path.exists(frame_path), f"Frame {frame_path} does not exist."
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = TF.to_tensor(frame)

        frames.append(frame)
    return frames


def load_spherical_frames(frame_dir, start_frame_idx=90, num_frames=49, view_idx=0, fps=8, ignore_fps=False):
    frames = []
    frame_step = 1  # if ignore_fps else 30 // fps
    for i in trange(
        start_frame_idx, start_frame_idx + num_frames * frame_step, frame_step, desc=f"Loading {view_idx:03d} frames"
    ):
        frame_path = os.path.join(frame_dir, f"render_frame{i:03d}_spherical{view_idx:03d}.png")
        assert os.path.exists(frame_path), f"Frame {frame_path} does not exist."
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = TF.to_tensor(frame)

        frames.append(frame)
    return frames


def load_zero123_frames(frame_dir, start_frame_idx=90, num_frames=49, max_frame_idx=119, fps=8, ignore_fps=False):
    frames = []
    frame_step = 1  # if ignore_fps else 30 // fps
    for i in trange(start_frame_idx, start_frame_idx + num_frames * frame_step, frame_step, desc="Loading frames"):
        fi = min(i, max_frame_idx)
        frame_path = os.path.join(frame_dir, f"frame_{fi:06d}.png")
        assert os.path.exists(frame_path), f"Frame {frame_path} does not exist."
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Remove the background zero123 output noise
        frame[frame < 10] = 0
        frame = TF.to_tensor(frame)

        frames.append(frame)
    return frames


def load_label(label_folder, start_frame_idx=90, max_frame_idx=110, view_idx=0):
    # For label, we only have [10, 110, 10], num_frames must be 49
    label_name = f"sim_000000_cam_{view_idx:02d}_start_{start_frame_idx:03d}_frames_049.txt"
    label_path = os.path.join(label_folder, label_name)
    assert os.path.exists(label_path), f"Label {label_path} does not exist."
    with open(label_path, "r") as fin:
        label = fin.read().strip()
    return label


def check_inputs(frame_dir):
    assert os.path.exists(frame_dir), f"Frame directory {frame_dir} does not exist."
