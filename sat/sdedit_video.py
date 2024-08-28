import argparse
import gc
import math
import os
import sys

from email import generator
from tabnanny import check
from typing import List, Union

import cv2
import imageio
import lovely_tensors as lt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TT
import torchvision.transforms.functional as TF

from arguments import get_args
from diffusion_video import SATVideoDiffusionEngine
from einops import rearrange, repeat
from fire import Fire
from omegaconf import ListConfig
from PIL import Image
from sgm.util import append_dims
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import center_crop, resize
from tqdm import tqdm, trange

from sat import mpu
from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint


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


def load_frames(frame_dir, start_frame_idx=90, num_frames=90, view_idx=0, fps=30, ignore_fps=False):
    frames = []
    frame_step = 1 if ignore_fps else 30 // fps
    for i in trange(start_frame_idx, start_frame_idx + num_frames * frame_step, frame_step, desc="Loading frames"):
        frame_path = os.path.join(frame_dir, f"render_frame{i:03d}_train{view_idx:02d}_last.png")
        assert os.path.exists(frame_path), f"Frame {frame_path} does not exist."
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = TF.to_tensor(frame)

        frames.append(frame)
    return frames


def check_inputs(frame_dir):
    assert os.path.exists(frame_dir), f"Frame directory {frame_dir} does not exist."


@torch.no_grad()
def sampling_main(args, model_cls):

    check_inputs(args.sdedit_frames_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    if isinstance(model_cls, type):
        model: SATVideoDiffusionEngine = get_model(args, model_cls)
    else:
        model: SATVideoDiffusionEngine = model_cls

    load_checkpoint(model, args)

    model.eval()
    device = model.device
    torch_dtype = model.dtype

    ### Reading input prompts
    # if args.input_type == "cli":
    #     data_iter = read_from_cli()
    # elif args.input_type == "txt":
    #     rank, world_size = mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
    #     print("rank and world_size", rank, world_size)
    #     data_iter = read_from_file(args.input_file, rank=rank, world_size=world_size)
    # else:
    #     raise NotImplementedError
    prompt_idx = args.sdedit_prompt_idx
    with open(args.input_file, "r") as f:
        text_prompts = f.readlines()
    text_prompts = [x.strip() for x in text_prompts]
    prompt = text_prompts[prompt_idx]

    ### Reading the frames
    frames_dir = args.sdedit_frames_dir
    start_idx = args.sdedit_start_idx
    num_frames = args.sdedit_num_frames
    view_idx = args.sdedit_view_idx
    ignore_input_fps = args.sdedit_ignore_input_fps

    frames_tensor = load_frames(
        frames_dir,
        start_frame_idx=start_idx,
        num_frames=num_frames,
        view_idx=view_idx,
        ignore_fps=ignore_input_fps,
    )

    out_fps = args.sampling_fps

    frames_tensor = torch.stack(frames_tensor, dim=0).to(torch_dtype)
    frames_tensor = frames_tensor.unsqueeze(0)  # B, T, C, H, W
    input_video_path = f"{args.output_dir}/input_sfi{start_idx}_nf{num_frames}_v{view_idx}_fps{out_fps}.mp4"
    if not os.path.exists(input_video_path):
        save_video(frames_tensor, input_video_path, fps=out_fps)
    frames_tensor_norm = frames_tensor * 2.0 - 1.0

    ### Prepare the model for sampling
    sdedit_strength = 1.0  # 1.0 means full sampling (all sigmas are returned)
    if args.sdedit_strength is not None:
        sdedit_strength = args.sdedit_strength
        if sdedit_strength == 'all':
            all_strenths = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
        elif isinstance(sdedit_strength, float) and 0.0 <= sdedit_strength <= 1.0:
            all_strenths = [sdedit_strength]
        else:
            raise ValueError(f"Invalid sdedit_strength: {sdedit_strength}")
    else:
        all_strenths = [sdedit_strength]

    image_size = [480, 720]

    T, H, W, C, F = args.sampling_num_frames, image_size[0], image_size[1], args.latent_channels, 8
    num_samples = [1]
    force_uc_zero_embeddings = ["txt"]

    value_dict = {
        "prompt": prompt,
        "negative_prompt": "",
        "num_frames": torch.tensor(T).unsqueeze(0),
    }

    batch, batch_uc = get_batch(
        get_unique_embedder_keys_from_conditioner(model.conditioner),
        value_dict,
        num_samples,
    )
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            print(key, batch[key].shape)
        elif isinstance(batch[key], list):
            print(key, [len(l) for l in batch[key]])
        else:
            print(key, batch[key])
    c, uc = model.conditioner.get_unconditional_conditioning(
        batch,
        batch_uc=batch_uc,
        force_uc_zero_embeddings=force_uc_zero_embeddings,
    )

    for k in c:
        if not k == "crossattn":
            c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc))

    # Offload the model from GPU to save GPU memory
    model.to("cpu")
    torch.cuda.empty_cache()
    model.first_stage_model.to(device)
    # B, T, C, H, W -> B, C, T, H, W
    frames_tensor_norm = frames_tensor_norm.permute(0, 2, 1, 3, 4).contiguous().to(device)

    # batch is not used in `encode_first_stage` method
    frames_z = model.encode_first_stage(frames_tensor_norm, batch)
    # B, C, T, H, W -> B, T, C, H, W
    frames_z = frames_z.permute(0, 2, 1, 3, 4).contiguous()
    assert frames_z.shape == (1, T, C, H // F, W // F), f"Encoded frames_z shape: {frames_z.shape} not correct"


    for strength in all_strenths:
        cur_sdedit_strength = strength
        # Unload the first stage model from GPU to save GPU memory
        model.to(device)
        model.first_stage_model.to("cpu")

        samples_z = model.sample(
            c,
            uc=uc,
            batch_size=1,
            shape=(T, C, H // F, W // F),
            frames_z=frames_z,
            sdedit_strength=cur_sdedit_strength,
        )
        # B, T, C, H, W -> B, C, T, H, W
        samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()

        # Unload the model from GPU to save GPU memory
        model.to("cpu")
        torch.cuda.empty_cache()
        model.first_stage_model.to(device)
        # first_stage_model = model.first_stage_model
        # first_stage_model = first_stage_model.to(device)

        latent = 1.0 / model.scale_factor * samples_z

        ## Decode latent serial to save GPU memory
        recons = []
        loop_num = (T - 1) // 2
        for i in range(loop_num):
            if i == 0:
                start_frame, end_frame = 0, 3
            else:
                start_frame, end_frame = i * 2 + 1, i * 2 + 3
            if i == loop_num - 1:
                clear_fake_cp_cache = True
            else:
                clear_fake_cp_cache = False

            recon = model.first_stage_model.decode(
                latent[:, :, start_frame:end_frame].contiguous(),
                clear_fake_cp_cache=clear_fake_cp_cache,
            )

            recons.append(recon)

        recon = torch.cat(recons, dim=2).to(torch.float32)
        # B, C, T, H, W -> B, T, C, H, W
        samples_x = recon.permute(0, 2, 1, 3, 4).contiguous()
        samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()

        basename = f"view{view_idx}_start{start_idx:03d}_frames{num_frames}_strength{cur_sdedit_strength}"
        basename = basename.replace(".", "d").replace("-", "n")
        output_video_path = os.path.join(args.output_dir, f"{basename}.mp4")
        output_frames_path = os.path.join(args.output_dir, f"{basename}_frames")
        os.makedirs(output_frames_path, exist_ok=True)
        save_frames(samples.squeeze(0), output_frames_path)
        save_video(samples, output_video_path, fps=out_fps)
        print(f"Saved video to {output_video_path}")
        print(f"Saved frames to {output_frames_path}")

    # save_path = os.path.join(
    #     args.output_dir, str(cnt) + "_" + prompt.replace(" ", "_").replace("/", "")[:120], str(index)
    # )
    # if mpu.get_model_parallel_rank() == 0:
    #     save_video_as_grid_and_mp4(samples, output_video_path, fps=args.sampling_fps)


if __name__ == "__main__":
    lt.monkey_patch()
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    py_parser = argparse.ArgumentParser(add_help=False)
    known, args_list = py_parser.parse_known_args()

    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    del args.deepspeed_config
    args.model_config.first_stage_config.params.cp_size = 1
    args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False

    sampling_main(args, model_cls=SATVideoDiffusionEngine)
