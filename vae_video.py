import argparse
import math
import os

import cv2
import lovely_tensors as lt
import torch
import torchvision.transforms.functional as TF

from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from tqdm import trange

from arguments import get_args
from diffusion_video import SATVideoDiffusionEngine
from sample_helpers import (
    check_inputs,
    get_batch,
    get_unique_embedder_keys_from_conditioner,
    save_frames,
    save_video,
)


def load_frames(
    frame_dir,
    output_root,
    cur_folder_name,
    sequence_name,
    view_id=0,
    clip_id=10,
    start_frame_idx=0,
    num_frames=96,
    vae_frame_size=(720, 480),
):
    out_dir_gt_vae_size = os.path.join(output_root, f"{cur_folder_name}_gtv")
    os.makedirs(out_dir_gt_vae_size, exist_ok=True)

    frames = []
    for i in trange(start_frame_idx, start_frame_idx + num_frames, desc="Loading frames"):
        # frame_path = os.path.join(frame_dir, sequence_name, f"view{view_id:02d}_clip{clip_id:02d}", f"im{i:05d}.png")
        # /data/Free/sparse_view_codec_outputs/out_bin/ds20220811_view0_clip10_seqview00_clip10_q35/im00001.png
        frame_path = os.path.join(frame_dir, f"im{i+1:05d}.png")
        assert os.path.exists(frame_path), f"Frame {frame_path} does not exist."
        frame = cv2.imread(frame_path)
        frame = cv2.resize(frame, vae_frame_size, interpolation=cv2.INTER_AREA)
        out_path = os.path.join(out_dir_gt_vae_size, f"im{i:05d}.png")
        cv2.imwrite(out_path, frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = TF.to_tensor(frame)
        frames.append(frame)

    return frames


@torch.no_grad()
def sampling_main(args, model_cls):

    if isinstance(model_cls, type):
        model: SATVideoDiffusionEngine = get_model(args, model_cls)
    else:
        model: SATVideoDiffusionEngine = model_cls

    load_checkpoint(model, args)

    model.eval()
    device = model.device
    torch_dtype = model.dtype
    prompt = ""

    view_id = 6
    clip_id = 10

    data_root = "/data/Free/sparse_view_codec_outputs"
    bin_root = f"{data_root}/out_bin/ds20220811_view{view_id}_clip{clip_id}_seqview0{view_id}_clip{clip_id}_q56"
    output_root = f"{data_root}/5b_0930"

    sequence_name = "20220811162658"
    cur_folder_name = f"vae55_{sequence_name}_view{view_id:02d}_clip{clip_id:02d}"

    start_frame_idx = 47
    total_frames = 49
    raw_frame_size = (1280, 960)
    vae_frame_size = (720, 480)

    frames = load_frames(
        bin_root,
        output_root,
        cur_folder_name,
        sequence_name,
        view_id,
        clip_id,
        start_frame_idx=start_frame_idx,
        num_frames=total_frames,
        vae_frame_size=vae_frame_size,
    )
    raw_size_output_dir = os.path.join(output_root, f"{cur_folder_name}_r")
    vae_size_output_dir = os.path.join(output_root, f"{cur_folder_name}_v")
    os.makedirs(raw_size_output_dir, exist_ok=True)
    os.makedirs(vae_size_output_dir, exist_ok=True)

    frames_tensor = torch.stack(frames, dim=0)
    frames_tensor = frames_tensor.to(torch_dtype)
    frames_tensor = frames_tensor.unsqueeze(0)  # B, T, C, H, W

    frames_tensor_norm = frames_tensor * 2.0 - 1.0

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
    assert frames_z.shape == (1, C, T, H // F, W // F), f"Encoded frames_z shape: {frames_z.shape} not correct"

    latent = 1.0 / model.scale_factor * frames_z

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
    recon_recon = samples[0]
    cur_n_frames = recon_recon.shape[0]
    for i in range(cur_n_frames):
        frame = recon_recon[i]
        frame = frame.permute(1, 2, 0).float().numpy()
        frame = (frame * 255).astype("uint8")
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        raw_size_frame = cv2.resize(frame, raw_frame_size, interpolation=cv2.INTER_CUBIC)
        raw_size_out_path = os.path.join(raw_size_output_dir, f"im{start_frame_idx + i:05d}.png")
        vae_size_out_path = os.path.join(vae_size_output_dir, f"im{start_frame_idx + i:05d}.png")
        cv2.imwrite(raw_size_out_path, raw_size_frame)
        cv2.imwrite(vae_size_out_path, frame)


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
