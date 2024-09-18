import argparse
import math
import os

from json import load

import lovely_tensors as lt
import numpy as np
import torch

from arguments import get_args
from diffusion_video import SATVideoDiffusionEngine
from sample_helpers import (
    check_inputs,
    get_batch,
    get_unique_embedder_keys_from_conditioner,
    load_label,
    load_zero123_frames,
    load_fake_prefix_frames,
    save_frames,
    save_video,
)
from tqdm import trange

from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint


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

    prefix_frames_dir = args.sdedit_prefix_frames_dir
    prefix_start_idx = args.sdedit_prefix_start_idx
    prefix_num_frames = args.sdedit_prefix_num_frames

    labels_dir = args.sdedit_labels_dir

    frames_dir = args.sdedit_frames_dir
    start_idx = args.sdedit_start_idx
    num_frames = args.sdedit_num_frames
    view_idx = args.sdedit_view_idx
    ignore_input_fps = args.sdedit_ignore_input_fps

    # the frame 20 in scalarflow is the first scalarreal frame
    frame_idx_to_label_idx_offset = 20
    frame_batch_size = 2
    prefix_num_latent_frames = prefix_num_frames // 3

    if args.sdedit_tgt_view_idx == "all":
        tgt_view_ids = [i for i in range(5)]
    else:
        tgt_view_ids = [args.sdedit_tgt_view_idx]


    for tgt_view in tgt_view_ids:
        if view_idx == tgt_view:
            continue

        if args.sdedit_zero123_use_ckp2 and args.sdedit_zero123_use_ckp2_psnr:
            zero123_output_dir = f"zero123_ckp2_finetune_38000_cam{view_idx}to{tgt_view}_psnr_for_cogvideox"
        elif args.sdedit_zero123_use_ckp2:
            zero123_output_dir = f"zero123_ckp2_finetune_38000_cam{view_idx}to{tgt_view}_for_cogvideox"
        else:
            zero123_output_dir = f"zero123_finetune_15000_cam{view_idx}to{tgt_view}_for_cogvideox"

        prefix_output_dir = zero123_output_dir.replace("for_cogvideox", "cogvideox_5b_all_pred_single")
        prefix_output_full_dir = os.path.join(args.output_dir, prefix_output_dir, f"output_sfi000_nf65_strength0d26_frames")

        cogvx_output_dir = zero123_output_dir.replace("for_cogvideox", f"cogvideox_5b_all_pred_prefix_one")
        cogvx_output_full_dir = os.path.join(args.output_dir, cogvx_output_dir)
        os.makedirs(cogvx_output_full_dir, exist_ok=True)

        model.to(device)

        prefix_frames_tensor = load_fake_prefix_frames(
            prefix_output_full_dir,
            start_frame_idx=prefix_start_idx,
            num_frames=prefix_num_frames,
            view_idx=view_idx,
            ignore_fps=ignore_input_fps,
        )
        cur_num_frames = num_frames - prefix_num_frames

        cur_frames_tensor = load_zero123_frames(
            os.path.join(frames_dir, zero123_output_dir),
            start_frame_idx=start_idx,
            num_frames=cur_num_frames,
            max_frame_idx=119,
            ignore_fps=ignore_input_fps,
        )
        frames_tensor = prefix_frames_tensor + cur_frames_tensor

        label_start_idx = prefix_start_idx

        prompt = load_label(
            labels_dir,
            start_frame_idx=(frame_idx_to_label_idx_offset + label_start_idx) // 10 * 10,
            max_frame_idx=110,
            view_idx=tgt_view,
        )

        out_fps = args.sampling_fps

        frames_tensor = torch.stack(frames_tensor, dim=0)
        frames_tensor = frames_tensor.to(torch_dtype)
        frames_tensor = frames_tensor.unsqueeze(0)  # B, T, C, H, W

        input_video_path = f"{cogvx_output_full_dir}/input_sfi{start_idx}_nf{num_frames}_fps{out_fps}.mp4"
        save_video(frames_tensor.float(), input_video_path, fps=out_fps)

        input_frames_path = f"{cogvx_output_full_dir}/input_sfi{start_idx}_nf{num_frames}_fps{out_fps}_frames"
        os.makedirs(input_frames_path, exist_ok=True)
        save_frames(frames_tensor.float().squeeze(0), input_frames_path)

        frames_tensor_norm = frames_tensor * 2.0 - 1.0

        ### Prepare the model for sampling
        # sdedit_strength = 1.0  # 1.0 means full sampling (all sigmas are returned)
        # if args.sdedit_strength is not None:
        #     sdedit_strength = args.sdedit_strength
        #     if sdedit_strength == "all":
        #         all_strenths = np.arange(args.sdedit_strength_min, args.sdedit_strength_max, args.sdedit_strength_step)
        #     elif isinstance(sdedit_strength, float) and 0.0 <= sdedit_strength <= 1.0:
        #         all_strenths = [sdedit_strength]
        #     else:
        #         raise ValueError(f"Invalid sdedit_strength: {sdedit_strength}")
        # else:
        #     all_strenths = [sdedit_strength]
        all_strenths = [0.26]

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

        prefix_frames_z = frames_z[:, :prefix_num_latent_frames].detach().clone()

        for strength in all_strenths:
            cur_sdedit_strength = strength
            cur_sdedit_strength = round(cur_sdedit_strength, 2)
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
                prefix_clean_frames=prefix_frames_z,
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
            loop_num = T // frame_batch_size  # 2 is vae decoding frame batch size
            remaining_frames = T % frame_batch_size

            print(f"loop_num: {loop_num}, remaining_frames: {remaining_frames}")

            # drop the last batch frame, as it is the same as the first frame in the next batch
            # this is important, as we keep the context cache in vae, if we decode, the context cache missmatch

            for i in trange(loop_num, desc="Decoding"):
                start_frame = frame_batch_size * i + (0 if i == 0 else remaining_frames)
                end_frame = frame_batch_size * (i + 1) + remaining_frames

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

            basename = f"output_sfi{start_idx:03d}_nf{num_frames}_strength{cur_sdedit_strength}"
            basename = basename.replace(".", "d").replace("-", "n")
            output_video_path = os.path.join(cogvx_output_full_dir, f"{basename}.mp4")
            output_frames_path = os.path.join(cogvx_output_full_dir, f"{basename}_frames")
            os.makedirs(output_frames_path, exist_ok=True)
            save_frames(samples.squeeze(0), output_frames_path)
            save_video(samples, output_video_path, fps=out_fps)
            print(f"Saved video to {output_video_path}")
            print(f"Saved frames to {output_frames_path}")


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