import sys
import torch
import torch.nn as nn
from diffsynth import ModelManager, WanVideoMultimodalPipeline, save_video, VideoData
from diffsynth.models.wan_video_dit import MLP, RMSNorm
from model import CAM_Encoder_frame_global_new as CAM_Encoder
import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import pandas as pd
import torchvision
from PIL import Image
import numpy as np
import json
import decord
import torch.nn.functional as F
from tqdm import tqdm
decord.bridge.set_bridge('torch')  # PyTorch tensor로 직접 반환

class CamCloneDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, 
                 dit_max_num_frames=81, dit_frame_interval=1, dit_num_frames=81, dit_height=384, dit_width=672):
        metadata = pd.read_csv(base_path)
        self.text = metadata["caption"].to_list()
        self.ref_path = metadata["ref_video_path"].to_list()
        
        if "video_path" in metadata.columns:
            self.path = metadata["video_path"].to_list()
        if "content_video_path" in metadata.columns:
            self.content_path = metadata["content_video_path"].to_list()
        if "first_frame_path" in metadata.columns:  # first frame for infer
            self.first_frame_path = metadata["first_frame_path"].to_list()
        else:
            self.first_frame_path = None
        
        # DiT pipeline용 설정 (content_video)
        self.dit_max_num_frames = dit_max_num_frames
        self.dit_frame_interval = dit_frame_interval
        self.dit_num_frames = dit_num_frames
        self.dit_height = dit_height
        self.dit_width = dit_width
        
            
        # CAM encoder용 frame process
        self.mean_latent = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std_latent = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        # DiT pipeline용 frame process
        self.dit_frame_process = v2.Compose([
            v2.CenterCrop(size=(dit_height, dit_width)),
            v2.Resize(size=(dit_height, dit_width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        # First frame용 image process (DiT pipeline용 크기 사용)
        self.image_process = v2.Compose([
            v2.CenterCrop(size=(dit_height, dit_width)),
        ])

    def crop_and_resize(self, image, target_width, target_height):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process, image_process, target_width, target_height):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            frame_indexs = list(range(num_frames))
            frame_indexs = [min(frame_index, reader.count_frames()-1) for frame_index in frame_indexs]
        else:
            frame_indexs = list(range(num_frames))
        
        frames = []
        first_frame = None
        for frame_id in frame_indexs:
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, target_width, target_height)
            if first_frame is None:
                first_frame = image_process(frame)  # 输入必须是PIL
                first_frame = np.array(first_frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")
        return frames, first_frame

    
    def load_video_for_dit(self, file_path):
        """DiT pipeline용 비디오 로드 (content_video)"""
        start_frame_id = 0
        frames, first_frame = self.load_frames_using_imageio(
            file_path, self.dit_max_num_frames, start_frame_id, 
            self.dit_frame_interval, self.dit_num_frames, 
            self.dit_frame_process, self.image_process,
            self.dit_width, self.dit_height
        )
        return frames, first_frame
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame, self.dit_width, self.dit_height)
        frame = self.image_process(frame)
        frame = np.array(frame)
        return frame


    def process_frames(self, frames, height, width, mean, std):
        """프레임 리사이즈, 크롭, 정규화"""
        # decord.bridge.set_bridge('torch')를 설정했으므로 이미 torch tensor
        T, H, W, C = frames.shape
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)
        
        orig_h, orig_w = frames[0].shape[1], frames[0].shape[2]
        scale = max(width / orig_w, height / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        
        processed_frames = []
        first_frame_array = None
        
        for i in range(T):
            frame = frames[i]  # (C, H, W)
            
            # 리사이즈
            frame = torch.nn.functional.interpolate(
                frame.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False
            ).squeeze(0)
            
            # 크롭
            start_y, start_x = (new_h - height) // 2, (new_w - width) // 2
            frame_cropped = frame[:, start_y:start_y+height, start_x:start_x+width]
            
            if i == 0:
                if frame_cropped.max() <= 1.0:
                    first_frame_array = (frame_cropped * 255.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                else:
                    first_frame_array = frame_cropped.clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            # 정규화
            if frame_cropped.max() > 1.0:
                frame_cropped = frame_cropped / 255.0
            frame_cropped = (frame_cropped - mean) / std
            processed_frames.append(frame_cropped)
        
        return torch.stack(processed_frames, dim=0), first_frame_array

    def load_video(self, file_path, interval, num_frames, height, width, mean, std):

        vr = decord.VideoReader(file_path, ctx=decord.cpu(0), num_threads=4)
        total_frames = len(vr)
        
        frame_indices = [min(idx * interval, total_frames - 1) for idx in range(num_frames)]
        frames = vr.get_batch(frame_indices)  # (T, H, W, C)
        
        frames, first_frame = self.process_frames(frames, height, width, mean, std)
        
        return frames, first_frame


    def __getitem__(self, data_id):
        text = self.text[data_id]
        ref_path = self.ref_path[data_id]
        # ref_video는 CAM encoder용으로 로드
        ref_video, first_frame = self.load_video(ref_path, 4, 21, 224, 448, self.mean_latent, self.std_latent)

        content_path, content_video = None, None
        first_frame = None
        if self.content_path is not None:
            content_path = self.content_path[data_id]
            content_video, first_frame = self.load_video_for_dit(content_path)
            target_video, _ = self.load_video_for_dit(ref_path)
        if self.first_frame_path is not None:
            first_frame_path = self.first_frame_path[data_id]
            first_frame = self.load_image(first_frame_path)
        data = {"text": text, "first_frame": first_frame, 'ref_video':ref_video, 'ref_path': ref_path, 'content_video': content_video, 'content_path': content_path, 'target_video': target_video}
           
        return data
    
    def __len__(self):
        return len(self.text)


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Latent Inference with CamClone Dataset")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="demo/example_csv/infer/example_camclone_testset.csv",
        help="The path of the CamClone Dataset CSV file.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoint/multimodal/camclone_new/lightning_logs/version_55637/checkpoints/step1400.ckpt",
        help="Path to the checkpoint.",
    )
    parser.add_argument(
        "--cam_encoder_ckpt_path",
        type=str,
        default="checkpoint/vggt_t5_pose_temporal/lightning_logs/version_53208/checkpoints/step111706.ckpt",
        help="Path to the CAM encoder checkpoint for encoding ref_video to reference_latent.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default="checkpoint/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default="checkpoint/Wan-AI/Wan2.1-T2V-1.3B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="checkpoint/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default="checkpoint/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        help="Path of DiT.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/two_modalities/video_text_interpolation_norm",
        help="Path to save the results.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    # DiT pipeline용 설정 (content_video)
    parser.add_argument(
        "--dit_num_frames",
        type=int,
        default=81,
        help="Number of frames for DiT pipeline (content_video).",
    )
    parser.add_argument(
        "--dit_height",
        type=int,
        default=384,
        help="Image height for DiT pipeline (content_video).",
    )
    parser.add_argument(
        "--dit_width",
        type=int,
        default=672,
        help="Image width for DiT pipeline (content_video).",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='v2v',
        choices=['i2v', 'v2v'],
        help="Mode of the pipeline.",
    )
    parser.add_argument(
        '--output_only',
        action='store_true',
        help="생성된 출력 비디오만 저장 (Source|Generated|Ref 합친 비디오 대신).",
    )



    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_path = [args.text_encoder_path, args.vae_path, args.dit_path]
    if args.image_encoder_path is not None:
        model_path.append(args.image_encoder_path)
    model_manager.load_models(model_path)
    
    pipe = WanVideoMultimodalPipeline.from_model_manager(model_manager)

    with torch.no_grad():
        patch_embedding_ori = pipe.dit.patch_embedding
        patch_embedding_new = nn.Conv3d(36, 1536, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        new_weights = torch.cat([
            patch_embedding_ori.weight,          # 第一轮复制 (16个通道)
            patch_embedding_ori.weight,          # 第二轮复制 (16个通道)
            patch_embedding_ori.weight[:, :4]    # 第三轮复制前4个通道
        ], dim=1) # 沿着输入通道维度(dim=1)拼接

        assert new_weights.shape[1] == 36
        patch_embedding_new.weight.copy_(new_weights)
        patch_embedding_new.bias.copy_(patch_embedding_ori.bias)

    pipe.dit.patch_embedding = patch_embedding_new

    dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
    pipe.dit.img_emb = MLP(1280, dim)
    for block in pipe.dit.blocks:  # add for I2V
        block.cross_attn.k_img = nn.Linear(dim, dim)
        block.cross_attn.v_img = nn.Linear(dim, dim)
        block.cross_attn.norm_k_img = RMSNorm(dim, eps=1e-6)

    for block in pipe.dit.blocks:
        block.cam_encoder = nn.Linear(768, dim)
        block.projector = nn.Linear(dim, dim)
        block.cam_encoder.weight.data.zero_()
        block.cam_encoder.bias.data.zero_()
        block.projector.weight = nn.Parameter(torch.eye(dim))
        block.projector.bias = nn.Parameter(torch.zeros(dim))
    
    pipe.denoising_model().has_image_input = True

    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    pipe.dit.load_state_dict(state_dict, strict=True)
    print(f"load ckpt from {args.ckpt_path}!")
    pipe.to(device)
    pipe.to(dtype=torch.bfloat16)

    assert args.cam_encoder_ckpt_path is not None, "CAM encoder checkpoint path is required."

    print(f"Loading CAM encoder from {args.cam_encoder_ckpt_path}")
    cam_encoder = CAM_Encoder()
    cam_state_dict = torch.load(args.cam_encoder_ckpt_path, map_location="cpu", weights_only=False)
    if 'state_dict' in cam_state_dict:
        cam_state_dict = cam_state_dict['state_dict']
    # Remove 'cam_encoder.' prefix from keys if present
    if any(key.startswith('cam_encoder.') for key in cam_state_dict.keys()):
        new_state_dict = {}
        for key, value in cam_state_dict.items():
            if key.startswith('cam_encoder.'):
                new_key = key[len('cam_encoder.'):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        cam_state_dict = new_state_dict
    
    cam_encoder.load_state_dict(cam_state_dict, strict=True)
    print(f"Loaded CAM encoder checkpoint from {args.cam_encoder_ckpt_path}!")
    cam_encoder.to(device)
    cam_encoder.eval()
    print(f"Loaded CAM encoder from {args.cam_encoder_ckpt_path}!")

    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    dataset = CamCloneDataset(
        args.dataset_path,
        dit_max_num_frames=args.dit_num_frames,
        dit_frame_interval=1,
        dit_num_frames=args.dit_num_frames,
        dit_height=args.dit_height,
        dit_width=args.dit_width,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )

    # 9. Inference
    pipe.eval()
    
    if pipe.device != device:
        pipe.device = device
    if pipe.torch_dtype != torch.bfloat16:
        pipe.torch_dtype = torch.bfloat16
    
    # Normalization 파라미터 (denormalize용 - CamCloneDataset용)
    mean_camclone = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1, 1).to("cuda")
    std_camclone = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1, 1).to("cuda")
    
    def denormalize_tensor(tensor):
        """Normalized tensor를 [0, 255] 범위의 PIL Image 리스트로 변환"""
        tensor = tensor.float()
        tensor_denorm = tensor * std_camclone + mean_camclone
        tensor_denorm = torch.clamp(tensor_denorm, 0, 1)
        tensor_denorm = rearrange(tensor_denorm, "C T H W -> T H W C")
        frames = (tensor_denorm * 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames
    
       
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx != 0:
                continue
            path = batch['content_path'][0].split('/')[-1].split('.')[0]
            ref_video = batch['ref_video'].to("cuda")  # (B, C, T, H, W) - CamCloneDataset normalization
            source_video = batch["content_video"].to("cuda") if batch["content_video"] is not None else None  # source video
            target_video = batch["target_video"].to("cuda") if batch["target_video"] is not None else None  # target video
            text = batch["text"]  # list of strings
            first_frames = batch["first_frame"]  # list of numpy arrays or None
            camera_caption = "The camera starts with a steady tilt down, maintaining consistent motion throughout the sequence."
            # camera_caption_list = ["Initially, the camera moves steadily to the left along the X-axis by 0.09 meters at each time interval, maintaining a consistent truck position throughout the sequence. Simultaneously, a constant pan to the right of 0.7 degrees is applied across all frames, resulting in a stable, uniform directional offset. There is no variation in dolly movement (Z-axis), with a minimal forward displacement of 0.17 meters occurring at the end, and no noticeable change in tilt or roll. The motion evolves with a steady pace, showing no acceleration or deceleration in any axis, and concludes with a total displacement",
            # "The camera starts with a steady left truck movement and upward pedestal motion, while slowly panning right and tilting down. At around 57%, it begins dollying in, continuing with a sustained left truck motion, upward pedestal movement, and increasing pan right, culminating in a stable, slightly tilted, and in-focused composition.",
            # "The camera starts with a steady dolly-in motion, then gradually shifts to a sustained truck-left movement while continuing to dolly in.",
            # "The camera starts stationary, then slowly moves up on the pedestal while gradually tilting down, maintaining a steady and smooth upward progression throughout.",
            # "The camera starts with a steady left truck movement, dollying in, and tilting down while panning right, then gradually transitions to a rightward truck motion, continuing to pan left while slowly reducing its upward pedestal movement.",
            # "The camera starts with a steady right truck movement while dollying in and tilting down, all while panning left. This motion continues smoothly until the end, where the camera gradually slows in all directions, ending with a slight upward shift in pedestal and minimal tilt.",
            # "The camera remains stationary throughout the entire sequence.",
            # "The camera starts with a smooth pan to the left, gradually slowing down, and then remains stationary.",
            # "The camera starts with a slow downward tilt, gradually reducing its angle, then remains stationary.",
            # "The camera starts with a slow pan left, then gradually transitions to a steady truck right while continuing to pan left and dolly in.",
            # "The camera starts with a steady right truck move and dolly in, while gradually panning left and tilting up. As the sequence progresses, the truck and dolly motion slow slightly, but the pan continues to increase, culminating in a smooth, sustained leftward pan with a steady upward tilt.",
            # "The camera starts with a steady Pedestal Down movement while gradually tilting up, then comes to a complete stop.",
            # "The camera starts with a steady rightward truck movement and dolly-out motion while slowly panning left, gradually slowing down in all directions, then becomes fully stationary.",
            # "The camera starts with a steady rightward truck movement and dolly-in motion, while panning slightly left, then gradually shifts to a leftward truck motion, continuing the dolly-in and downward tilt while maintaining a leftward pan.",
            # "The camera starts with a steady pan left while trucking right, then transitions to a slow dolly out and tilt down, followed by a gradual rise in pedestal movement while maintaining the dolly and tilt, ending with a smooth return to a more stable, elevated position.",
            # "The camera starts with a slow upward pedestal motion that gradually slows and peaks at 5%, then steadily descends while maintaining a steady upward trend until the end, ending with a return to stillness.",
            # ]
            camera_caption_list = ["The camera tilts up very fast keep the motion."]
            reference_latent_texts = []

            for camera_caption in camera_caption_list:
                encoder_outputs = cam_encoder(videos=None, texts=camera_caption, poses=None)
                reference_latent_texts.append(encoder_outputs['text_frame_embeds'])
            encoder_outputs = cam_encoder(videos=ref_video, texts=None, poses=None)
            reference_latent_video = encoder_outputs['video_frame_embeds']  # (B, output_dim)


            ref_video = ref_video.to(dtype=pipe.torch_dtype, device=pipe.device)
            if source_video is not None:
                source_video = source_video.to(dtype=pipe.torch_dtype, device=pipe.device)
            reference_latent_video = reference_latent_video.to(dtype=pipe.torch_dtype, device=pipe.device)
            
            # DiT pipeline용 크기 (source_video 또는 ref_video 중 하나가 있을 때)
            if source_video is not None:
                _, _, num_frames, height, width = source_video.shape
            else:
                # source_video가 None인 경우 ref_video 크기 사용 (하지만 이 경우는 거의 없을 것)
                _, _, num_frames, height, width = ref_video.shape
            first_frames_pil = [Image.fromarray(frame.cpu().numpy()) for frame in first_frames]
            front = reference_latent_video[:, ::2]   # (B, T1, 768)

            for index, reference_latent_text in enumerate(tqdm(reference_latent_texts)):
                reference_latent_text = reference_latent_text.to(dtype=pipe.torch_dtype, device=pipe.device)
            # Interpolation
                scales = [0, 10, 20]
                for scl in scales:
                    scale = scl / 20
                    target_camera = torch.nn.functional.normalize((scale * reference_latent_video + (1-scale) * reference_latent_text),dim=-1)# (B, 21, 768)
                

                    # Inference using pipeline
                    generated_video = pipe(
                        prompt=text[0] if isinstance(text, list) else text,
                        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                        source_video=source_video,
                        target_camera=target_camera,
                        input_image=first_frames_pil if len(first_frames_pil) > 0 else None,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        cfg_scale=args.cfg_scale,
                        num_inference_steps=50,
                        seed=0,
                        mode=args.mode,
                    )
                    
                    # Source video 프레임 변환 (content video)
                    source_video_frames = denormalize_tensor(source_video[0])  # batch에서 첫 번째만
                    target_video_frames = denormalize_tensor(target_video[0])  # batch에서 첫 번째만
                    
                    # Generated video를 PIL Image 리스트로 변환
                    if isinstance(generated_video, list):
                        generated_frames = generated_video
                    elif isinstance(generated_video, np.ndarray):
                        # numpy array인 경우 PIL Image로 변환
                        if generated_video.dtype != np.uint8:
                            generated_video = (np.clip(generated_video, 0, 1) * 255).astype(np.uint8)
                        generated_frames = [Image.fromarray(frame) for frame in generated_video]
                    else:
                        # VideoData나 다른 형식인 경우
                        generated_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame 
                                        for frame in generated_video]
                    
                    # 프레임 수 맞추기 (가장 짧은 비디오 기준)
                    video_list = [f for f in [generated_frames, target_video_frames, source_video_frames] if len(f) > 0]
                    if len(video_list) == 0:
                        continue
                    min_frames = min(len(frames) for frames in video_list)
                    generated_frames = generated_frames[:min_frames] if len(generated_frames) > 0 else []
                    target_video_frames = target_video_frames[:min_frames] if len(target_video_frames) > 0 else []
                    source_video_frames = source_video_frames[:min_frames] if len(source_video_frames) > 0 else []
                    
                    # # 각 프레임의 크기 확인 및 리사이즈
                    # gen_frame = generated_frames[0]
                    # tgt_frame = target_video_frames[0]
                    # source_frame = source_video_frames[0] if len(source_video_frames) > 0 else tgt_frame
                    
                    # # 모든 프레임을 같은 높이로 맞춤
                    # target_height = min(gen_frame.height, tgt_frame.height, source_frame.height)
                        
                    #     # 3개 비디오를 가로로 연결한 프레임 생성
                    # combined_frames = []
                    # for i in range(min_frames):
                    #     gen_img = generated_frames[i].resize((gen_frame.width * target_height // gen_frame.height, target_height), Image.LANCZOS)
                    #     tgt_img = target_video_frames[i].resize((tgt_frame.width * target_height // tgt_frame.height, target_height), Image.LANCZOS)
                    #     if args.mode == 'i2v':
                    #         src_img = source_video_frames[0].resize((source_frame.width * target_height // source_frame.height, target_height), Image.LANCZOS)
                    #     else:
                    #         src_img = source_video_frames[i].resize((source_frame.width * target_height // source_frame.height, target_height), Image.LANCZOS)
                    #     # 가로로 연결 (Source | Generated | Ref)
                    #     combined_width = gen_img.width + tgt_img.width + src_img.width
                    #     combined_img = Image.new('RGB', (combined_width, target_height))
                    #     combined_img.paste(src_img, (0, 0))  # Source (왼쪽)
                    #     combined_img.paste(gen_img, (src_img.width, 0))  # Generated (중간)
                    #     combined_img.paste(tgt_img, (src_img.width + gen_img.width, 0))  # Ref (오른쪽)

                    #     combined_frames.append(combined_img)
                    
                    # output_only: 생성된 비디오만 저장 / 아니면 Source|Generated|Ref 합친 비디오 저장
                    out_path_generated = os.path.join(output_dir, f"video{batch_idx}_{scl}_{index}.mp4")
                    out_path_source = os.path.join(output_dir, f"video{batch_idx}_source.mp4")
                    out_path_target = os.path.join(output_dir, f"video{batch_idx}_target.mp4")
                    save_video(source_video_frames, out_path_source, fps=24, quality=5)
                    print(f"Saved source video to {out_path_source}")
                    save_video(target_video_frames, out_path_target, fps=24, quality=5)
                    print(f"Saved target video to {out_path_target}")
                    save_video(generated_frames, out_path_generated, fps=24, quality=5)
                    print(f"Saved output video to {out_path_generated}")
