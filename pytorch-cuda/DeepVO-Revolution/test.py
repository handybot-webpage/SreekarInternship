# inference-only test.py (no ground truth required)
# Put this file at /workspace/test.py (overwrite the existing one), then run: python test.py

from params import par
from model import DeepVO
import numpy as np
from PIL import Image
import glob
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from helper import eulerAnglesToRotationMatrix


class FrameSequenceDataset(Dataset):
    def __init__(self, frame_paths, seq_len, resize_mode, new_size, img_mean, img_std, minus_point_5):
        self.frame_paths = list(frame_paths)
        self.seq_len = int(seq_len)

        transform_ops = []
        if resize_mode == 'crop':
            transform_ops.append(transforms.CenterCrop((new_size[0], new_size[1])))
        elif resize_mode == 'rescale':
            transform_ops.append(transforms.Resize((new_size[0], new_size[1])))
        transform_ops.append(transforms.ToTensor())
        self.transformer = transforms.Compose(transform_ops)

        self.normalizer = transforms.Normalize(mean=img_mean, std=img_std)
        self.minus_point_5 = minus_point_5

        # build sliding windows: jump=1 (overlap=seq_len-1)
        n = len(self.frame_paths)
        if n < self.seq_len:
            self.windows = []
        else:
            self.windows = [self.frame_paths[i:i + self.seq_len] for i in range(0, n - self.seq_len + 1, 1)]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        paths = self.windows[idx]
        seq = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            x = self.transformer(img)
            if self.minus_point_5:
                x = x - 0.5
            x = self.normalizer(x)
            seq.append(x.unsqueeze(0))
        x_seq = torch.cat(seq, dim=0)  # (T, C, H, W)
        seq_len_tensor = torch.tensor(self.seq_len, dtype=torch.int64)
        dummy_y = torch.zeros((self.seq_len, 6), dtype=torch.float32)  # unused
        return seq_len_tensor, x_seq, dummy_y


def run_one_video(video_id: str, model: DeepVO, device: torch.device):
    # This script expects your frames at: par.image_dir + video_id + "/*.png"
    # With your current layout, set par.image_dir = "KITTI/images/" in params.py
    frame_glob = f"{par.image_dir}{video_id}/*.png"
    frame_paths = sorted(glob.glob(frame_glob))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found at: {frame_glob}")

    seq_len = int((par.seq_len[0] + par.seq_len[1]) / 2)
    overlap = seq_len - 1
    print(f"Video {video_id}: frames={len(frame_paths)}  seq_len={seq_len}  overlap={overlap}")

    dataset = FrameSequenceDataset(
        frame_paths=frame_paths,
        seq_len=seq_len,
        resize_mode=par.resize_mode,
        new_size=(par.img_w, par.img_h),
        img_mean=par.img_means,
        img_std=par.img_stds,
        minus_point_5=par.minus_point_5,
    )

    if len(dataset) == 0:
        raise RuntimeError(f"Not enough frames ({len(frame_paths)}) for seq_len={seq_len}")

    dataloader = DataLoader(dataset, batch_size=par.batch_size, shuffle=False, num_workers=1)

    model.eval()

    answer = [[0.0] * 6]  # absolute pose list, seed at origin
    st_t = time.time()
    n_batch = len(dataloader)

    with torch.no_grad():
        for bi, batch in enumerate(dataloader):
            print(f"{bi+1} / {n_batch}", end="\r", flush=True)

            _, x, _ = batch  # x: (B, T, C, H, W)
            x = x.to(device)

            batch_pred = model.forward(x)  # (B, T, 6) relative-ish outputs
            batch_pred = batch_pred.detach().cpu().numpy()

            # For first batch, append all poses from the first sequence to bootstrap
            if bi == 0:
                first_seq = batch_pred[0]
                for pose in first_seq:
                    pose = pose.copy()
                    for k in range(6):
                        pose[k] += answer[-1][k]
                    answer.append(pose.tolist())
                batch_pred = batch_pred[1:]  # remaining sequences only contribute last pose

            # For subsequent sequences, use only last pose in each predicted sequence
            for pred_seq in batch_pred:
                pred_seq = pred_seq.copy()

                # Rotate translation into current frame (repo logic)
                ang = eulerAnglesToRotationMatrix([0, answer[-1][0], 0])
                loc = ang.dot(pred_seq[-1][3:])
                pred_seq[-1][3:] = loc[:]

                last_pose = pred_seq[-1]
                for k in range(6):
                    last_pose[k] += answer[-1][k]
                last_pose[0] = (last_pose[0] + np.pi) % (2 * np.pi) - np.pi
                answer.append(last_pose.tolist())

    print()
    print("len(answer):", len(answer))
    print("frames:", len(frame_paths))
    print("Predict took {:.2f} sec".format(time.time() - st_t))

    os.makedirs("result", exist_ok=True)
    out_path = f"result/out_{video_id}.txt"
    with open(out_path, "w") as f:
        for pose in answer:
            f.write(", ".join(map(str, pose)) + "\n")

    print("Wrote:", out_path)


if __name__ == "__main__":
    videos_to_test = ["11"]  # change to your folder name(s) under KITTI/images/

    # Load model
    load_model_path = par.load_model_path
    model = DeepVO(par.img_h, par.img_w, par.batch_norm)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # NOTE: this warning is fine; it’s about PyTorch security defaults.
    state = torch.load(load_model_path, map_location=device)
    model.load_state_dict(state)
    print("Load model from:", load_model_path)
    print("Device:", device)

    for vid in videos_to_test:
        run_one_video(vid, model, device)
        print("=" * 50)
