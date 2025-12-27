import time
from collections import deque

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import socket, json

from params import par
from model import DeepVO
from helper import eulerAnglesToRotationMatrix

CAM_INDEX = 0
PLOT_EVERY = 1
PRINT_EVERY = 10  # print rate every N pose updates
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# frame debug
SHOW_FRAME = True
FRAME_PRINT_EVERY = 10      # print frame stats every N frames
SAVE_FRAME_EVERY_SEC = 2.0  # write debug_frame.png every N seconds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepVO(par.img_h, par.img_w, par.batch_norm).to(device)
state = torch.load(par.load_model_path, map_location=device)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
model.load_state_dict(state)
model.eval()

seq_len = int((par.seq_len[0] + par.seq_len[1]) / 2)
print(seq_len)

ops = []
if par.resize_mode == "crop":
    ops.append(transforms.CenterCrop((par.img_h, par.img_w)))
elif par.resize_mode == "rescale":
    ops.append(transforms.Resize((par.img_h, par.img_w)))
ops.append(transforms.ToTensor())
to_tensor = transforms.Compose(ops)
normalizer = transforms.Normalize(mean=par.img_means, std=par.img_stds)

def prep_frame(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = to_tensor(pil)
    if par.minus_point_5:
        x = x - 0.5
    x = normalizer(x)
    return x

def send_pose(pose6):
    msg = json.dumps({
        "rx": float(pose6[0]),
        "ry": float(pose6[1]),
        "rz": float(pose6[2]),
        "tx": float(pose6[3]),
        "ty": float(pose6[4]),
        "tz": float(pose6[5]),
        "t": time.time()
    }).encode("utf-8")
    sock.sendto(msg, (UDP_IP, UDP_PORT))

buf = deque(maxlen=seq_len)
answer = [[0.0] * 6]

plt.ion()
fig = plt.figure()

ax1 = fig.add_subplot(131)
ax1.set_aspect("equal", adjustable="datalim")
line_xz, = ax1.plot([], [], lw=2)
ax1.set_title("DeepVO live trajectory (tx vs tz)")
ax1.set_xlabel("tx")
ax1.set_ylabel("tz")

ax2 = fig.add_subplot(132)
ax2.set_aspect("equal", adjustable="datalim")
line_xy, = ax2.plot([], [], lw=2)
ax2.set_title("DeepVO live trajectory (tx vs ty)")
ax2.set_xlabel("tx")
ax2.set_ylabel("ty")

ax3 = fig.add_subplot(133)
ax3.set_aspect("equal", adjustable="datalim")
line_yz, = ax3.plot([], [], lw=2)
ax3.set_title("DeepVO live trajectory (ty vs tz)")
ax3.set_xlabel("ty")
ax3.set_ylabel("tz")

def update_plot():
    poses = np.array(answer, dtype=np.float32)
    tx = poses[:, 3]
    ty = poses[:, 4]
    tz = poses[:, 5]

    line_xz.set_data(tx, tz)
    line_xy.set_data(tx, ty)
    line_yz.set_data(ty, tz)

    for a in (ax1, ax2, ax3):
        a.relim()
        a.autoscale_view()

    fig.canvas.draw()
    fig.canvas.flush_events()

def show_model_input(x_chw, win="MODEL_INPUT_FRAME"):
    # x_chw: torch tensor (C,H,W) AFTER all preprocessing
    x0 = x_chw.detach().cpu()

    if par.minus_point_5:
        x0 = x0 + 0.5

    mean = torch.tensor(par.img_means, dtype=x0.dtype).view(3, 1, 1)
    std = torch.tensor(par.img_stds, dtype=x0.dtype).view(3, 1, 1)
    x0 = x0 * std + mean  # undo Normalize

    x0 = x0.clamp(0, 1)
    img = (x0.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # RGB
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow(win, img_bgr)

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Could not open camera device. Change CAM_INDEX.")

count = 0
frame_count = 0

last_pose = np.array(answer[-1], dtype=np.float32)
last_t = time.time()

last_frame_print = time.time()
last_frame_save = 0.0

with torch.no_grad():
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        frame_count += 1

        # ---- show + print frame debug ----
        if SHOW_FRAME:
            cv2.imshow("CAPTURED_FRAME", frame)

        if frame_count % FRAME_PRINT_EVERY == 0:
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)
            sig = int(np.sum(small))
            print(f"[FRAME] n={frame_count} shape={h}x{w} mean={frame.mean():.1f} std={frame.std():.1f} sig={sig}")

        now = time.time()
        if now - last_frame_save >= SAVE_FRAME_EVERY_SEC:
            cv2.imwrite("debug_frame.png", frame)
            last_frame_save = now

        # quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # ---- DeepVO inference ----
        x = prep_frame(frame).unsqueeze(0)
        show_model_input(x[0], "MODEL_INPUT_FRAME")
        buf.append(x)

        if len(buf) < seq_len:
            continue

        seq = torch.cat(list(buf), dim=0).unsqueeze(0).to(device)
        pred_seq = model.forward(seq).detach().cpu().numpy()[0]
        rel = pred_seq[-1].copy()

        ang = eulerAnglesToRotationMatrix([0, answer[-1][0], 0])
        rel[3:] = ang.dot(rel[3:])

        abs_pose = rel.copy()
        for k in range(6):
            abs_pose[k] += answer[-1][k]
        abs_pose[0] = (abs_pose[0] + np.pi) % (2 * np.pi) - np.pi

        answer.append(abs_pose.tolist())
        send_pose(answer[-1])
        count += 1

        # ---- rate reporting (pose updates) ----
        if count % PRINT_EVERY == 0:
            now = time.time()
            dt = now - last_t
            cur_pose = np.array(answer[-1], dtype=np.float32)

            dxyz = cur_pose[3:6] - last_pose[3:6]
            dist = float(np.linalg.norm(dxyz))
            speed = dist / dt if dt > 0 else 0.0

            drot = cur_pose[0:3] - last_pose[0:3]
            rot_mag = float(np.linalg.norm(drot))
            rot_rate = rot_mag / dt if dt > 0 else 0.0

            print(
                f"[POSE] updates={count} dt={dt:.3f}s "
                f"dpos={dist:.6f} speed={speed:.6f} units/s "
                f"drot={rot_mag:.6f} rad rot_rate={rot_rate:.6f} rad/s"
            )

            last_pose = cur_pose
            last_t = now

        if count % PLOT_EVERY == 0:
            update_plot()

cap.release()
cv2.destroyAllWindows()
