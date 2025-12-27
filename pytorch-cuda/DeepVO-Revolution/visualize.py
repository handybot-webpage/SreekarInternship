import matplotlib.pyplot as plt
import numpy as np

predicted_result_dir = './result/'
gradient_color = True

def plot_route(out, c_out='r'):
    x_idx = 3
    y_idx = 5
    x = out[:, x_idx]
    y = out[:, y_idx]
    plt.plot(x, y, color=c_out, label='DeepVO')
    plt.gca().set_aspect('equal', adjustable='datalim')

video_list = ['11']

for video in video_list:
    print('=' * 50)
    print(f'Video {video}')

    pose_result_path = f'{predicted_result_dir}out_{video}.txt'
    with open(pose_result_path) as f:
        out = np.array([[float(v) for v in line.strip().split(',')] for line in f])

    if gradient_color:
        step = 200
        plt.clf()
        plt.scatter(out[0][3], out[0][5], label='sequence start', marker='s', color='k')

        for st in range(0, len(out), step):
            end = st + step
            g = max(0.2, st / len(out))
            c_out = (1, g, 0)
            plot_route(out[st:end], c_out)
            if st == 0:
                plt.legend()
            plt.title(f'Video {video}')

        save_name = f'{predicted_result_dir}route_{video}_gradient.png'
        plt.savefig(save_name)

    else:
        plt.clf()
        plt.scatter(out[0][3], out[0][5], label='sequence start', marker='s', color='k')
        plot_route(out, 'r')
        plt.legend()
        plt.title(f'Video {video}')
        save_name = f'{predicted_result_dir}route_{video}.png'
        plt.savefig(save_name)
