import json
import os
import time

from multiprocessing import Pool
import torch


def compute_difference(x):
    diff = []

    for i, xx in enumerate(x):
        temp = []
        for j, xxx in enumerate(x):
            if i != j:
                temp.append(xx - xxx)

        diff.append(temp)

    return diff


def gen(entry_list):
    for i, entry in enumerate(entry_list):
        for instance in entry['instances']:
            vid = instance['video_id']

            frame_start = instance['frame_start']
            frame_end = instance['frame_end']

            save_to = os.path.join('/home/dxli/workspace/nslt/code/Pose-GCN/posegcn/features', vid)

            if not os.path.exists(save_to):
                os.mkdir(save_to)

            for frame_id in range(frame_start, frame_end + 1):
                frame_id = 'image_{}'.format(str(frame_id).zfill(5))

                ft_path = os.path.join(save_to, frame_id + '_ft.pt')
                if not os.path.exists(ft_path):
                    try:
                        pose_content = json.load(open(os.path.join('/home/dxli/workspace/nslt/data/pose/pose_per_individual_videos',
                                                                   vid, frame_id + '_keypoints.json')))["people"][0]
                    except IndexError:
                        continue

                    body_pose = pose_content["pose_keypoints_2d"]
                    left_hand_pose = pose_content["hand_left_keypoints_2d"]
                    right_hand_pose = pose_content["hand_right_keypoints_2d"]

                    body_pose.extend(left_hand_pose)
                    body_pose.extend(right_hand_pose)

                    x = [v for i, v in enumerate(body_pose) if i % 3 == 0 and i // 3 not in body_pose_exclude]
                    y = [v for i, v in enumerate(body_pose) if i % 3 == 1 and i // 3 not in body_pose_exclude]
                    # conf = [v for i, v in enumerate(body_pose) if i % 3 == 2 and i // 3 not in body_pose_exclude]

                    x = 2 * ((torch.FloatTensor(x) / 256.0) - 0.5)
                    y = 2 * ((torch.FloatTensor(y) / 256.0) - 0.5)
                    # conf = torch.FloatTensor(conf)

                    x_diff = torch.FloatTensor(compute_difference(x)) / 2
                    y_diff = torch.FloatTensor(compute_difference(y)) / 2

                    zero_indices = (x_diff == 0).nonzero()
                    orient = y_diff / x_diff
                    orient[zero_indices] = 0

                    xy = torch.stack([x, y]).transpose_(0, 1)
                    ft = torch.cat([xy, x_diff, y_diff, orient], dim=1)

                    torch.save(ft, ft_path)

        print('Finish {}-th entry'.format(i))


body_pose_exclude = {9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21}
index_file_path = '/home/dxli/workspace/nslt/data/splits-with-dialect-annotated/asl2000.json'

with open(index_file_path, 'r') as f:
    content = json.load(f)

# create label encoder

start_time = time.time()

entries_1 = content[0: 700]
entries_2 = content[700: 1400]
entries_3 = content[1400: ]

entry_splits = [entries_1, entries_2, entries_3]

p = Pool(3)
print(p.map(gen, entry_splits))

