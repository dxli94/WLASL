# preprocessing script for WLASL dataset
# 1. Convert .swf, .mkv file to mp4.
# 2. Extract YouTube frames and create video instances.

import os
import json
import cv2

import shutil

def convert_everything_to_mp4():
    cmd = 'bash scripts/swf2mp4.sh'

    os.system(cmd)


def video_to_frames(video_path, size=None):
    """
    video_path -> str, path to video.
    size -> (int, int), width, height.
    """

    cap = cv2.VideoCapture(video_path)

    frames = []
    
    while True:
        ret, frame = cap.read()
    
        if ret:
            if size:
                frame = cv2.resize(frame, size)
            frames.append(frame)
        else:
            break

    cap.release()

    return frames


def convert_frames_to_video(frame_array, path_out, size, fps=25):
    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def extract_frame_as_video(src_video_path, start_frame, end_frame):
    frames = video_to_frames(src_video_path)

    return frames[start_frame: end_frame+1]


def extract_all_yt_instances(content):
    cnt = 1

    if not os.path.exists('videos'):
        os.mkdir('videos')

    for entry in content:
        instances = entry['instances']

        for inst in instances:
            url = inst['url']
            video_id = inst['video_id']

            if 'youtube' in url or 'youtu.be' in url:
                cnt += 1
                
                yt_identifier = url[-11:]

                src_video_path = os.path.join('raw_videos_mp4', yt_identifier + '.mp4')
                dst_video_path = os.path.join('videos', video_id + '.mp4')

                if not os.path.exists(src_video_path):
                    continue

                if os.path.exists(dst_video_path):
                    print('{} exists.'.format(dst_video_path))
                    continue

                # because the JSON file indexes from 1.
                start_frame = inst['frame_start'] - 1
                end_frame = inst['frame_end'] - 1

                if end_frame <= 0:
                    shutil.copyfile(src_video_path, dst_video_path)
                    continue

                selected_frames = extract_frame_as_video(src_video_path, start_frame, end_frame)
                
                # when OpenCV reads an image, it returns size in (h, w, c)
                # when OpenCV creates a writer, it requres size in (w, h).
                size = selected_frames[0].shape[:2][::-1]
                
                convert_frames_to_video(selected_frames, dst_video_path, size)

                print(cnt, dst_video_path)
            else:
                cnt += 1

                src_video_path = os.path.join('raw_videos_mp4', video_id + '.mp4')
                dst_video_path = os.path.join('videos', video_id + '.mp4')

                if os.path.exists(dst_video_path):
                    print('{} exists.'.format(dst_video_path))
                    continue

                if not os.path.exists(src_video_path):
                    continue

                print(cnt, dst_video_path)
                shutil.copyfile(src_video_path, dst_video_path)

        
def main():
    # 1. Convert .swf, .mkv file to mp4.
    convert_everything_to_mp4()

    content = json.load(open('WLASL_v0.3.json'))
    extract_all_yt_instances(content)


if __name__ == "__main__":
    main()

