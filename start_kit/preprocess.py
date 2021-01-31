# preprocessing script for WLASL dataset
# 1. Convert .swf, .mkv file to mp4.
# 2. Extract YouTube frames and create video instances.

import os
import sys
import glob
import json
import cv2
import shutil
import re

import logging
logging.basicConfig(
    filename="preProc.log",
    filemode='w',
    level=logging.DEBUG
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def convert_frames_to_video(frame_array, path_out, size, fps=25):
    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def video_to_frames(video_path, size=None):
    """
    video_path -> str, path to video.
    size -> (int, int), width, height.
    """

    print(f"video_path: {video_path} size: {size}")
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


def extract_frame_as_video(src_video_path, start_frame, end_frame):
    frames = video_to_frames(src_video_path)

    return frames[start_frame: end_frame+1]


class Preproc:
    def __init__(self,
                 idxf="WLASL_v0.3.json",
                 videoDir="data"):
        self.indexFile = idxf
        self.vd = videoDir

    def convertTomp4(self):
        for f in os.scandir(self.vd):
            if (
                not f.path.endswith(".mp4") and
                not glob.glob(
                    os.path.join(
                        self.vd,
                        os.path.splitext(f.name)[0]) + '.mp4'
                )
            ):
                dest = os.path.join(self.vd,
                                    os.path.splitext(f.name)[0] + '.mp4'
                                    )
                if (
                    os.system(
                        f"ffmpeg -loglevel panic -i {f.path} -vf "
                        f"pad=\"width=ceil(iw/2)*2\" {dest}"
                    ) == 0
                ):
                    logging.info(f"Conversion Successful\t-\t{f.name}")
                else:
                    logging.error(f"Conversion Failed\t\t-\t{f.name}")
            elif f.path.endswith(".swf"):
                logging.info(f"{f.name} already converted - Skipping")

    def extractVideo(self):
        idx = json.load(open(self.indexFile))

        for i in idx:
            for j in i["instances"]:
                if re.search(r"youtu\.?be", j["url"]):
                    src = os.path.join(
                        self.vd, j["video_id"] + '.yt.mp4'
                    )
                    dst = os.path.join(
                        self.vd, j["video_id"] + '.mp4'
                    )
                    if not os.path.exists(src):
                        continue
                    if os.path.exists(dst):
                        logging.info(f"{src} already extracted - Skipping ")
                        continue

                    if j["frame_end"] - 1 <= 0:
                        shutil.copyfile(src, dst)
                        continue

                    print(f"src: {src}")
                    selected_frames = extract_frame_as_video(
                        src,
                        j["frame_start"] - 1,
                        j["frame_end"] - 1
                    )

                    size = selected_frames[0].shape[:2][::-1]
                    convert_frames_to_video(selected_frames, dst, size)

    def main(self):
        # logging.info(">>>Converting files to mp4")
        # self.convertTomp4()
        # logging.info(">>>Extracting youtube videos")
        # self.extractVideo()
        for r, d, f in os.walk(self.vd):
            print(r)
            print(d)
            print(f)
            print("==============")


if __name__ == "__main__":
    preproc = Preproc()
    preproc.main()
