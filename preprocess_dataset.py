import argparse
import os

from glob import glob
from tqdm import tqdm
import cv2
from PIL import Image


def parse_args():
    """ Parse the arguments."""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_path', help='Path to Nutrition5k dataset.', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    camera_files = ['camera_A.h264', 'camera_B.h264', 'camera_C.h264', 'camera_D.h264']
    videos_path = os.path.join(args.dataset_path, 'imagery', 'side_angles')
    video_directories = glob(videos_path + '/*/')
    for directory in tqdm(video_directories):
        for camera_file in camera_files:
            file_path = os.path.join(directory, camera_file)
            frame_dir_path = os.path.join(directory, camera_file.split('.')[0])
            if os.path.isdir(frame_dir_path):
                continue
            cap = cv2.VideoCapture(file_path)
            os.makedirs(frame_dir_path, exist_ok=True)
            frame = None
            count = 1
            while cap.isOpened():
                ret, frame = cap.read()
                if frame is None:
                    break
                else:
                    #frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 0)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                    image.save(os.path.join(frame_dir_path, str(count)) + '.jpg')
                    count += 1
