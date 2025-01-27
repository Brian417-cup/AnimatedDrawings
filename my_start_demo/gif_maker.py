import argparse
import os
import cv2
import imageio


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_path', type=str, default='demo.mp4', help='src video path')
    args = parser.parse_args()
    return args


def convert_video2gif(src_video_path: str):
    video_reader = cv2.VideoCapture(src_video_path)
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))
    frame_list = []
    print('Start Convert Gif!!')
    while True:
        ret, frame = video_reader.read()
        if ret is False:
            break

        # bgr -> rgb
        frame_list.append(frame[..., ::-1])

    output_save_path = os.path.join(os.path.dirname(src_video_path),
                                    os.path.basename(src_video_path).split('.')[0] + '.gif')

    imageio.mimsave(output_save_path, frame_list, 'GIF', duration=1 / fps * 1000, loop=0)
    print('Gif Has Convert Done!!')


if __name__ == '__main__':
    args = get_args()
    convert_video2gif(args.vid_path)
