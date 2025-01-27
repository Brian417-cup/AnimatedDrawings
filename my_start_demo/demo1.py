from animated_drawings import render
import os.path as osp

if __name__ == '__main__':
    example_yaml_path = osp.join('../examples/config/mvc/export_mp4_example.yaml')
    render.start(example_yaml_path)