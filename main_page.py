import os.path as osp
import sys

sys.path.append(osp.abspath(osp.dirname(__file__)))

import gradio as gr
from pages.top_bar import top_bar
from pages.upload_image import upload_image
from pages.show_joint_and_triangle import show_drawing_result
from pages.upload_motion import upload_motion
from pages.animation_generation import animation_generation
from util import global_state

with gr.Blocks(css="css/background.css") as app:
    # with gr.Blocks(css=".gradio-container { background-data: url('http://localhost:8000/static/bg.png');}") as app:
    # 标题
    top_bar()

    # 内容区
    # 第一步
    with gr.Tab("第一步") as step1:
        upload_image()

    # 第二步
    with gr.Tab("第二步") as step2:
        show_drawing_result()

    # 第三步
    with gr.Tab("第三步") as step3:
        upload_motion()

    # 第四步
    with gr.Tab("第四步") as step4:
        animation_generation()

if __name__ == '__main__':
    app.launch()
