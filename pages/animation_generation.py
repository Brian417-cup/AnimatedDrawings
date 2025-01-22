import gradio as gr
from util.media_io import fetch_image_as_numpy, generate_animation_post
from util import global_state
from util.netconfig import base_url
from util.netconfig import acquire_animation_url


def on_animation_button_down():
    gr.Info("获取媒体资源,时间较长，请耐心等待...")
    content = generate_animation_post(server_url=f"{acquire_animation_url}",
                                      other_info_dict={"uuid": global_state.remote_sketch_base_dir})
    if hasattr(content, 'error'):
        gr.Error(message=content['detail'])
        return

    sketch_file_path = content['sketch_file_path'].replace("\\", "/")
    animation_file_path = content['animation_file_path'].replace("\\", "/")

    remote_sketch_image = f"{base_url}/{sketch_file_path}"
    remote_sketch_animation = f"{base_url}/{animation_file_path}"
    gr.Info("获取媒体资源成功!!")

    return gr.Image(label="原图", value=fetch_image_as_numpy(image_url=remote_sketch_image)), \
           gr.Video(value=remote_sketch_animation)


def animation_generation():
    with gr.Blocks():
        gr.Markdown("# 第四步：二维动画生成结果")

        with gr.Row():
            with gr.Column():
                gr.Markdown("# 原图")
                sketch_image = gr.Image(interactive=False)
            with gr.Column():
                gr.Markdown("# 二维动画")
                animation_video = gr.Video(interactive=False)

        # 提交文件按钮
        with gr.Row():
            with gr.Column(scale=1):
                pass

            with gr.Column(scale=2):
                pass

            with gr.Column(scale=1):
                acquire_result = gr.Button("获取结果")

        acquire_result.click(fn=on_animation_button_down, inputs=[], outputs=[sketch_image, animation_video])
