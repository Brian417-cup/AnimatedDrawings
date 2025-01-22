import gradio as gr
from util.media_io import fetch_image_as_numpy
from util import global_state
from util.netconfig import base_url


def on_require_sketch_result_button_down():
    remote_sketch_joint = f"{base_url}/{global_state.joint_image_url}"
    remote_sketch_triangle = f"{base_url}/{global_state.triangle_image_url}"

    return gr.Image(label="结果", value=fetch_image_as_numpy(image_url=remote_sketch_joint), interactive=False), \
           gr.Image(label="结果", value=fetch_image_as_numpy(image_url=remote_sketch_triangle), interactive=False)


def show_drawing_result():
    with gr.Row():
        # 样例说明与展示
        with gr.Column():
            gr.Markdown("# 第二步：骨骼绑定与面片三角化")
            gr.Markdown("**给我们的角色绑定可视化关节展示。在下一步中，我们将使用这一步的关节位置，通过后续上传的动作捕捉数据为您的角色制作动画。**")

            from util.netconfig import static_joint_res_url, static_triangle_res_url

            with gr.Row():
                gr.Image(interactive=False,
                         value=fetch_image_as_numpy(image_url=f"{static_joint_res_url}/char1.png"),
                         show_label=False, height=256, width=256)
                gr.Image(interactive=False,
                         value=fetch_image_as_numpy(image_url=f"{static_triangle_res_url}/char1.png"),
                         show_label=False, height=256, width=256)
            with gr.Row():
                gr.Image(interactive=False,
                         value=fetch_image_as_numpy(image_url=f"{static_joint_res_url}/char2.png"),
                         show_label=False, height=200, width=256)
                gr.Image(interactive=False,
                         value=fetch_image_as_numpy(image_url=f"{static_triangle_res_url}/char2.png"),
                         show_label=False, height=200, width=256)

        # 真实结果展示
        sketch_joint_image = gr.Image(label="结果", interactive=False)

        sketch_triangle_image = gr.Image(label="结果", interactive=False)

    with gr.Row():
        with gr.Column(scale=1):
            pass

        with gr.Column(scale=2):
            pass

        with gr.Column(scale=1):
            acquire_sketch_button = gr.Button("获取结果")

    acquire_sketch_button.click(fn=on_require_sketch_result_button_down, inputs=[],
                                outputs=[sketch_joint_image, sketch_triangle_image])
