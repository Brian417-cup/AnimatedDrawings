import gradio as gr
from util.media_io import fetch_image_as_numpy, on_image_upload, generate_random_dirname
from typing import Union, Tuple, List
from util import global_state


def on_upload_button_down(data: Union[str, gr.Image], upload_btn: gr.Button):
    from util.netconfig import upload_sketch_url

    gr.Info("开始上传图片，请勿频繁点击按钮重复上传！！")
    if global_state.remote_sketch_base_dir == "":
        global_state.remote_sketch_base_dir = generate_random_dirname(length=16)
    ret_value = on_image_upload(data=data, server_url=upload_sketch_url,
                                other_info_dict={'sketch_name': global_state.remote_sketch_base_dir,
                                                 'step_number': 1,
                                                 'file_suffix': 'png'}
                                )
    if ret_value is not None:
        if ret_value.get("error", None) is not None:
            gr.Error(message=f"图片处理失败，具体原因  {ret_value['error']}  {ret_value['detail']}")
            return
        else:
            global_state.joint_image_url = ret_value["joint_image_url"].replace("\\", "/")
            global_state.triangle_image_url = ret_value["triangle_image_url"].replace("\\", "/")
    gr.Info("图片上传完成")


def upload_image():
    with gr.Row():
        with gr.Column():
            gr.Markdown("# 第一步: 上传图片")
            gr.Markdown("**上传一个角色的图画，请确保胳膊和腿不与身体重叠（参考如下示例）**")
            # gr.Markdown("## 检查清单")
            # gr.Markdown("- 请确保角色画在一张没有线条、皱纹或撕裂的白纸上")
            # gr.Markdown("- 请确保绘图光线充足，要最大程度的减少阴影，请将相机拿的更远一些，然后放大绘图")
            # gr.Markdown("- 请勿包含任何可识别身份的信息")

            with gr.Column():
                from util.netconfig import static_drawing_res_url

                with gr.Row():
                    gr.Image(interactive=False,
                             value=fetch_image_as_numpy(image_url=f"{static_drawing_res_url}/char1.png"),
                             show_label=False)
                    gr.Image(interactive=False,
                             value=fetch_image_as_numpy(image_url=f"{static_drawing_res_url}/char2.png"),
                             show_label=False)
                with gr.Row():
                    gr.Image(interactive=False,
                             value=fetch_image_as_numpy(image_url=f"{static_drawing_res_url}/char3.png"),
                             show_label=False)
                    gr.Image(interactive=False,
                             value=fetch_image_as_numpy(image_url=f"{static_drawing_res_url}/char4.png"),
                             show_label=False)

        with gr.Column(scale=1):
            image_input = gr.Image(label="上传图片", height=400, type="filepath")

    # 提交文件按钮
    with gr.Row():
        with gr.Column(scale=1):
            pass

        with gr.Column(scale=2):
            pass

        with gr.Column(scale=1):
            upload_btn = gr.Button("上传文件")

    upload_btn.click(fn=on_upload_button_down, inputs=[image_input, upload_btn])
