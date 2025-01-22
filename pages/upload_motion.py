import gradio as gr
from util.media_io import fetch_image_as_numpy, on_file_upload, generate_random_dirname
from util import global_state
from typing import Union


def on_radio_change(choice):
    if choice == "*.mp4文件":
        global_state.motion_suffix = "mp4"
        return gr.Markdown("## 上传一个人体运动视频片段，请尽量保证拍摄分辨率的清晰", visible=True), gr.Video(visible=True)
    elif choice == "*.bvh文件":
        from util.netconfig import static_bvh_res_url
        global_state.motion_suffix = "bvh"
        return gr.Markdown(
            f"## 上传一个规范的\*.bvh动捕文件，请保证满足Human3.6M的17个关键点格式定义，关节点结构和命名示例如下：  ![bvh_demo]({static_bvh_res_url}/bvh_demo.png)",
            visible=True), gr.File(visible=True)
    else:
        assert NotImplementedError, 'Undefined choice!!'


def on_upload_button_down(data: Union[str, gr.File]):
    from util.netconfig import upload_sketch_url

    if global_state.remote_sketch_base_dir == "":
        global_state.remote_sketch_base_dir = generate_random_dirname(length=16)

    gr.Info("开始制作动画，请勿频繁点击按钮重复上传！！")
    ret_value = on_file_upload(data=data, server_url=upload_sketch_url,
                               other_info_dict={'sketch_name': global_state.remote_sketch_base_dir,
                                                'step_number': 2,
                                                'file_suffix': global_state.motion_suffix}
                               )
    if ret_value is not None:
        if ret_value.get("error", None) is not None:
            gr.Error(message=f"文件上传失败，具体原因  {ret_value['error']}  {ret_value['detail']}")
            return
    gr.Info("动画制作完成")


def upload_motion():
    with gr.Blocks():
        # with gr.Column():
        gr.Markdown("# 第三步：上传视频或动捕文件")
        motion_radio = gr.Radio(label="请选择上传文件类型：", choices=['*.mp4文件', '*.bvh文件'], type="value")
        markdown_video = gr.Markdown("**上传一个人体运动视频片段，请尽量保证拍摄分辨率的清晰**", visible=False)
        motion_file = gr.File(visible=False, interactive=True)

        upload_btn = gr.Button("上传文件")

        # 事件定义
        motion_radio.change(fn=on_radio_change, inputs=motion_radio, outputs=[markdown_video, motion_file])
        upload_btn.click(fn=on_upload_button_down, inputs=[motion_file], outputs=[])
