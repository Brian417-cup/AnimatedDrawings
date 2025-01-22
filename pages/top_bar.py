import gradio as gr

def top_bar():
    with gr.Row():
        with gr.Column(scale=2):
            gr.Blocks()

        with gr.Column(scale=1):
            gr.Markdown("# 动画速通宝")

        with gr.Column(scale=2):
            gr.Blocks()

    # 简介
    with gr.Row():
        gr.Markdown("### Tips: 这是一款基于视频动捕的二维手绘人形简笔对象动画制作器，你仅需要提供手绘的人形简笔画和单人运动视频")