# Simplified Animated Drawings(Fronted Branch)

This is the simplified version
of [facebookresearch/AnimatedDrawings: Code to accompany "A Method for Animating Children's Drawings of the Human Figure" (github.com)](https://github.com/facebookresearch/AnimatedDrawings)
. here we can **custom your monocular video and drawing as input**, framework will automatically generate corresponding.

Here are some generated 2D animation:

|  demo1             |         demo2        |      demo3     |  
| -------------------| -------------------- |----------------|
|  ![demo1](demo/offline_combine_demo_1.gif)    |   ![demo2](demo/offline_combine_demo_2.gif)                |  ![demo3](demo/offline_combine_demo_3.gif) |  

## Environment Preparation

The Python version is 3.10.13. And you can install other packages use below command:

```shell
pip install -r requirements.txt
```

## Operation Steps

1. Upload image.
   ![sketch](demo/1.png)
2. Check intermediate results.(Optional)
   ![check](demo/2.png)
3. Upload the specified action files. These can be *.mp4 video files shot with the person facing the camera, or *.bvh
   files that conform to the example specification (see the page for details).
   ![motion](demo/3.png)
4. See animation result.
   ![animation](demo/4.gif)