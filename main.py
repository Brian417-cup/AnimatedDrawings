import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import os.path as osp

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

STATIC_DIR = "static"  # static resource directory
UPLOAD_DIR = "upload"  # upload resource directory
CHARACTER_DIR = 'char'
SKETCH_DETECTOR_PATH = "examples/offline_res/checkpoint/sketch_detector.onnx"
SKETCH_ESTIMATOR_PATH = "examples/offline_res/checkpoint/sketch_estimator.onnx"
MOR_ITE = 5
ANIMATION_SCRIPT_DIR = "examples"
SCRIPT_SOURCE_DIR = osp.abspath(osp.dirname(__file__))


@app.get("/dynamic-static/{file_path:path}")
async def get_dynamic_static(file_path: str):
    file_path_full = os.path.join(STATIC_DIR, file_path)

    if os.path.exists(file_path_full):
        return FileResponse(file_path_full)
    else:
        raise HTTPException(status_code=404, detail="File not found")


@app.get("/upload/{file_path:path}")
async def get_dynamic_static(file_path: str):
    file_path_full = os.path.join(UPLOAD_DIR, file_path)

    if os.path.exists(file_path_full):
        return FileResponse(file_path_full)
    else:
        raise HTTPException(status_code=404, detail="File not found")


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...), sketch_name: str = Form(...),
                       step_number: int = Form(...), file_suffix: str = Form(...)):
    import os
    import os.path as osp
    import shutil

    os.makedirs(osp.join(UPLOAD_DIR, sketch_name), exist_ok=True)

    # upload image sketch
    if step_number == 1:
        file_name = 'char.png'
    # upload motion file resource
    elif step_number == 2:
        if file_suffix == 'mp4':
            file_name = 'motion.mp4'
        elif file_suffix == 'bvh':
            file_name = 'motion.bvh'
        else:
            assert NotImplementedError, 'Unsupport file format!!'
    else:
        assert NotImplementedError, 'Currently, other format is not supported!!'

    file_path = osp.join(UPLOAD_DIR, sketch_name, file_name)

    # save into custom directory
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # for different return json format
    if step_number == 1:
        # joint data
        from examples.offline_lib.sketch.sketch_estimator_and_detector import SketchDetectorAndEstimator
        sketch_estimator_dir = osp.join(UPLOAD_DIR, sketch_name)
        joint_image_path = osp.join(sketch_estimator_dir, CHARACTER_DIR, 'joint_image.png')

        sketch_executor = SketchDetectorAndEstimator(
            detector_path=SKETCH_DETECTOR_PATH,
            estimator_path=SKETCH_ESTIMATOR_PATH
        )
        sketch_executor.inference(img_path=file_path, show_res=False, need_mask=True,
                                  morphops_iteration=MOR_ITE, out_dir=sketch_estimator_dir)

        # triangle data
        from examples.offline_lib.sketch import triangle_algorithm
        import numpy as np
        triangle_base_dir = osp.join(UPLOAD_DIR, sketch_name, CHARACTER_DIR)
        triangle_image_path = osp.join(triangle_base_dir, 'triangle_image.png')

        mask_path = osp.join(UPLOAD_DIR, sketch_name, CHARACTER_DIR, 'mask_image.png')
        mask_data = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        ret, mask_copy = cv2.threshold(mask_data.copy(), 0, 255, cv2.THRESH_OTSU)

        res = triangle_algorithm.generate_triangle_mesh(mask_copy)
        triangle_algorithm.plot_mesh_plt(vertices=res['vertices'], triangles_idxs=res['triangles_idxs'],
                                         pins_xy=np.array([]),
                                         x_axis_range=(0, mask_data.shape[1]), y_axis_range=(0, mask_data.shape[0]),
                                         show_res=False,
                                         save_data=True, save_base_dir=triangle_base_dir)

        return JSONResponse(content={"joint_image_url": f"{joint_image_path}",
                                     "triangle_image_url": f"{triangle_image_path}"}, status_code=200)
    elif step_number == 2:
        sketch_file_path = osp.join(SCRIPT_SOURCE_DIR, UPLOAD_DIR, sketch_name, 'char.png')
        motion_file_path = osp.join(SCRIPT_SOURCE_DIR, file_path)
        output_video_path = osp.join(SCRIPT_SOURCE_DIR, UPLOAD_DIR, sketch_name, 'animation.mp4')

        import sys
        os.system(
            f"cd {ANIMATION_SCRIPT_DIR} && {sys.executable} offline_demo.py "
            f"--src_sketch {sketch_file_path} --src_motion {motion_file_path} --out_vid {output_video_path}")

        return JSONResponse(content={"animation_url": output_video_path}, status_code=200)
    else:
        assert NotImplementedError, "Current resource doesn't exist"


@app.get("/animation/")
async def generate_animation_2d(uuid: str = Form(...)):
    import os
    import os.path as osp

    sketch_base_dir = os.path.join('upload', uuid)
    sketch_file_path = osp.join(sketch_base_dir, 'char.png')
    animation_file_path = osp.join(sketch_base_dir, 'animation.mp4')

    return JSONResponse(content={"sketch_file_path": sketch_file_path,
                                 'animation_file_path': animation_file_path}, status_code=200)
