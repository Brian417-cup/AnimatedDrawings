# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from animated_drawings.view.view import View
from animated_drawings.view.shaders.shader import Shader
from animated_drawings.view.utils import get_projection_matrix
from animated_drawings.utils import read_background_image
from animated_drawings.model.scene import Scene
from animated_drawings.model.camera import Camera
from animated_drawings.model.transform import Transform
from animated_drawings.config import ViewConfig
import glfw
import OpenGL.GL as GL
import logging
from typing import Tuple, Dict
import numpy as np
import numpy.typing as npt
from pathlib import Path
from pkg_resources import resource_filename


class WindowView(View):
    """Window View for rendering into a visible window"""

    def __init__(self, cfg: ViewConfig) -> None:
        super().__init__(cfg)

        glfw.init()

        # 存储了从世界空间到相机空间的旋转矩阵
        # create a camera, in this class, they use look at
        self.camera: Camera = Camera(cfg.camera_pos, cfg.camera_fwd)

        self.win: glfw._GLFWwindow
        self._create_window(*cfg.window_dimensions)  # pyright: ignore[reportGeneralTypeIssues]

        # prepare shader relative
        self.shaders: Dict[str, Shader] = {}
        self.shader_ids: Dict[str, int] = {}
        self._prep_shaders()

        # frame buffer id
        self.fboId: GL.GLint
        self._prep_background_image()

        self._set_shader_projections(get_projection_matrix(*self.get_framebuffer_size()))

    def _prep_background_image(self) -> None:
        """ Initialize framebuffer object for background image, if specified. """

        # if nothing specified, return
        if not self.cfg.background_image:
            return

        # load background image (RGBA)
        _txtr: npt.NDArray[np.uint8] = read_background_image(self.cfg.background_image)

        # OpenGL texture loading process
        # create the opengl texture and send it data
        self.txtr_h, self.txtr_w, _ = _txtr.shape
        # 1. create a id and bind a corresponding texture
        # create 1 texture class
        self.txtr_id = GL.glGenTextures(1)
        # set texture store location
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)
        # bind texture reousrce from its id
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.txtr_id)
        # set basic and max level for texture
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_BASE_LEVEL, 0)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAX_LEVEL, 0)
        # create a truly texture2D class
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, self.txtr_w, self.txtr_h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE,
                        _txtr)

        # 2. make framebuffer object (means texture will be used under frame buffer)
        # The data here we will stored in the read frame buffer
        self.fboId: GL.GLint = GL.glGenFramebuffers(1)
        # bind frame buffer id to its corresponding resource
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fboId)
        # 3. put texture resource into its framebuffer
        # In common OpenGL framework, the correct API is glTexImage2D
        GL.glFramebufferTexture2D(GL.GL_READ_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.txtr_id, 0)

    def _prep_shaders(self) -> None:
        '''
        Each shader consists of vertex shader and fragment shader. The shader id and shader class are store in
        self.shaders and self.shader_ids
        '''
        #########################################################################
        #### Following 2 shader is rebundant currently
        BVH_VERT = Path(resource_filename(__name__, "shaders/bvh.vert"))
        BVH_FRAG = Path(resource_filename(__name__, "shaders/bvh.frag"))
        self._initiatize_shader('bvh_shader', str(BVH_VERT), str(BVH_FRAG))

        COLOR_VERT = Path(resource_filename(__name__, "shaders/color.vert"))
        COLOR_FRAG = Path(resource_filename(__name__, "shaders/color.frag"))
        self._initiatize_shader('color_shader', str(COLOR_VERT), str(COLOR_FRAG))
        #########################################################################
        TEXTURE_VERT = Path(resource_filename(__name__, "shaders/texture.vert"))
        TEXTURE_FRAG = Path(resource_filename(__name__, "shaders/texture.frag"))
        self._initiatize_shader('texture_shader', str(TEXTURE_VERT), str(TEXTURE_FRAG), texture=True)

    def _update_shaders_view_transform(self, camera: Camera) -> None:
        '''
        This function is called during animation for setting view matrix in all vertex shader.
        '''
        try:
            view_transform: npt.NDArray[np.float32] = np.linalg.inv(camera.get_world_transform())
        except Exception as e:
            msg = f'Error inverting camera world transform: {e}'
            logging.critical(msg)
            assert False, msg

        # for each shader, we set view matrix that transform from world space -> camera sapce
        for shader_name in self.shaders:
            GL.glUseProgram(self.shader_ids[shader_name])
            view_loc = GL.glGetUniformLocation(self.shader_ids[shader_name], "view")
            GL.glUniformMatrix4fv(view_loc, 1, GL.GL_FALSE, view_transform.T)

    def _set_shader_projections(self, proj_m: npt.NDArray[np.float32]) -> None:
        '''
        Chanage projection matrix in vertex shader
        '''
        for shader_id in self.shader_ids.values():
            # 1. activate one shader id
            GL.glUseProgram(shader_id)
            # 2. fetch variable's id that need to be modified
            proj_loc = GL.glGetUniformLocation(shader_id, "proj")
            # 3. modify the variable in accrondance with its id and set value
            GL.glUniformMatrix4fv(proj_loc, 1, GL.GL_FALSE, proj_m.T)

    def _initiatize_shader(self, shader_name: str, vert_path: str, frag_path: str, **kwargs) -> None:
        # In a Shader class, there are vertex shader and fragment shader 2 files, they share a created glid
        # It directly store shader class
        self.shaders[shader_name] = Shader(vert_path, frag_path)
        # It store shader id
        self.shader_ids[shader_name] = self.shaders[shader_name].glid  # pyright: ignore[reportGeneralTypeIssues]

        # It's for namaed 'texture' shader
        if 'texture' in kwargs and kwargs['texture'] is True:
            # means opengl activate currently shader
            GL.glUseProgram(self.shader_ids[shader_name])
            # GL.glUniform1i is to set variable values, it is settled to be 0 by default
            GL.glUniform1i(
                # get a corresponding variable location by its name
                GL.glGetUniformLocation(
                    self.shader_ids[shader_name], 'texture0')
                , 0)

    def _create_window(self, width: int, height: int) -> None:
        # fllowing setting is to set opengl basic configuration by glfw library
        # set OpenGL version min~max
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        # compatible before ? -> True
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        # set prifile is opengl_core_profile
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        # This window can't be resized
        glfw.window_hint(glfw.RESIZABLE, False)

        # create blank view
        self.win = glfw.create_window(width, height, 'Viewer', None, None)

        # set created view into opengl current context
        glfw.make_context_current(self.win)

        # Then, we can directly use OpenGL correctly!!
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(*self.cfg.clear_color)

        logging.info(
            f'OpenGL Version: {GL.glGetString(GL.GL_VERSION).decode()}')  # pyright: ignore[reportGeneralTypeIssues]
        logging.info(
            f'GLSL: {GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode()}')  # pyright: ignore[reportGeneralTypeIssues]
        logging.info(f'Renderer: {GL.glGetString(GL.GL_RENDERER).decode()}')  # pyright: ignore[reportGeneralTypeIssues]

    def set_scene(self, scene: Scene) -> None:
        self.scene = scene

    def render(self, scene: Transform) -> None:
        '''
        It will be called during animation progress.
        Use OpenGL to render result.
        '''
        GL.glViewport(0, 0, *self.get_framebuffer_size())

        # draw the background image if exists
        # Here, please attention in the frame buffer there exists 2 types framebuffer:
        # read framebuffer and draw framebuffer
        # We want to do is to fetch data from read framebuffer into draw framebuffer
        if self.cfg.background_image:
            # bind framebuffer background from location 0 (0 is default framebuffer)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)
            # read frame buffer for background image
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fboId)
            win_w, win_h = self.get_framebuffer_size()
            # set background data from tagged read framebuffer into tagged draw framebuffer
            GL.glBlitFramebuffer(0, 0, self.txtr_w, self.txtr_h, 0, 0, win_w, win_h, GL.GL_COLOR_BUFFER_BIT,
                                 GL.GL_LINEAR)

        # transform view matrix in every shader file
        self._update_shaders_view_transform(self.camera)

        # A Scene class has many AnimatedDrawing class, therefore, it will call their own _draw() function,
        # here please attention, in each shader, every AnimatedDrawing class will set its own model matrix
        scene.draw(shader_ids=self.shader_ids, viewer_cfg=self.cfg)

    def get_framebuffer_size(self) -> Tuple[int, int]:
        """ Return (width, height) of view's window. """
        return glfw.get_framebuffer_size(self.win)

    def swap_buffers(self) -> None:
        glfw.swap_buffers(self.win)

    def clear_window(self) -> None:
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)  # type: ignore

    def cleanup(self) -> None:
        """ Destroy the window when it's no longer being used. """
        glfw.destroy_window(self.win)
