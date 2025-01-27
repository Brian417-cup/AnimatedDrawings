# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys


def start(user_mvc_cfg_fn: str):

    # build cfg
    from animated_drawings.config import Config
    cfg: Config = Config(user_mvc_cfg_fn)

    # create view
    from animated_drawings.view.view import View
    view = View.create_view(cfg.view)

    # create scene
    from animated_drawings.model.scene import Scene
    scene = Scene(cfg.scene)

    # create controller
    from animated_drawings.controller.controller import Controller
    controller = Controller.create_controller(cfg.controller, scene, view)

    # if call .run() function, it will start the animation maker loop
    #         self._prep_for_run_loop() # to get start time
    #         while not self._is_run_over(): # get all frame time is over?
    #
    #             In windowview, clearing window will be used as OpenGL:
    #                   GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    #
    #             self._start_run_loop_iteration()
    #
    #             Due to Scene class implements with Transform class.
    #             Therefore, if it uses update, it will excute update_transforms in Transform class
    #             self._update()
    #
    #             self._render()
    #
    #             Tick the time for plus one step
    #             self._tick()
    #
    #             Currently, this function has not been used
    #             self._handle_user_input()
    #
    #
    #             self._finish_run_loop_iteration()
    #
    #         self._cleanup_after_run_loop()

    controller.run()


if __name__ == '__main__':
    logging.basicConfig(filename='log.txt', level=logging.DEBUG)

    # user-specified mvc configuration filepath. Can be absolute, relative to cwd, or relative to ${AD_ROOT_DIR}
    user_mvc_cfg_fn = sys.argv[1]

    start(user_mvc_cfg_fn)
