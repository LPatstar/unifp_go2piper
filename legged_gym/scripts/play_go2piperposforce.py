import os
import sys

unitree_rl_gym_path = os.path.abspath(__file__ + "../../../../")
sys.path.append(unitree_rl_gym_path)

import legged_gym.scripts.play_b2z1posforce as play_impl
from legged_gym.utils import get_args


if __name__ == "__main__":
    play_impl.EXPORT_POLICY = True
    play_impl.RECORD_FRAMES = False
    play_impl.MOVE_CAMERA = False
    play_impl.FIX_COMMAND = False
    play_impl.VISUAL_PRED = True
    play_impl.ENABLE_PLAY_CMD_FORCE = True

    args = get_args()
    if not getattr(args, "task", None):
        args.task = "go2_piper_pos_force"
    play_impl.play(args)
