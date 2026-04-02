import os
import sys

unitree_rl_gym_path = os.path.abspath(__file__ + "../../../../")
sys.path.append(unitree_rl_gym_path)

from legged_gym.scripts.train_b2z1posforce import train
from legged_gym.utils import get_args


if __name__ == "__main__":
    args = get_args()
    if not getattr(args, "task", None):
        args.task = "go2_piper_pos_force"
    args.headless = True
    train(args)
