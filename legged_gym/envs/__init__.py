from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


from legged_gym.envs.b2.b2z1_pos_force_config import B2Z1PosForceRoughCfg, B2Z1PosForceRoughCfgPPO
from legged_gym.envs.go2.go2_piper_pos_force_config import Go2PiperPosForceRoughCfg, Go2PiperPosForceRoughCfgPPO
from legged_gym.envs.door.b2z1_door_open_config import B2Z1DoorOpenRoughCfg, B2Z1DoorOpenRoughCfgPPO

from .base.legged_robot import LeggedRobot
from .b2.legged_robot_b2z1_pos_force import LeggedRobot_b2z1_pos_force
from .go2.legged_robot_go2_piper_pos_force import LeggedRobot_go2_piper_pos_force
from .door.legged_robot_b2z1_door_open import LeggedRobot_b2z1_door_open

from legged_gym.utils.task_registry import task_registry


task_registry.register( "b2z1_pos_force", LeggedRobot_b2z1_pos_force, B2Z1PosForceRoughCfg(), B2Z1PosForceRoughCfgPPO())
task_registry.register( "go2_piper_pos_force", LeggedRobot_go2_piper_pos_force, Go2PiperPosForceRoughCfg(), Go2PiperPosForceRoughCfgPPO())
task_registry.register( "b2z1_door_open", LeggedRobot_b2z1_door_open, B2Z1DoorOpenRoughCfg(), B2Z1DoorOpenRoughCfgPPO())
