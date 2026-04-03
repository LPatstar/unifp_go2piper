# UniFP Go2+Piper Fork


<div align="center">
Go2+Piper-focused fork of the UniFP project

[[Website]](https://unified-force.github.io/)
[[Arxiv]](https://arxiv.org/pdf/2505.20829)
[[Oral Talk]](https://youtu.be/9lzFVQoc4Do?t=2652)

<p align="center">
    <img src="docs/teaser.jpg" height=400px"> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
</p>

[![IsaacGym](https://img.shields.io/badge/IsaacGym-Preview4-b.svg)](https://developer.nvidia.com/isaac-gym)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://docs.python.org/3/whatsnew/3.8.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)

</div>

## Overview

This repository is a Go2+Piper-oriented fork of the original UniFP project. It keeps the reinforcement learning-based whole-body control framework from UniFP, while adapting the current codebase, robot assets, and training setup toward Go2+Piper unified position and force control in Isaac Gym.

The original UniFP project, paper, and website are linked above for method background. This fork is intended to be a practical workspace for Go2+Piper simulation training, evaluation, debugging, and further adaptation.

**Key Features**:
- Go2+Piper whole-body control setup built on UniFP
- Unified policy learning for position and force control
- PPO-based reinforcement learning training in Isaac Gym
- Go2+Piper-specific robot assets, configs, and play scripts
- Keyboard-driven keyplay tooling for manual policy inspection

## TODO
- [x] Release UniFP training pipeline
- [ ] Release sim2real with ROS2
- [ ] Release sim2sim in MuJoCo
- [ ] Release imitation learing data collection pipeline

## Installation

### System Requirements
- Ubuntu 20.04/22.04
- Python 3.8
- CUDA 11.2+
- Isaac Gym Preview 4 (requires NVIDIA developer account)

### Installation Steps

1. **Clone this project**
   ```bash
   git clone https://github.com/LPatstar/unifp_go2piper.git
   cd unifp_go2piper
   ```

2. **Set up the environment**
   ```bash
   conda create -n unifp python=3.8 
   # isaacgym requires python <=3.8
   conda activate unifp
   # Download the Isaac Gym binaries from https://developer.nvidia.com/isaac-gym 
   wget https://developer.nvidia.com/isaac-gym-preview-4
   tar -xvzf isaac-gym-preview-4
   
   cd isaacgym/python && pip install -e .
   ```
    For libpython error:
    - Set LD_LIBRARY_PATH:
        ```bash
        export LD_LIBRARY_PATH=</path/to/conda/envs/your_env/lib>:$LD_LIBRARY_PATH
        ```

3. **Install Python dependencies**
   ```bash
   # Install PyTorch
   conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
   
   # Install other dependencies
   pip install numpy matplotlib wandb
   ```


## Usage

### Policy Training

#### Go2+Piper Position-Force Control Training
```bash
cd legged_gym/scripts
python train_go2piperposforce.py --task=go2_piper_pos_force --headless
```

### Policy Evaluation and Testing

#### Run Trained Policies
```bash
# Go2+Piper position-force control testing
python play_go2piperposforce.py --task=go2_piper_pos_force --load_run=<run_name>
```

#### Keyboard-Controlled Keyplay
Use the keyplay script to manually command the Go2+Piper policy from the viewer instead of relying on randomized commands.

```bash
# Go2+Piper keyboard-controlled evaluation
python keyplay_go2piperposforce.py --task=go2_piper_pos_force --load_run=<run_name>
```

This mode keeps policy inference running, but replaces the usual randomized command stream with viewer keyboard input:

- Base motion commands:
  - `W/S`: increase/decrease forward velocity
  - `A/D`: increase/decrease lateral velocity
  - `Q/E`: increase/decrease yaw rate (`Q` is counterclockwise)
- End-effector Cartesian target commands:
  - Numpad `8/2`: move EE target along local `x +/-`
  - Numpad `4/6`: move EE target along local `y +/-`
  - Numpad `9/3`: move EE target along local `z +/-`
  - Numpad `0`: set EE target to the current end-effector pose
- Force command adjustments:
  - `J/K`: decrease/increase end-effector force command on `x`
  - `O/I`: decrease/increase base force command on `x`
- Reset shortcuts:
  - `R`: reset base motion commands to zero
  - Numpad `5`: reset EE target to the home position
  - `N`: reset force commands to zero
- Viewer controls:
  - `F`: toggle follow camera
  - `V`: toggle viewer sync
  - `SPACE`: pause/unpause

Notes:

- The end-effector target is controlled in a yaw-aligned Cartesian frame for more intuitive teleoperation.
- Keyplay is intended for debugging, qualitative policy inspection, and fast command-side testing.

#### Automated Evaluation Benchmark
Use the automated evaluation script to run a reproducible benchmark suite for the current Go2+Piper checkpoint. It evaluates position tracking, hybrid force-position behavior, base disturbance compensation, estimator quality, and whole-body robustness, then exports a machine-readable JSON report plus a Markdown summary.

```bash
# Go2+Piper automated evaluation
python eval_go2piperposforce.py --task=go2_piper_pos_force --load_run=<run_name> --headless
```

Useful options:

- `--eval_case all`
  Run the full benchmark suite. You can also choose `position_only`, `hybrid_force_position`, `base_disturbance`, or `mixed_whole_body`.
- `--eval_repeats <N>`
  Repeat each scripted scenario `N` times before aggregating the final report.
- `--output_dir <dir>`
  Directory used to save the exported evaluation reports. Default is `eval_reports/`.

Outputs:

- `summary.json`
  Structured metrics for each evaluation case, estimator quality, runtime quality metrics, the final overall score, and the resolved model run/checkpoint that was actually evaluated.
- `summary.md`
  Human-readable report with the main scores, raw physical metrics, and the resolved model run/checkpoint metadata.

For a detailed explanation of the benchmark design and every metric, see [GO2_PIPER_EVAL_METRICS.md](GO2_PIPER_EVAL_METRICS.md).

### Parameter Configuration

#### Training Parameters
- `--task`: Task name (`go2_piper_pos_force`, `b2z1_pos_force`, etc.)
- `--headless`: Run in headless mode
- `--num_envs`: Number of parallel environments
- `--max_iterations`: Maximum training iterations

#### Environment Parameters
- `--flat_terrain`: Use flat terrain
- `--physics_engine`: Physics engine (physx)
- `--sim_device`: Simulation device (cuda:0)


### Core Components

- **Environment Configuration** (`legged_gym/envs/go2/go2_piper_pos_force_config.py`)
  - Robot initial state configuration
  - Reward function parameters
  - Observation space definition
  - Action space definition

- **Environment Implementation** (`legged_gym/envs/go2/legged_robot_go2_piper_pos_force.py`)
  - Simulation environment logic
  - Reward calculation
  - Observation space construction
  - Action execution

- **Training Algorithm** (`legged_gym/b2_gym_learn/ppo_cse_pf/`)
  - PPO algorithm implementation
  - Policy network structure
  - Value network structure

- **Task Registration** (`legged_gym/envs/__init__.py`)
  - Task registration management
  - Environment creation
  - Trainer creation

## Upstream Reference

This repository is derived from the original UniFP project. If you are looking for the original paper/project context rather than this Go2+Piper-oriented fork, use the links at the top of this README.
