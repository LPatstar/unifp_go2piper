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

#### Position-Force Control Training
```bash
cd legged_gym/scripts
python train_b2z1posforce.py --headless
```

For Go2+Piper-specific training, use:

```bash
python train_go2piperposforce.py --headless
```

Default task behavior:

- `train_b2z1posforce.py` / `play_b2z1posforce.py`: default to `b2z1_pos_force`
- `train_go2piperposforce.py` / `play_go2piperposforce.py`: default to `go2_piper_pos_force`
- `eval_posforce.py` / `keyplay_posforce.py`: generic scripts, default to `b2z1_pos_force`; add `--task=go2_piper_pos_force` when inspecting Go2+Piper runs

### Policy Evaluation and Testing

#### Run Trained Policies
```bash
# Position-force control testing
python play_b2z1posforce.py --load_run=<run_name>
```

For Go2+Piper-specific play, use `python play_go2piperposforce.py --load_run=<run_name>`.

Useful options:

- `--draw`
  Run a short play window, record joint command/actual trajectories, then stop and save two plots: one for a front-left leg and one for the task-relevant arm joints. The shared script auto-resolves the correct arm joint names for Go2+Piper and B2+Z1.
- `--draw_steps <N>`
  Number of play steps to record before saving the plots. Default is `1000`.

When `--draw` is enabled, the saved figures are written under `play_draws/` in the project root.

#### Keyboard-Controlled Keyplay
Use the generic keyplay script to manually command a position-force policy from the viewer instead of relying on randomized commands.

```bash
# Keyboard-controlled evaluation
python keyplay_posforce.py --load_run=<run_name>

# Keyplay with draw export; press X in the viewer to save plots and exit
python keyplay_posforce.py --load_run=<run_name> --draw
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
- Exit shortcut:
  - `X`: exit keyplay
- Viewer controls:
  - `F`: toggle follow camera
  - `V`: toggle viewer sync
  - `SPACE`: pause/unpause

When `--draw` is enabled in keyplay:

- Joint command/actual trajectories are recorded continuously while you inspect the policy.
- Press `X` to save the leg and arm plots under `play_draws/` and exit the session.
- In this draw mode, `V` still toggles viewer sync.

Notes:

- The end-effector target is controlled in a yaw-aligned Cartesian frame for more intuitive teleoperation.
- Keyplay is intended for debugging, qualitative policy inspection, and fast command-side testing.

#### Automated Evaluation Benchmark
Use the automated evaluation script to run a reproducible benchmark suite for the current checkpoint. It evaluates position tracking, end-effector RPY tracking, hybrid force-position behavior, dedicated arm/base force estimation, base disturbance compensation, estimator quality, and whole-body robustness, then exports a machine-readable JSON report plus a Markdown summary.

```bash
# Automated evaluation
python eval_posforce.py --load_run=<run_name> --headless
```

Useful options:

- `--eval_case all`
  Run the full benchmark suite. You can also choose `position_only`, `hybrid_force_position`, `arm_force_estimation`, `base_force_estimation`, `base_disturbance`, or `mixed_whole_body`.
- `--eval_repeats <N>`
  Repeat each scripted scenario `N` times before aggregating the final report. With `N=1`, eval uses the normal `--seed` / config seed path. With `N>1`, each case repeat gets a fresh random seed and the used seeds are recorded in `summary.json`.
- `--output_dir <dir>`
  Directory used to save the exported evaluation reports. Default is `eval_reports/`.
- `--no_report`
  Run the benchmark without exporting `summary.json` or `summary.md`. Useful for quick terminal-only checks.

Outputs:

- `summary.json`
  Structured metrics for each evaluation case, including separate EE XYZ and RPY tracking fields, estimator quality, runtime quality metrics, the final overall score, and the resolved model run/checkpoint that was actually evaluated.
- `summary.md`
  Human-readable version of the same evaluation result. The exported report folder name also includes the resolved run name and checkpoint so each eval can be matched back to a tuning note.

If `--no_report` is set, the script still runs the full benchmark and prints the console summary, but it skips creating the output directory and does not write any report files.

For a detailed explanation of the benchmark design and every metric, see [EVAL_METRICS.md](EVAL_METRICS.md).

#### WandB Data Export
Use the root-level WandB export helper to pull training curves and metadata into a local folder for plotting or offline comparison.

```bash
# Export the latest local training run recorded under logs/
python fetch_wandb_data.py

# Export a specific run by name
python fetch_wandb_data.py --run_name=Apr03_22-02-52_go2_piper_tuned2

# Export locally and also sync the full TensorBoard iteration history back to WandB
python fetch_wandb_data.py --run_name=Apr03_22-02-52_go2_piper_tuned2 --sync

# Skip local export files and only sync the run back to WandB
python fetch_wandb_data.py --run_name=Apr03_22-02-52_go2_piper_tuned2 --no_fetch --sync
```

Useful options:

- `--run_name <name>`
  Target a specific training run. The script accepts an exact WandB run name, a local `logs/...` directory name, or the short local `run_name` suffix.
- `--history_stride <N>`
  Downsample WandB history during export. By default the script keeps one row every `1000` training iterations (`global_step`) and also keeps the final row.
- `--sync`
  In addition to the normal local export, also create a new WandB run named `*_tb_sync` from the local TensorBoard event file so the online charts use the real iteration-aligned step axis.
- `--no_fetch`
  Skip local file export. If used together with `--sync`, the script only performs the WandB sync. If used without `--sync`, the script resolves the run and exits without writing export files.
- `--entity <entity>`
  Override the WandB entity/team if it cannot be inferred automatically from the current WandB login state.
- `--project <project>`
  WandB project name. Default is `UniFP`.
- `--output_dir <dir>`
  Directory used to store the exported run files. Default is `wandb_exports/`.

Outputs:

- `run_info.json`
  Resolved run metadata, including the final WandB run name, id, URL, and how the target run was resolved.
- `summary.json`
  WandB summary metrics for the selected run.
- `config.json`
  WandB config for the selected run.
- `history.jsonl`
  Downsampled, cleaned metric history aligned by training iteration.
- `history.csv`
  Downsampled, cleaned metric history in CSV format for spreadsheet-style inspection.
- `ai_ready.json`
  The preferred compact file for AI-assisted tuning. It bundles run metadata, summary, config, and the cleaned/downsampled iteration-aligned history in one place.

Notes:

- By default the script keeps one iteration-aligned history row every `1000` training steps plus the final row. You can change this with `--history_stride <N>`.
- When the matching local TensorBoard event file is available, the script prefers it over WandB history so the exported `history` follows the real training-step axis instead of WandB's internal event indexing.
- Before downsampling, scalar rows from the same training iteration are merged into a single record, and the export is filtered by the keys that appear in `summary.json`, so repeated timestamp lines and obvious low-value system metrics such as GPU fan speed are skipped.
- Metrics logged only on the TensorBoard `/time` axis are intentionally excluded so `ai_ready.json` stays focused on training-iteration data that is useful for tuning.
- If `--sync` is enabled, the script additionally uploads the full local TensorBoard iteration history to a new WandB run with the suffix `*_tb_sync`. This does not rewrite the original online run.
- If `--no_fetch` is enabled, the compact local export files are skipped.
- Combined behavior:
  - no flags: fetch/export only
  - `--sync`: fetch/export + WandB sync
  - `--no_fetch --sync`: WandB sync only
  - `--no_fetch` alone: resolve/read only, then exit without writing local export files

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

- **Shared B2+Z1 Environment Configuration** (`legged_gym/envs/b2/b2z1_pos_force_config.py`)
  - Base position-force task configuration
  - Reward function parameters
  - Observation space definition
  - Action space definition

- **Go2+Piper Configuration Override** (`legged_gym/envs/go2/go2_piper_pos_force_config.py`)
  - Robot initial state configuration
  - Reward function parameters
  - Observation space definition
  - Action space definition

- **Shared Environment Implementation** (`legged_gym/envs/b2/legged_robot_b2z1_pos_force.py`)
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
