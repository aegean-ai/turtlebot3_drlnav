# CLAUDE.md

## Hardware safety

These repos command real machines. Treat every rule here as a hard gate.

- **Never run a command that can move real hardware** unless the user asked for real-hardware
  operation in this session. That includes any service, launch file, or script whose name
  contains `hardware`, `real`, `driver`, or `calibrate`, and anything that opens a `/dev/tty*`
  port. When in doubt, run the simulation variant and say which one you ran.
- **Simulation is the default.** Develop, test, and reproduce in Gazebo first. Confirm which
  environment you are pointed at before publishing to `/cmd_vel` or any controller topic.
- **Always stop what you start.** End every commanded motion sequence with an explicit stop
  (zero twist on `/cmd_vel`, or halt the controller). If a sequence is interrupted or errors
  out, publish the stop yourself before doing anything else.
- **Limits and gains are read-only.** Do not change joint limits, velocity or acceleration
  caps, controller gains, or safety thresholds (URDF, ros2_control YAML, MoveIt
  `joint_limits`, `REAL_SPEED_*` / `REAL_THRESHOLD_*`) without the user signing off on the
  specific values.
- **Shared GPU etiquette.** Training runs take hours on a shared card. Check `nvidia-smi`
  before starting, never kill a process you did not start, and launch training detached
  rather than blocking the session. Prefer the tailnet GPU hosts over the laptop card;
  gpu-node-2 is offline, use gpu-node-1 or gpu-node-3.
- **Do not tear down running work.** Never `docker compose down`, prune, or restart
  containers without checking what is live first (`docker compose ps`, `nvidia-smi`).

## Project Overview

TurtleBot3 DRL Navigation: a ROS2 framework for training Deep Reinforcement Learning agents (DDPG, TD3, DQN) to navigate a TurtleBot3 robot in Gazebo simulation, with real robot deployment support. Built on ROS2 Foxy, PyTorch, and Gazebo 11.

## Build & Run Commands

```bash
# Build all packages (from repo root)
colcon build

# Source workspace after building
source install/setup.bash

# Build a single package
colcon build --packages-select turtlebot3_drl

# Preferred: docker compose runs the whole stack (simulation, environment,
# gazebo-goals, drl-agent, plus zenoh-router/zenoh-bridge for the real robot).
#   docker compose up simulation environment gazebo-goals drl-agent
# The raw ros2 commands below are the inside-container reference for those services.

# Manual equivalent (4 terminals, each sourced):
# 1. Launch Gazebo simulation (stages 1-10 available)
ros2 launch turtlebot3_gazebo turtlebot3_drl_stage4.launch.py

# 2. Start goal spawner
ros2 run turtlebot3_drl gazebo_goals

# 3. Start environment node
ros2 run turtlebot3_drl environment

# 4. Start training (algorithm: ddpg, td3, or dqn)
ros2 run turtlebot3_drl train_agent ddpg

# Testing a trained model: test_agent <algorithm> <model_name> <num_episodes>
ros2 run turtlebot3_drl test_agent ddpg "ddpg_0" 500

# Real robot deployment
ros2 run turtlebot3_drl real_environment
ros2 run turtlebot3_drl real_agent ddpg ddpg_1_stage4 1000
```

Required environment variables (set in Dockerfile or manually):
- `TURTLEBOT3_MODEL=burger`
- `GAZEBO_MODEL_PATH` must include turtlebot3 simulation models

## Architecture

### ROS2 Package Layout

- **`src/turtlebot3_drl/`**: Main DRL package (ament_python). All DRL logic lives here.
- **`src/turtlebot3_simulations/`**: Gazebo worlds and launch files for stages 1-10.
- **`src/turtlebot3_msgs/`**: Custom ROS2 service/message definitions (`DrlStep.srv`, `Goal.srv`, `RingGoal.srv`).

### DRL Agent Architecture (`src/turtlebot3_drl/turtlebot3_drl/`)

**Agent layer** (`drl_agent/`):
- `drl_agent.py`: ROS2 node with training/testing loop. Communicates with environment via `step_comm` service. Entry points: `main_train`, `main_test`, `main_real`.
- `off_policy_agent.py`: Base class defining the interface for off-policy algorithms (network init, action selection, train step, save/load).
- `ddpg.py`, `td3.py`, `dqn.py`: Algorithm implementations subclassing `OffPolicyAgent`.

**Environment layer** (`drl_environment/`):
- `drl_environment.py`: ROS2 node that manages Gazebo interaction. Subscribes to LiDAR (`/scan`), odometry (`/odom`), publishes velocity commands (`/cmd_vel`). Handles collision detection, goal checking, episode resets.
- `drl_environment_real.py`: Real robot variant with hardware-specific thresholds and LiDAR correction.
- `reward.py`: Reward function definitions. Selected via `REWARD_FUNCTION` setting (default "A").

**Shared utilities** (`common/`):
- `settings.py`: **Central configuration file**. All hyperparameters, environment constants, and feature flags. This is the primary file to edit when tuning.
- `utilities.py`: Model loading, action translation, state space construction.
- `replaybuffer.py`: Experience replay buffer.
- `storagemanager.py`: Model checkpoint save/load, graph data persistence.

### Inter-node Communication

The agent and environment run as separate ROS2 nodes communicating via services:
1. Agent calls `step_comm` (DrlStep service) with an action
2. Environment executes action, advances simulation, computes reward
3. Environment returns new state, reward, done flag
4. Agent calls `goal_comm` (Goal service) for new goals on episode reset

### State and Action Spaces

**State**: `NUM_SCAN_SAMPLES` (40) LiDAR readings + goal distance + goal angle + previous linear action + previous angular action = 44 values.

**Actions**: DDPG/TD3 output 2 continuous values [-1, 1] mapped to linear/angular velocity. DQN selects from 5 discrete predefined actions.

### Model Storage

Models are saved to `src/turtlebot3_drl/model/<HOSTNAME>/<MODEL_NAME>/` with weights, graph data, replay buffer, and training logs. `MODEL_STORE_INTERVAL` (default 100 episodes) controls checkpoint frequency.

## Key Conventions

- **Adding a new algorithm**: Subclass `OffPolicyAgent` in `drl_agent/`, implement required methods (network definition, `get_action`, `train`, `save/load`).
- **Modifying rewards**: Add a new function in `reward.py` and set `REWARD_FUNCTION` in `settings.py`.
- **Scan sample count must match the trained model.** `NUM_SCAN_SAMPLES` is not a `settings.py` constant: it is computed at runtime by `util.get_scan_count()` in `drl_environment/drl_environment.py`. `REAL_N_SCAN_SAMPLES` in `settings.py` and the hardcoded value in `zenoh_bridge/zenoh_adapter.py` must both equal the count the model was trained with. Change all three together or the real robot feeds the network a wrong-sized state.
- **Subpackages**: `common/`, `drl_agent/`, `drl_environment/`, `drl_gazebo/`, and `zenoh_bridge/` (the real-robot transport bridge).
- **Stage environments**: 10 Gazebo worlds of increasing difficulty. Stage number is written to `/tmp/drlnav_current_stage.txt` by launch files.
- **Docker**: Based on CUDA 11.3.1 + Ubuntu 20.04 + ROS2 Foxy. Use `docker compose` (services: `simulation`, `environment`, `gazebo-goals`, `drl-agent`, `zenoh-router`, `zenoh-bridge`, `dev`); `Dockerfile.gpu` and `Dockerfile.torch.gpu` also exist.
- **ROS distro is pinned to Foxy with Gazebo Classic 11.** Sibling robotics repos here use Jazzy and Gazebo Harmonic. Do not port their patterns into this repo.
