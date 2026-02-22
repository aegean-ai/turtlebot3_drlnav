# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

# Training workflow (4 terminals, each sourced):
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

- **`src/turtlebot3_drl/`** — Main DRL package (ament_python). All DRL logic lives here.
- **`src/turtlebot3_simulations/`** — Gazebo worlds and launch files for stages 1-10.
- **`src/turtlebot3_msgs/`** — Custom ROS2 service/message definitions (`DrlStep.srv`, `Goal.srv`, `RingGoal.srv`).

### DRL Agent Architecture (`src/turtlebot3_drl/turtlebot3_drl/`)

**Agent layer** (`drl_agent/`):
- `drl_agent.py` — ROS2 node with training/testing loop. Communicates with environment via `step_comm` service. Entry points: `main_train`, `main_test`, `main_real`.
- `off_policy_agent.py` — Base class defining the interface for off-policy algorithms (network init, action selection, train step, save/load).
- `ddpg.py`, `td3.py`, `dqn.py` — Algorithm implementations subclassing `OffPolicyAgent`.

**Environment layer** (`drl_environment/`):
- `drl_environment.py` — ROS2 node that manages Gazebo interaction. Subscribes to LiDAR (`/scan`), odometry (`/odom`), publishes velocity commands (`/cmd_vel`). Handles collision detection, goal checking, episode resets.
- `drl_environment_real.py` — Real robot variant with hardware-specific thresholds and LiDAR correction.
- `reward.py` — Reward function definitions. Selected via `REWARD_FUNCTION` setting (default "A").

**Shared utilities** (`common/`):
- `settings.py` — **Central configuration file**. All hyperparameters, environment constants, and feature flags. This is the primary file to edit when tuning.
- `utilities.py` — Model loading, action translation, state space construction.
- `replaybuffer.py` — Experience replay buffer.
- `storagemanager.py` — Model checkpoint save/load, graph data persistence.

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
- **REAL_N_SCAN_SAMPLES must match model input size**: Changing LiDAR sample count for real robot requires matching changes in `NUM_SCAN_SAMPLES` used during training.
- **Stage environments**: 10 Gazebo worlds of increasing difficulty. Stage number is written to `/tmp/drlnav_current_stage.txt` by launch files.
- **Docker**: Based on CUDA 11.3.1 + Ubuntu 20.04 + ROS2 Foxy. Build with `docker build -t turtlebot3_drlnav .`
