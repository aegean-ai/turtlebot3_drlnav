# Training Convergence Validation Plan

Validates that the modernized stack (ROS Jazzy + PyTorch 2.7 + Zenoh) produces
equivalent or better training performance compared to the legacy stack
(ROS Foxy + PyTorch 1.10).

## Baselines

Pre-trained models included in `src/turtlebot3_drl/model/examples/`:

| Model | Algorithm | Stage | Episodes | Training Date |
|-------|-----------|-------|----------|---------------|
| `examples/ddpg_0_stage9` | DDPG | 9 | 8,186 | 2022-08-08 |
| `examples/td3_0_stage9` | TD3 | 9 | 7,856 | 2022-10-03 |

## Validation Experiments

### Experiment 1: DDPG Stage 4 (basic)

```bash
# Build and launch the new stack
docker compose build
docker compose up simulation environment gazebo-goals

# In a separate terminal (or add to compose)
docker compose run drl-agent \
    python -m drl_agent.drl_agent --mode train --algorithm ddpg
```

Train for 2,000 episodes. Checkpoint saved every 100 episodes.

### Experiment 2: TD3 Stage 4 (basic)

```bash
DRL_ALGORITHM=td3 docker compose up simulation environment gazebo-goals drl-agent
```

Train for 2,000 episodes.

### Experiment 3: DDPG Stage 9 (full comparison)

```bash
DRL_STAGE=9 docker compose up simulation environment gazebo-goals

DRL_STAGE=9 DRL_ALGORITHM=ddpg docker compose run drl-agent \
    python -m drl_agent.drl_agent --mode train --algorithm ddpg
```

Train for 8,000 episodes (matches baseline).

### Experiment 4: TD3 Stage 9 (full comparison)

Same as above with `DRL_ALGORITHM=td3`, train for 7,500 episodes.

## Running Validation

After training completes:

```bash
# Compare DDPG against baseline
python3 util/validate_convergence.py \
    --baseline examples/ddpg_0_stage9 \
    --candidate ddpg_0 \
    --interval 100

# Compare TD3 against baseline
python3 util/validate_convergence.py \
    --baseline examples/td3_0_stage9 \
    --candidate td3_0 \
    --interval 100

# Compare both at once
python3 util/validate_convergence.py \
    --baseline examples/ddpg_0_stage9 examples/td3_0_stage9 \
    --candidate ddpg_0 td3_0 \
    --interval 100
```

## Pass/Fail Criteria

The validation script checks:

| Check | Criterion | Default Tolerance |
|-------|-----------|-------------------|
| Minimum episodes | Candidate trained >= 500 episodes | N/A |
| Peak reward | Candidate peak >= 90% of baseline peak | 90% |
| Final avg reward | Candidate final window >= 90% of baseline | 90% |
| Peak success rate | Candidate peak SR >= 90% of baseline peak SR | 90% |

All four checks must pass for validation to succeed. Tolerance is configurable
via `--tolerance` flag.

## Output

The validation script generates:
- **Comparison plot** (`util/graphs/validation_<timestamp>.png`): Side-by-side reward and success rate curves
- **Text report** (`util/graphs/validation_<timestamp>.txt`): Detailed metrics and pass/fail for each check

## Checkpoint Persistence Across Restarts

Verify model checkpoints survive container restarts:

```bash
# Start training, let it run for ~200 episodes
docker compose up simulation environment gazebo-goals drl-agent

# Stop all containers
docker compose down

# Restart and continue training (model dir is volume-mounted)
docker compose up simulation environment gazebo-goals drl-agent
```

The model directory is volume-mounted at `./src/turtlebot3_drl/model:/drl_agent/model`,
so checkpoints persist on the host filesystem.

## Expected Timeline

| Experiment | Episodes | Est. Time (GPU) |
|-----------|----------|-----------------|
| DDPG Stage 4 | 2,000 | ~4-6 hours |
| TD3 Stage 4 | 2,000 | ~4-6 hours |
| DDPG Stage 9 | 8,000 | ~16-24 hours |
| TD3 Stage 9 | 7,500 | ~15-22 hours |

Times depend on GPU performance and simulation speed.
