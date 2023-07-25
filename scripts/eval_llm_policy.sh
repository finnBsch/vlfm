#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.

python -um zsos.run \
  habitat_baselines.evaluate=True \
  habitat_baselines.eval_ckpt_path_dir=dummy_policy.pth \
  habitat_baselines.load_resume_state_config=False \
  habitat_baselines.rl.policy.name=LLMPolicy \
  habitat_baselines.rl.ddppo.reset_critic=False \
  habitat.simulator.habitat_sim_v0.allow_sliding=True \
  habitat.task.lab_sensors.base_explorer.turn_angle=30 \
  habitat_baselines.num_environments=1 \
  habitat_baselines.eval.video_option='["disk"]'
#  habitat_baselines.eval.video_option='[]'
#  habitat.environment.max_episode_steps=20 \  # Useful for debugging
