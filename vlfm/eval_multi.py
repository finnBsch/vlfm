# scipt to launch vlfm multi object eval similar to my own work
from eval import get_closest_dist
# from mapping.rerun_logger as rerun_logger
from config import EvalConf
from onemap_utils import monochannel_to_inferno_rgb
from eval.dataset_utils import *
from vlfm.policy.habitat_policies import HM3D_ID_TO_NAME

# os / filsystem
import bz2
import os
from os import listdir
import gzip
import json
import pathlib

import torch

# cv2
import cv2

# numpy
import numpy as np

# skimage
import skimage

# dataclasses
from dataclasses import dataclass

# quaternion
import quaternion

# typing
from typing import Tuple, List, Dict
import enum

# habitat
import habitat_sim
from habitat_sim import ActionSpec, ActuationSpec
from habitat_sim.utils import common as utils
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat.tasks.nav.nav import EpisodicGPSSensor, EpisodicCompassSensor, HeadingSensor


# tabulate
from tabulate import tabulate

# rerun
import rerun as rr

# pandas
import pandas as pd

# pickle
import pickle

import rerun as rr
SEQ_LEN = 4
# scipy
from scipy.spatial.transform import Rotation as R


class Result(enum.Enum):
    SUCCESS = 1
    FAILURE_MISDETECT = 2
    FAILURE_STUCK = 3
    FAILURE_OOT = 4
    FAILURE_NOT_REACHED = 5
    FAILURE_ALL_EXPLORED = 6

def process_depth(obs, min_depth_value=0.5, max_depth_value=5.0, normalize_depth=True):
    if isinstance(obs, np.ndarray):
        obs = np.clip(obs, min_depth_value, max_depth_value)

        obs = np.expand_dims(
            obs, axis=2
        )  # make depth observation a 3D array
    else:
        obs = obs.clamp(min_depth_value, max_depth_value)  # type: ignore[attr-defined, unreachable]

        obs = obs.unsqueeze(-1)  # type: ignore[attr-defined]

    if normalize_depth:
        # normalize depth observation to [0, 1]
        obs = (obs - min_depth_value) / (
            max_depth_value - min_depth_value
        )

    return obs


class Metrics:
    def __init__(self, ep_id) -> None:
        self.sequence_lengths = []
        self.sequence_results = []
        self.sequence_poses = []
        self.ep_id = ep_id
        self.sequence_object = []

    def add_sequence(self, sequence: np.ndarray, result: Result, target_object: str) -> None:
        start_id = 0
        if len(self.sequence_poses) > 0:
            start_id = sum([len(seq) for seq in self.sequence_poses])
        seq_poses = sequence[start_id:, :]
        self.sequence_poses.append(seq_poses)
        length = np.linalg.norm(seq_poses[1:, :2] - seq_poses[:-1, :2])
        self.sequence_results.append(result)
        self.sequence_lengths.append(length)
        self.sequence_object.append(target_object)

    def get_progress(self):
        return self.sequence_results.count(Result.SUCCESS) / SEQ_LEN


class SimAdapter:
    def __init__(self, sim):
        self.sim = sim

    def get_agent_state(self, agent_id=0):
        return self.sim.get_agent(agent_id).get_state()

class HabitatMultiEvaluator:
    def __init__(self,
                 config: EvalConf,
                 habitat_trainer,
                 ) -> None:
        self.habitat_trainer = habitat_trainer
        self.config = config
        self.multi_object = config.multi_object
        self.max_steps = config.max_steps
        self.max_dist = config.max_dist
        self.controller = config.controller
        self.mapping = config.mapping
        self.planner = config.planner
        self.log_rerun = config.log_rerun
        self.object_nav_path = config.object_nav_path
        self.scene_path = config.scene_path
        self.scene_data = {}
        self.episodes = []
        self.exclude_ids = []
        self.is_gibson = config.is_gibson

        self.sim = None

        if self.multi_object:
            self.episodes, self.scene_data = HM3DMultiDataset.load_hm3d_multi_episodes(self.episodes,
                                                                                       self.scene_data,
                                                                                       self.object_nav_path)
        else:
            raise RuntimeError("You are running the multi object evaluation with a single object config.")
        self.results_path = "/home/finn/active/MON/results_gibson_multi" if self.is_gibson else "/home/finn/active/MON/results_vlfm_4"
        self.class_map = {}
        self.class_map["chair"] = "chair"
        self.class_map["tv_monitor"] = "tv"
        self.class_map["tv"] = "tv"
        self.class_map["plant"] = "potted plant"
        self.class_map["potted plant"] = "potted plant"
        self.class_map["sofa"] = "couch"
        self.class_map["couch"] = "couch"
        self.class_map["bed"] = "bed"
        self.class_map["toilet"] = "toilet"
        rr.init("VLFM", spawn=False)
        rr.connect("127.0.0.1:9876")
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # Set an up-axis
        rr.log(
            "world/xyz",
            rr.Arrows3D(
                vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )

    def load_scene(self, scene_id: str):
        if self.sim is not None:
            self.sim.close()
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self.scene_path + scene_id

        backend_cfg.scene_dataset_config_file = self.scene_path + "hm3d/hm3d_annotated_basis.scene_dataset_config.json"

        hfov = 79
        rgb = habitat_sim.CameraSensorSpec()
        rgb.uuid = "rgb"
        rgb.hfov = hfov
        rgb.position = np.array([0, 0.88, 0])
        rgb.sensor_type = habitat_sim.SensorType.COLOR
        rgb.resolution = [480, 640]

        depth = habitat_sim.CameraSensorSpec()
        depth.uuid = "depth"
        depth.hfov = hfov
        depth.sensor_type = habitat_sim.SensorType.DEPTH
        depth.position = np.array([0, 0.88, 0])
        depth.resolution = [480, 640]
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        # TODO Fix depth sensor config according to what VLFM does. If not possible, rescale manuallu
        agent_cfg.sensor_specifications = [rgb, depth]
        sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(sim_cfg)
        if self.scene_data[scene_id].objects_loaded:
            return
        self.scene_data = HM3DDataset.load_hm3d_objects(self.scene_data, self.sim.semantic_scene.objects, scene_id)

    def execute_action(self, action: int
                       ):
        if action == 1:
            self.sim.step("move_forward")
        elif action == 2:
            self.sim.step("turn_left")
        elif action == 3:
            self.sim.step("turn_right")

    def read_results(self, path, sort_by):
        state_dir = os.path.join(path, 'state')
        state_results = {}


        # Check if the state directory exists
        if not os.path.isdir(state_dir):
            print(f"Error: {state_dir} is not a valid directory")
            return state_results
        pose_dir = os.path.join(os.path.abspath(os.path.join(state_dir, os.pardir)), "trajectories")

        # Iterate through all files in the state directory
        data = []
        for filename in os.listdir(state_dir):
            if filename.startswith('state_') and filename.endswith('.txt'):
                try:
                    # Extract the experiment number from the filename
                    experiment_num = int(filename[6:-4])  # removes 'state_' and '.txt'
                    # Read the content of the file
                    with open(os.path.join(state_dir, filename), 'r') as file:
                        content = file.read().strip()

                    # Convert the content to a number (assuming it's a float)
                    state_values = content.split(',')
                    state_values = [int(val) for val in state_values]
                    # Store the result in the dictionary
                    # Create a row for each sequence in the experiment
                    for seq_num, value in enumerate(state_values):
                        data.append({
                            'experiment': experiment_num,
                            'sequence': seq_num,
                            'state': value,
                            'object': self.episodes[experiment_num].obj_sequence[seq_num],
                            'scene': self.episodes[experiment_num].scene_id
                        })
                    poses = np.genfromtxt(os.path.join(pose_dir, "poses_" + str(experiment_num) + ".csv"),
                                          delimiter=",")
                    # deltas = poses[1:, :3] - poses[:-1, :3]
                    # distance_traveled = np.linalg.norm(deltas, axis=1).sum()
                    # if state_value == 1:
                    #     spl[experiment_num] = self.episodes[experiment_num].best_dist / max(
                    #         self.episodes[experiment_num].best_dist, distance_traveled)
                    # else:
                    #     spl[experiment_num] = 0
                    if self.episodes[experiment_num].episode_id != experiment_num:
                        print(
                            f"Warning, experiment_num {experiment_num} does not correctly resolve to episode_id {self.episodes[experiment_num].episode_id}")
                except ValueError:
                    print(f"Warning: Skipping {filename} due to invalid format")
                except Exception as e:
                    print(f"Error reading {filename}: {str(e)}")
        data = pd.DataFrame(data)

        states = data["state"].unique()

        def has_success(group, seq_id):
            # Check if there's an entry with seq_id = 0 and success state
            return group[(group['sequence'] == seq_id) & (group['state'] == 1)].shape[0] > 0
        def calc_per_episode(group):
            num_sequences = SEQ_LEN
            successes = group.groupby('experiment').apply(lambda x: (x['state'] == 1).sum())
            progress = successes / num_sequences
            return progress
        def calculate_percentages(group):
            total = len(group)
            result = pd.Series({Result(state).name: (group['state'] == state).sum() / total for state in states})
            progress = calc_per_episode(group)
            result['Progress'] = progress.mean()

            # Calculate average SPL and multiply by 100
            # avg_spl = group['spl'].mean()
            # result['Average SPL'] = avg_spl

            return result

        # Per-object results
        object_results = data.groupby('object').apply(calculate_percentages).reset_index()
        object_results = object_results.rename(columns={'object': 'Object'})

        # Per-scene results
        scene_results = data.groupby('scene').apply(calculate_percentages).reset_index()
        scene_results = scene_results.rename(columns={'scene': 'Scene'})

        # Overall results
        overall_percentages = calculate_percentages(data)
        overall_row = pd.DataFrame([{'Object': 'Overall'} | overall_percentages.to_dict()])
        object_results = pd.concat([overall_row, object_results], ignore_index=True)

        overall_row = pd.DataFrame([{'Scene': 'Overall'} | overall_percentages.to_dict()])
        scene_results = pd.concat([overall_row, scene_results], ignore_index=True)

        # Sorting
        object_results = object_results.sort_values(by=sort_by, ascending=False)
        scene_results = scene_results.sort_values(by=sort_by, ascending=False)

        # Function to format percentages
        def format_percentages(val):
            return f"{val:.2%}" if isinstance(val, float) else val

        # Apply formatting to all columns except the first one (Object/Scene)
        object_table = object_results.iloc[:, 0].to_frame().join(
            object_results.iloc[:, 1:].applymap(format_percentages))
        scene_table = scene_results.iloc[:, 0].to_frame().join(
            scene_results.iloc[:, 1:].applymap(format_percentages))

        print(f"Results by Object (sorted by {sort_by} rate, descending):")
        print(tabulate(object_table, headers='keys', tablefmt='pretty', floatfmt='.2%'))

        print(f"\nResults by Scene (sorted by {sort_by} rate, descending):")
        print(tabulate(scene_table, headers='keys', tablefmt='pretty', floatfmt='.2%'))

        exp_data = data.groupby('experiment')
        all_ids = exp_data['experiment'].unique()
        successful_experiments = exp_data.filter(lambda x: has_success(x, 0))
        selected_experiment_ids = successful_experiments['experiment'].unique()

        experiments_with_second_success = successful_experiments.groupby('experiment').filter(
            lambda x: has_success(x, 1))
        successful_second_ids = experiments_with_second_success['experiment'].unique()
        fraction_successful = len(successful_second_ids) / len(selected_experiment_ids) if len(
            selected_experiment_ids) > 0 else 0
        print(f"Fraction of successful first experiments: {len(selected_experiment_ids)/len(all_ids):.2%}")
        print(f"Fraction of successful second, conditioned on first: {fraction_successful:.2%}")
        return data

    def evaluate(self):
        n_eps = 0
        device = torch.device("cuda")  # type: ignore
        action_shape, discrete_actions = get_action_space_info(self.habitat_trainer._agent.policy_action_space)

        results = []
        self.habitat_trainer._agent.eval()
        for n_ep, episode in enumerate(self.episodes[41:]):
            gps_sensor_conf = {'uuid': 'gps'}
            compass_sensor_conf = {'uuid': 'compass'}
            heading_sensor_conf = {'uuid': 'heading'}

            poses = []
            metric = Metrics(episode.episode_id)
            results.append(metric)
            if n_ep in self.exclude_ids:
                continue
            n_eps += 1
            if self.sim is None or not self.sim.curr_scene_name in episode.scene_id:
                self.load_scene(episode.scene_id)

            self.sim.initialize_agent(0, habitat_sim.AgentState(episode.start_position, episode.start_rotation))
            sim_adapter = SimAdapter(self.sim)
            gps_sensor = EpisodicGPSSensor(sim=sim_adapter, config=gps_sensor_conf)
            compass_sensor = EpisodicCompassSensor(sim=sim_adapter, config=compass_sensor_conf)
            heading_sensor = HeadingSensor(sim=sim_adapter, config=heading_sensor_conf)
            sequence_id = 0
            current_obj = episode.obj_sequence[sequence_id]

            not_failed = True
            while not_failed and sequence_id < len(episode.obj_sequence):
                steps = 0
                running = True
                test_recurrent_hidden_states = torch.zeros(
                    (
                        1,
                        *self.habitat_trainer._agent.hidden_state_shape,
                    ),
                    device=device,
                )
                prev_actions = torch.zeros(
                    1,
                    *action_shape,
                    device=device,
                    dtype=torch.long,
                )
                not_done_masks = torch.zeros(
                    1,
                    1,
                    device=device,
                    dtype=torch.bool,
                )
                while steps < self.max_steps and running:
                    observations = self.sim.get_sensor_observations()
                    obs_vlfm = {}
                    obs_vlfm['rgb'] = observations['rgb']
                    obs_vlfm['depth'] = process_depth(observations['depth'])


                    observations['state'] = self.sim.get_agent(0).get_state()
                    pose = np.zeros((4,))
                    pose[0] = -observations['state'].position[2]
                    pose[1] = -observations['state'].position[0]
                    pose[2] = observations['state'].position[1]
                    # gps is -position[2], position[0],
                    # according to https://github.com/facebookresearch/habitat-lab/blob/604e6382a9e6955fc1fb57b2853699088343d081/habitat-lab/habitat/tasks/nav/nav.py#L435
                    obs_vlfm['gps'] = gps_sensor.get_observation(None, episode)
                    obs_vlfm['compass'] = compass_sensor.get_observation(None, episode)
                    obs_vlfm['heading'] = heading_sensor.get_observation(None, episode)
                    obs_vlfm['objectgoal'] = np.array([HM3D_ID_TO_NAME.index(self.class_map[current_obj])])

                    batch = batch_obs([obs_vlfm], device=device)

                    poses.append(pose)
                    if self.log_rerun:
                        cam_x = -self.sim.get_agent(0).get_state().position[2]
                        cam_y = -self.sim.get_agent(0).get_state().position[0]
                        rr.log("camera/rgb", rr.Image(observations["rgb"][:, :, :3]).compress(jpeg_quality=50))
                        markers = []

                        # Draw frontiers on to the cost map
                        frontiers = self.habitat_trainer._agent.actor_critic._obstacle_map.frontiers
                        for frontier in frontiers:
                            marker_kwargs = {
                                "radius": self.habitat_trainer._agent.actor_critic._circle_marker_radius,
                                "thickness": self.habitat_trainer._agent.actor_critic._circle_marker_thickness,
                                "color": self.habitat_trainer._agent.actor_critic._frontier_color,
                            }
                            markers.append((frontier[:2], marker_kwargs))

                        if not np.array_equal(self.habitat_trainer._agent.actor_critic._last_goal, np.zeros(2)):
                            # Draw the pointnav goal on to the cost map
                            if any(np.array_equal(self.habitat_trainer._agent.actor_critic._last_goal, frontier) for frontier in frontiers):
                                color = self.habitat_trainer._agent.actor_critic._selected__frontier_color
                            else:
                                color = self.habitat_trainer._agent.actor_critic._target_object_color
                            marker_kwargs = {
                                "radius": self.habitat_trainer._agent.actor_critic._circle_marker_radius,
                                "thickness": self.habitat_trainer._agent.actor_critic._circle_marker_thickness,
                                "color": color,
                            }
                            markers.append((self.habitat_trainer._agent.actor_critic._last_goal, marker_kwargs))
                        vlfm_map = self.habitat_trainer._agent.actor_critic._value_map.visualize(markers)
                        # Add obstacle map!
                        obstacle_map = self.habitat_trainer._agent.actor_critic._obstacle_map.visualize()
                        rr.log("vlfm_map", rr.Image(np.flip(vlfm_map, axis=-1)))
                        rr.log("obstacle_map", rr.Image(obstacle_map))
                        # rr.log("camera/depth", rr.Image((observations["depth"] - observations["depth"].min()) / (
                        #         observations["depth"].max() - observations["depth"].min())))
                        # self.logger.log_pos(cam_x, cam_y)
                    with torch.inference_mode():
                        action_data = self.habitat_trainer._agent.actor_critic.act(
                            batch,
                            test_recurrent_hidden_states,
                            prev_actions,
                            not_done_masks,
                            deterministic=False,
                        )
                        not_done_masks = torch.ones(
                            1,
                            1,
                            device=device,
                            dtype=torch.bool,
                        )
                    if action_data.should_inserts is None:
                        test_recurrent_hidden_states = action_data.rnn_hidden_states
                        prev_actions.copy_(action_data.actions)  # type: ignore
                    else:
                        for i, should_insert in enumerate(action_data.should_inserts):
                            if should_insert.item():
                                test_recurrent_hidden_states[i] = action_data.rnn_hidden_states[i]
                                prev_actions[i].copy_(action_data.actions[i])  # type: ignore
                    action = action_data.actions[0,0].item()
                    if steps % 100 == 0:
                        dist = get_closest_dist(self.sim.get_agent(0).get_state().position[[0, 2]],
                                                self.scene_data[episode.scene_id].object_locations[current_obj],
                                                self.is_gibson)
                        print(
                            f"Step {steps}, current object: {current_obj}, episode_id: {episode.episode_id}, distance to closest object: {dist}")
                    steps += 1
                    if action != 0 and steps < self.max_steps:
                        self.execute_action(action)
                    # if self.log_rerun:
                    #     self.logger.log_map()
                    else:
                        final_sim = monochannel_to_inferno_rgb(self.habitat_trainer._agent.actor_critic._value_map._value_map)
                        cv2.imwrite(f"{self.results_path}/similarities/final_sim_{episode.episode_id}_{sequence_id}.png", final_sim)
                        not_done_masks = torch.zeros(
                            1,
                            1,
                            device=device,
                            dtype=torch.bool,
                        )
                        running = False
                        result = Result.FAILURE_OOT
                        # We will now compute the closest distance to the bounding box of the object
                        if action == 0:
                            dist = get_closest_dist(self.sim.get_agent(0).get_state().position[[0, 2]],
                                                    self.scene_data[episode.scene_id].object_locations[current_obj],
                                                    self.is_gibson)
                            if dist < self.max_dist:
                                result = Result.SUCCESS
                                print("Object found!")
                            else:
                                not_failed = False
                                result = Result.FAILURE_MISDETECT
                                print(f"Object not found! Dist {dist}.")
                        else:
                            not_failed = False
                            if result == Result.FAILURE_OOT and np.linalg.norm(poses[-1][:2] - poses[-10][:2]) < 0.05:
                                result = Result.FAILURE_STUCK
                        results[-1].add_sequence(np.array(poses), result, current_obj)
                        sequence_id += 1
                        if sequence_id < len(episode.obj_sequence):
                            current_obj = episode.obj_sequence[sequence_id]
                            # self.habitat_trainer._agent.actor_critic._reset()  # TODO check this reset

            for seq_id, seq in enumerate(results[n_ep].sequence_poses):
                np.savetxt(f"{self.results_path}/trajectories/poses_{episode.episode_id}_{seq_id}.csv", seq,
                           delimiter=",")
            # save final sim to image file

            print(f"Overall progress: {sum([m.get_progress() for m in results]) / (n_eps)}, per object: ")
            # for obj in success_per_obj.keys():
            #     print(f"{obj}: {success_per_obj[obj] / obj_count[obj]}")
            # print(
            #     f"Result distribution: successes: {results.count(Result.SUCCESS)}, misdetects: {results.count(Result.FAILURE_MISDETECT)}, OOT: {results.count(Result.FAILURE_OOT)}, stuck: {results.count(Result.FAILURE_STUCK)}, not reached: {results.count(Result.FAILURE_NOT_REACHED)}, all explored: {results.count(Result.FAILURE_ALL_EXPLORED)}")
            # Write result to file
            with open(f"{self.results_path}/state/state_{episode.episode_id}.txt", 'w') as f:
                f.write(','.join(
                    str(results[n_ep].sequence_results[i].value) for i in range(len(results[n_ep].sequence_results))))
