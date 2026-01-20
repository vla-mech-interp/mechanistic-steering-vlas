"""Main evaluation loop for LIBERO with optional neuron interventions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tqdm
from libero.libero import benchmark

from libero_experiments.config import RunConfig
from libero_experiments.hooks import apply_gate_proj_hooks
from libero_experiments.interventions import load_intervention_dict
from libero_experiments.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from libero_experiments.logging_utils import (
    append_csv_row,
    create_run_dir,
    get_run_id,
    open_log_file,
    save_actions_json,
    write_csv_header,
)
from libero_experiments.model import get_action, get_processor, load_model
from libero_experiments.utils import (
    get_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class EvalResult:
    run_dir: str
    success_rate: float


def eval_libero(cfg: RunConfig, intervention_config_path: str) -> EvalResult:
    set_seed_everywhere(cfg.env.seed)
    cfg_unnorm_key = cfg.env.task_suite_name

    model = load_model(cfg)
    if cfg.model.family == "openvla":
        if hasattr(model, "norm_stats"):
            if cfg_unnorm_key not in model.norm_stats and f"{cfg_unnorm_key}_no_noops" in model.norm_stats:
                cfg_unnorm_key = f"{cfg_unnorm_key}_no_noops"
            assert cfg_unnorm_key in model.norm_stats, (
                f"Action un-norm key {cfg_unnorm_key} not found in model norm stats."
            )

    processor = get_processor(cfg) if cfg.model.family == "openvla" else None

    intervention_name = cfg.intervention.dict_name if cfg.intervention.enabled else "blank"
    run_id = get_run_id(cfg.env.task_suite_name, cfg.model.family, intervention_name, cfg.intervention.coef)
    run_dir = create_run_dir(cfg.logging.root_dir, run_id)

    log_path = open_log_file(run_dir)
    log_file = open(log_path, "w")
    print(f"Logging to local log file: {log_path}")
    log_file.write(f"Logging to local log file: {log_path}\n")

    if cfg.intervention.enabled:
        intervention_dict = load_intervention_dict(cfg.intervention.dict_name, intervention_config_path)
        hooks = apply_gate_proj_hooks(model, intervention_dict, coef=cfg.intervention.coef)
        log_file.write(f"Intervention dict: {cfg.intervention.dict_name}\n")
        log_file.write(f"Intervention coef: {cfg.intervention.coef}\n")
    else:
        hooks = []

    csv_path = os.path.join(run_dir, "events.csv")
    write_csv_header(csv_path)

    actions_path = os.path.join(run_dir, "actions.json")
    all_actions_by_task = {}

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.env.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.env.task_suite_name}")
    log_file.write(f"Task suite: {cfg.env.task_suite_name}\n")

    resize_size = get_resize_size(cfg.model.family)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, resolution=256)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.env.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            if cfg.env.task_suite_name == "libero_spatial":
                max_steps = 220
            elif cfg.env.task_suite_name == "libero_object":
                max_steps = 280
            elif cfg.env.task_suite_name == "libero_goal":
                max_steps = 300
            elif cfg.env.task_suite_name == "libero_10":
                max_steps = 520
            elif cfg.env.task_suite_name == "libero_90":
                max_steps = 400
            else:
                raise ValueError("Unexpected task suite")

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")

            current_episode_actions = []

            while t < max_steps + cfg.env.num_steps_wait:
                try:
                    if t < cfg.env.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action())
                        t += 1
                        continue

                    img = get_libero_image(obs, resize_size)
                    replay_images.append(img)

                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    action = get_action(
                        model,
                        processor,
                        cfg,
                        observation,
                        task_description,
                        unnorm_key=cfg_unnorm_key,
                    )

                    action = normalize_gripper_action(action, binarize=True)
                    if cfg.model.family == "openvla":
                        action = invert_gripper_action(action)

                    obs, reward, done, info = env.step(action.tolist())
                    current_episode_actions.append(action.tolist())

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as exc:
                    print(f"Caught exception: {exc}")
                    log_file.write(f"Caught exception: {exc}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            if task_description not in all_actions_by_task:
                all_actions_by_task[task_description] = {}
            all_actions_by_task[task_description][episode_idx] = current_episode_actions
            if cfg.logging.save_actions:
                save_actions_json(actions_path, all_actions_by_task)

            if cfg.logging.save_video:
                save_rollout_video(
                    replay_images,
                    total_episodes,
                    success=done,
                    task_description=task_description,
                    out_dir=os.path.join(run_dir, "videos"),
                    log_file=log_file,
                )

            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        task_success_rate = float(task_successes) / float(task_episodes)
        print(f"Current task success rate: {task_success_rate}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {task_success_rate}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()

        append_csv_row(csv_path, task_description, task_success_rate)

    if cfg.logging.save_actions:
        save_actions_json(actions_path, all_actions_by_task)

    for hook in hooks:
        hook.remove()

    log_file.close()
    return EvalResult(run_dir=run_dir, success_rate=float(total_successes) / float(total_episodes))
