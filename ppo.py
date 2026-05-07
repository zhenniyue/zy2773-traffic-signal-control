import os
import sys
import shutil
import warnings
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci
import traci.constants as tc
import ray
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.tune.registry import register_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks


# PATH SETUP

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
SIM_DIR = os.path.join(PROJECT_ROOT, "simulation")

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

try:
    from grid import Grid
except ImportError:
    class Grid:
        def __init__(self, grid_size, bounds): pass
        def get_stacked_observation(self): return np.zeros((4, 8, 8))


# SUMO BINARY DISCOVERY

# def find_sumo_binary(gui=False):
#     if gui:
#         candidates = ["/opt/homebrew/bin/sumo-gui", "/usr/local/bin/sumo-gui", "sumo-gui"]
#     else:
#         candidates = ["/opt/homebrew/bin/sumo", "/usr/local/bin/sumo", "sumo"]

#     for c in candidates:
#         if os.path.exists(c) or shutil.which(c):
#             return c
#     return "sumo-gui" if gui else "sumo" # Fallback

def find_sumo_binary(gui=False):
    # 你当前机器上的真实 GUI 可执行文件路径（已验证存在）
    app_gui = "/Applications/SUMO sumo-gui.app/Contents/MacOS/SUMO sumo-gui"

    if gui:
        candidates = [
            app_gui,
            "/opt/homebrew/bin/sumo-gui",
            "/usr/local/bin/sumo-gui",
            "sumo-gui",
        ]
    else:
        candidates = [
            # 临时先用 GUI binary 顶上，先把流程跑通（后续再换成真正 sumo）
            app_gui,
            "/opt/homebrew/bin/sumo",
            "/usr/local/bin/sumo",
            "sumo",
        ]

    for c in candidates:
        if os.path.exists(c) or shutil.which(c):
            print(f"[DEBUG] Using SUMO binary: {c}")
            return c

    raise FileNotFoundError("No SUMO binary found. Please check SUMO installation path.")

# CALLBACKS

class TrafficCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, episode, **kwargs):
        episode.user_data["collisions"] = 0
        episode.user_data["danger"] = 0
        episode.user_data["waiting"] = 0

    def on_episode_step(self, *, episode, base_env, **kwargs):
        info = episode.last_info_for()
        if info:
            episode.user_data["collisions"] += info.get("step_collisions", 0)
            episode.user_data["danger"] += info.get("danger_events", 0)
            episode.user_data["waiting"] += info.get("waiting", 0)

    def on_episode_end(self, *, episode, **kwargs):
        episode.custom_metrics["total_collisions"] = episode.user_data["collisions"]
        episode.custom_metrics["total_danger"] = episode.user_data["danger"]
        episode.custom_metrics["avg_waiting"] = episode.user_data["waiting"] / max(1, episode.length)


# ENVIRONMENT

class TrafficEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config):
        super().__init__()
        self.gui = config.get("gui", False)
        self.max_steps = config.get("max_steps", 1000)
        self.net_file = config["net_file"]
        self.route_file = config["route_file"]
        self.add_file = config["add_file"]

        self.grid = Grid(grid_size=(8, 8), bounds=(120, 120, 280, 280))
        self.total_green_time = 45.0

        # Action: Discrete actions representing 10%, 20%, ..., 90% split
        self.action_space = spaces.Discrete(9)  # 9 actions: 10% to 90%
        # Observation: Grid data
        self.observation_space = spaces.Box(0.0, 100.0, shape=(4 * 8 * 8,), dtype=np.float32)

        self.worker_id = f"sumo_{os.getpid()}"
        self.current_step = 0
        self.current_direction = 1
        self.sumo_running = False

    def _start_sumo(self):
        sumo_bin = find_sumo_binary(self.gui)
        # Port 0 lets TRACI find a free port automatically
        sumo_cmd = [
            sumo_bin,
            "-n", self.net_file,
            "-r", self.route_file,
            "-a", self.add_file,
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--time-to-teleport", "-1",
            "--step-length", "0.5",
            "--collision.action", "remove",
            "--collision.check-junctions", "true",
        ]

        try:
            traci.start(sumo_cmd, label=self.worker_id)
            traci.switch(self.worker_id)
            self.sumo_running = True
            self._setup_subscriptions()
        # except Exception as e:
            # Only raise if it's a real startup error, not a label collision
        #     if "socket" in str(e) or "connection" in str(e).lower():
        #         raise RuntimeError(f"SUMO failed to start: {e}")

#  Change:
        except Exception as e:
            print("[DEBUG] SUMO start failed:", repr(e))
            raise
    def _setup_subscriptions(self):
        traci.trafficlight.setProgram("center", "protective_permitted_paper")
        traci.trafficlight.setPhase("center", 2)

        traci.junction.subscribeContext(
            "center", tc.CMD_GET_VEHICLE_VARIABLE, 200.0,
            [tc.VAR_POSITION, tc.VAR_SPEED]
        )
        try:
            traci.poi.add("center_anchor", 200.0, 200.0, (255, 0, 0, 0))
            traci.poi.subscribeContext(
                "center_anchor", tc.CMD_GET_PERSON_VARIABLE, 200.0, [tc.VAR_POSITION]
            )
        except Exception:
            pass

    def _get_observation(self):
        obs = self.grid.get_stacked_observation()
        obs[2] /= 14.0
        return obs.flatten().astype(np.float32)

    def _get_metrics(self):
        collisions = 0
        danger = 0
        waiting = 0

        try:
            # Use len(getCollisions())
            collisions = len(traci.simulation.getCollisions())

            vehs = traci.junction.getContextSubscriptionResults("center") or {}
            peds = traci.poi.getContextSubscriptionResults("center_anchor") or {}

            if vehs and peds:
                v_pos = np.array([v[tc.VAR_POSITION] for v in vehs.values()])
                v_spd = np.array([v[tc.VAR_SPEED] for v in vehs.values()])
                p_pos = np.array([p[tc.VAR_POSITION] for p in peds.values()])
                
                moving = v_spd > 1.0
                if np.any(moving):
                    # Broadcasting distance check
                    dists = np.linalg.norm(p_pos[:, None, :] - v_pos[moving][None, :, :], axis=2)
                    danger = int(np.sum(dists < 5.0))

            for e in traci.edge.getIDList():
                waiting += traci.edge.getWaitingTime(e)

        except Exception:
            pass

        return collisions, danger, waiting

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if not self.sumo_running:
            self._start_sumo()
        else:
            try:
                traci.switch(self.worker_id)
                traci.load([
                    "-n", self.net_file,
                    "-r", self.route_file,
                    "-a", self.add_file,
                    "--no-warnings", "true",
                    "--no-step-log", "true",
                    "--time-to-teleport", "-1",
                    "--step-length", "0.5",
                    "--collision.action", "remove",
                    "--collision.check-junctions", "true"
                ])
                self._setup_subscriptions()
            except Exception:
                try: traci.close()
                except: pass
                self._start_sumo()

        self.current_step = 0
        self.current_direction = 1
        return self._get_observation(), {}

    def step(self, action):
        traci.switch(self.worker_id)

        # Map discrete action (0-8) to percentage (10%-90%)
        percentage = (action + 1) * 10  # 0→10%, 1→20%, ..., 8→90%
        protected = int(round(self.total_green_time * percentage / 100.0))
        permissive = int(round(max(5, self.total_green_time - protected)))

        if self.current_direction == 1:
            cycle = [(3, 3), (4, protected), (5, 3), (6, permissive)]
        else:
            cycle = [(7, 3), (0, protected), (1, 3), (2, permissive)]

        step_collisions = 0
        step_danger = 0
        step_waiting = 0

        for phase, dur in cycle:
            traci.trafficlight.setPhase("center", phase)
            # dur is seconds, step is 0.5s -> dur*2 steps
            for _ in range(dur * 2):
                if self.current_step >= self.max_steps: break
                traci.simulationStep()
                self.current_step += 1
                c, d, w = self._get_metrics()
                step_collisions += c
                step_danger += d
                step_waiting += w

        self.current_direction ^= 1

        reward = -10.0 * step_collisions - 1.0 * step_danger - 0.001 * step_waiting + 1.0
        terminated = self.current_step >= self.max_steps

        info = {
            "step_collisions": step_collisions,
            "danger_events": step_danger,
            "waiting": step_waiting,
            "protected_time": protected,
        }

        return self._get_observation(), reward, terminated, False, info

    def close(self):
        if self.sumo_running:
            try:
                traci.switch(self.worker_id)
                traci.close()
            except Exception:
                pass
            self.sumo_running = False


# MAIN EXECUTION

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--checkpoint", type=str)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    # Disable dashboard to prevent RPC errors on Mac
    ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level="ERROR")

    sim_config = {
        "net_file": os.path.join(SIM_DIR, "network.net.xml"),
        "route_file": os.path.join(SIM_DIR, "routes.rou.xml"),
        "add_file": os.path.join(SIM_DIR, "traffic_light.add.xml"),
        "max_steps": 1000,
    }

    register_env("TrafficEnv", lambda cfg: TrafficEnv(cfg))

    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment("TrafficEnv", env_config=sim_config)
        .framework("torch")
        .callbacks(TrafficCallbacks)
        .training(
            train_batch_size=1000, 
            lr=1e-4, 
            gamma=0.99
        )
        .env_runners(
            num_env_runners=1, 
            rollout_fragment_length=1000, 
            batch_mode="complete_episodes",
            sample_timeout_s=600.0
        )
        .resources(num_gpus=0, num_cpus_per_worker=1)
    )

    if args.mode == "train":
        import json

        print("\n=== Starting Training ===")
        print(f"{'Iter':>4} | {'Reward':>8} | {'Collisions':>10} | {'Danger':>8} | {'Avg Wait':>10} | {'Episodes':>8}")
        print("-" * 75)

        # Track metrics for plotting
        training_data = {
            "iterations": [],
            "rewards": [],
            "collisions": [],
            "danger": [],
            "avg_waiting": [],
            "episodes": []
        }

        algo = config.build()
        best_reward = float('-inf')

        for i in range(50):
            res = algo.train()

            # --- SAFE DATA ACCESS ---
            # Checks multiple locations for metrics to avoid crashes
            rew = res.get('episode_reward_mean')
            if rew is None:
                rew = res.get('env_runners', {}).get('episode_reward_mean')

            # Extract custom metrics
            custom_metrics = res.get('custom_metrics', {})
            if not custom_metrics:
                custom_metrics = res.get('env_runners', {}).get('custom_metrics', {})

            collisions = custom_metrics.get('total_collisions_mean', 0)
            danger = custom_metrics.get('total_danger_mean', 0)
            avg_wait = custom_metrics.get('avg_waiting_mean', 0)

            # Get episode count
            episodes = res.get('episodes_this_iter', res.get('env_runners', {}).get('num_episodes', 0))

            if rew is not None:
                print(f"{i+1:4d} | {rew:8.2f} | {collisions:10.1f} | {danger:8.1f} | {avg_wait:10.2f} | {episodes:8d}")

                # Save training data
                training_data["iterations"].append(i + 1)
                training_data["rewards"].append(float(rew))
                training_data["collisions"].append(float(collisions))
                training_data["danger"].append(float(danger))
                training_data["avg_waiting"].append(float(avg_wait))
                training_data["episodes"].append(int(episodes))

                # Save best model
                if rew > best_reward:
                    best_reward = rew
                    algo.save("./ppo_best_checkpoint")
                    print(f"     → New best model saved! (reward: {rew:.2f})")
            else:
                print(f"{i+1:4d} | {'--':>8} | {'--':>10} | {'--':>8} | {'--':>10} | {'--':>8}")

            if (i+1) % 10 == 0:
                algo.save("./ppo_checkpoints")
                print("-" * 75)

        # Save training results
        with open("training_results.json", "w") as f:
            json.dump(training_data, f, indent=2)
        print("\n Training results saved to training_results.json")

        algo.stop()
        
    else:
        if not args.checkpoint:
            print("Error: --checkpoint required for test mode.")
            sys.exit(1)
        
        algo = PPO.from_checkpoint(args.checkpoint)
        sim_config["gui"] = True
        env = TrafficEnv(sim_config)
        obs, _ = env.reset()
        done = False
        
        while not done:
            act = algo.compute_single_action(obs, explore=False)
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
            percentage = (act + 1) * 10
            print(f"Action: {act} ({percentage}%) | Info: {info}")
        
        env.close()

    ray.shutdown()