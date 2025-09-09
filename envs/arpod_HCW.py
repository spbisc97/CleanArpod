"""
Minimal HCW-based SE2 rendezvous environment (Gymnasium)
--------------------------------------------------------
- Planar Clohessy–Wiltshire (HCW) linearized relative motion around a circular orbit.
- State: [x, y, vx, vy, theta, omega]
    x, y     : LVLH position of the chaser relative to target [m]
    vx, vy   : LVLH velocity [m/s]
    theta    : chaser body yaw angle w.r.t. LVLH +x axis [rad]
    omega    : yaw rate [rad/s]
- Action: [u_thrust, u_alpha]
    u_thrust : normalized thrust command in [0, 1] (forward body thruster)
    u_alpha  : normalized rotational acceleration in [-1, 1]

Dynamics
========
Translational HCW (planar):
    x_ddot =  3 n^2 x + 2 n vy + a_x
    y_ddot = -2 n vx       + a_y
where (a_x, a_y) is the commanded body-forward acceleration rotated into LVLH:
    a_t = (u_thrust * T_max) / m
    [a_x, a_y] = a_t * [cos(theta), sin(theta)]

Attitude (simple integrator):
    omega_dot = u_alpha * alpha_max

Integration uses semi-implicit Euler for translational and rotational states.

Reward (shaping):
    r = - (w_p * ||p|| + w_v * ||v|| + w_theta * |theta| + w_u * (u_thrust^2 + u_alpha^2))
Success when within position/velocity/attitude tolerances simultaneously.

This is a tiny, clean starting point. No reaction-wheel model, no actuator lag.
You can train with SAC from Stable-Baselines3.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def wrap_pi(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    a = (a + math.pi) % (2 * math.pi) - math.pi
    return a


def semi_implicit_sat_dynamics(
    state: np.ndarray,
    u: np.ndarray,
    dt: float,
    n: float,
    m: float,
    T_max: float,
    alpha_max: float,
) -> np.ndarray:    
    """Semi-implicit Euler integration of HCW + attitude dynamics with saturation."""
    x, y, vx, vy, theta, omega = state
    u_thrust = float(np.clip(u[0], 0.0, 1.0))
    u_alpha = float(np.clip(u[1], -1.0, 1.0))

    # Compute commanded accelerations
    a_t = (u_thrust * T_max) / m  # [m/s^2]
    ax = a_t * math.cos(theta)
    ay = a_t * math.sin(theta)

    # HCW planar dynamics
    x_ddot = 3.0 * n**2 * x + 2.0 * n * vy + ax
    y_ddot = -2.0 * n * vx + ay

    # Semi-implicit Euler integration (symplectic-like for better stability)
    vx = vx + x_ddot * dt
    vy = vy + y_ddot * dt
    x = x + vx * dt
    y = y + vy * dt

    # Attitude dynamics
    omega_dot = u_alpha * alpha_max
    omega = omega + omega_dot * dt
    theta = wrap_pi(theta + omega * dt)

    new_state = np.array([x, y, vx, vy, theta, omega], dtype=np.float64)
    return new_state



class HCWSE2Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        n: float = 0.0011,          # [rad/s] mean motion (LEO ~ 90 min period)
        dt: float = 1.0,            # [s] integration step
        m: float = 8.0,             # [kg] small CubeSat mass
        T_max: float = 0.1,        # [N] max thrust (cold-gas class)
        alpha_max: float = 0.02,    # [rad/s^2] max rotational acceleration
        pos_limit: float = 500.0,  # [m] keep-out bound (episode terminates if exceeded)
        v_limit: float = 50.0,      # [m/s]
        theta_limit: float = math.pi,  # [rad]
        omega_limit: float = 10.0,  # [rad/s] (not enforced)
        max_steps: int = 3000,
        # Reward weights
        w_p: float = 0.1,
        w_p_prev: float = 1.0,
        w_v: float = 0.01,
        w_theta: float = 0.02,
        w_omega: float = 1,
        w_u: float = 0.1,
        # Success tolerances
        pos_tol: float = 2.0,       # [m]
        vel_tol: float = 0.05,      # [m/s]
        theta_tol: float = math.radians(2.0),  # [rad]
        spawn_radius: Tuple[float, float] = (5.0, 50.0),  # [m] min/max spawn distance
        spawn_angle: Tuple[float, float] = (1.5, 1.7),  # [rad] min/max spawn angle
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        render_folder: Optional[str] = None,
    ):
        super().__init__()
        self.n = float(n)
        self.dt = float(dt)
        self.m = float(m)
        self.T_max = float(T_max)
        self.alpha_max = float(alpha_max)
        self.pos_limit = float(pos_limit)
        self.v_limit = float(v_limit)
        self.theta_limit = float(theta_limit)
        self.omega_limit = float(omega_limit)
        
        
        self.max_steps = int(max_steps)
        # Reward weights
        self.w_p = float(w_p)
        self.w_p_prev = float(w_p_prev)
        self.w_v = float(w_v)
        self.w_theta = float(w_theta)
        self.w_omega = float(w_omega)
        self.w_u = float(w_u)
        
        self.pos_tol = float(pos_tol)
        self.vel_tol = float(vel_tol)
        self.theta_tol = float(theta_tol)
        self.spawn_radius = (float(spawn_radius[0]), float(spawn_radius[1]))
        self.spawn_angle = (float(spawn_angle[0]), float(spawn_angle[1]))
        self.render_mode = render_mode
        self.render_folder = render_folder  # if not None, save trajectory images here

        # RNG
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # State: x, y, vx, vy, theta, omega
        high = np.array([
            self.pos_limit, self.pos_limit,
            self.v_limit, self.v_limit,
            self.theta_limit, 5.0  # omega rough bound
        ], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Action: [u_thrust, u_alpha]
        #   u_thrust in [0, 1]
        #   u_alpha  in [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Buffers
        self.state = np.zeros(6, dtype=np.float64)
        self.steps = 0
        self.history = []  # for potential future use
        self.action_history = []  # for potential future use
        self.reward_history = []  # for potential future use
        self._last_info: Dict[str, Any] = {}
        self.reset(seed=seed)

    # ----------------------
    # Gym API
    # ----------------------
    
    
    def spawn_stable_state(self):
        """Generate a random initial state within spawn parameters."""
        r_min, r_max = self.spawn_radius
        r = self.np_random.uniform(r_min, r_max)
        phi = math.pi/2
        x0, y0 = r * math.cos(phi), r * math.sin(phi)
        vx0 = self.n * self.n * x0 / 2.0 + self.np_random.uniform(-0.00001, 0.0001)
        vy0 = self.np_random.uniform(-0.00001, 0.00001)
        theta0 = self.np_random.uniform(-math.pi, math.pi)
        omega0 = self.np_random.uniform(-0.001, 0.001)
        return np.array([x0, y0, vx0, vy0, theta0, omega0], dtype=np.float64)
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
            
        if self.render_mode == "rgb_array" and self.render_folder is not None:
            self.render()

        # Random spawn at a ring [r_min, r_max], small random velocity, random attitude
        x0, y0, vx0, vy0, theta0, omega0 = self.spawn_stable_state()

        self.state = np.array([x0, y0, vx0, vy0, theta0, omega0], dtype=np.float64)
        self.steps = 0
        self._last_info = {"success": False, "done_reason": "reset"}
        self.history = [self.state.copy()]
        self.action_history = [np.array([0.0, 0.0], dtype=np.float32)]
        self.reward_history = [np.float32(0.0)]

        obs = self.state.astype(np.float32)
        return obs, self._last_info.copy()

    def step(self, action: np.ndarray):
        self.steps += 1
        u = np.asarray(action, dtype=np.float64).copy()
        # Clip action to bounds
        u_thrust = float(np.clip(u[0], 0.0, 1.0))
        u_alpha = float(np.clip(u[1], -1.0, 1.0))
        previous_state = self.state.copy()

        self.state = semi_implicit_sat_dynamics(
            state=self.state,
            u=np.array([u_thrust, u_alpha], dtype=np.float64),
            dt=self.dt,
            n=self.n,
            m=self.m,
            T_max=self.T_max,
            alpha_max=self.alpha_max,
        )
        x, y, vx, vy, theta, omega = self.state
        
        # Observations

        obs = self.state.astype(np.float32)
        # print(np.array(self.history)[:,0])
        # print(np.array(self.action_history)[:,0])

        # Reward shaping
        p_norm = math.hypot(x, y)
        p_prev = math.hypot(previous_state[0], previous_state[1])
        v_norm = math.hypot(vx, vy)

        reward = 0.0

        reward_shaping =  self.w_p_prev*(p_prev - p_norm)  # reward for getting closer in position
        
        
        reward += reward_shaping
        reward -= self.w_v * v_norm
        reward -= self.w_omega * abs(omega)
        reward -= self.w_u * u_thrust
        reward -= self.w_u * abs(u_alpha)

        # Termination conditions
        terminated = False
        success = False
        truncated = False  # we use 'terminated' only here
        if (p_norm < self.pos_tol) and (v_norm < self.vel_tol) and (abs(theta) < self.theta_tol):
            terminated = True
            success = True
            done_reason = "success"
            
        # Failure cases
        elif (p_norm < self.pos_tol) and (v_norm < self.vel_tol):
            # close enough in position and velocity, but attitude not aligned
            terminated = True
            success = False
            done_reason = "attitude_fail"
        elif (p_norm < self.pos_tol) and (abs(theta) < self.theta_tol):
            # close enough in position and attitude, but velocity not aligned
            terminated = True
            success = False
            done_reason = "velocity_fail"
        elif (p_norm < self.pos_tol):
            # close enough in position, but attitude and velocity not aligned
            terminated = True
            success = False
            done_reason = "velocity_omega_fail"
        elif abs(omega) > self.omega_limit:
            # not enforced, but we can terminate if it runs away
            terminated = True
            success = False
            done_reason = "omega_fail"
        elif (abs(x) > self.pos_limit) or (abs(y) > self.pos_limit) or (abs(vx) > self.v_limit) or (abs(vy) > self.v_limit) or (abs(theta) > self.theta_limit):
            terminated = True
            done_reason = "out_of_bounds"
        elif self.steps >= self.max_steps:
            terminated = False
            truncated = True
            done_reason = "timeout"
        else:
            done_reason = "continue"
            
            
        # add rewards for staying alive
        if terminated and not success:
            reward -= 100.0  # big penalty for failure
            
        if not terminated and not truncated:
            reward += 0.1  # small reward for each step alive

        if terminated and success:
            reward += 200.0
        
        
        
        info = {
            "is_success": success,
            "done_reason": done_reason,
            "p_norm": p_norm,
            "v_norm": v_norm,
            "u_thrust": u_thrust,
            "u_alpha": u_alpha,
        }
        
        self._last_info = info
        self.history.append(self.state.copy())
        self.action_history.append(np.array([u_thrust, u_alpha], dtype=np.float32))
        self.reward_history.append(reward)
        
        return obs, reward, terminated, truncated, info

    def _render_text(self):
        x, y, vx, vy, theta, omega = self.state
        print(
            f"t={self.steps*self.dt:6.1f}s | r=({x:7.2f},{y:7.2f}) m | v=({vx:6.3f},{vy:6.3f}) m/s | "
            f"theta={theta:+6.3f} rad | omega={omega:+6.3f} rad/s"
        )



    def trajectory_image(self, history, action_history):

        """Generate a trajectory plot as an RGB array."""
        
        # print(f"done reason: {self._last_info.get('done_reason', 'N/A')}, history length: {len(history)}")
        from matplotlib import pyplot as plt
        
        if self._last_info.get('done_reason', 'N/A') != 'continue' and len(history) > 1:
            from matplotlib.patches import Circle

            # make plots of trajectory , x-t, y-t, vx-t, vy-t, theta-t, omega-t. make a 6x1 grid of subplots
            fig, axs = plt.subplots(9, 1, figsize=(6, 12), sharex=True)
            t = np.arange(len(history)) * self.dt
            history = np.array(history)
            action_history = np.array(action_history)
            reward_history = np.array(self.reward_history)
            x = history[:, 0]
            y = history[:, 1]
            vx = history[:, 2]
            vy = history[:, 3]
            theta = history[:, 4]
            omega = history[:, 5]
            trust = action_history[:, 0]
            alpha = action_history[:, 1]
            # reward = reward_history
            
            # Trajectory plot
            axs[0].plot(t, x, label="x")
            axs[1].plot(t, y, label="y")
            axs[2].plot(t, vx, label="vx")
            axs[3].plot(t, vy, label="vy")
            axs[4].plot(t, theta, label="theta")
            axs[5].plot(t, omega, label="omega")
            axs[6].plot(t, trust, label="trust")
            axs[7].plot(t, alpha, label="alpha")
            axs[8].plot(t, reward_history, label="reward")
            axs[0].set_title(f"Trajectory History,{self._last_info.get('done_reason', 'N/A')}")
            for ax in axs:
                ax.legend()
                ax.set_ylabel("Value")
            axs[-1].set_xlabel("Time (s)")
            plt.tight_layout()
            fig.canvas.draw()
            # Convert to RGB array
            canvas = fig.canvas
            w, h = canvas.get_width_height()
            rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
            img = rgba[:, :, :3].copy()
            plt.close(fig)
            
            # Save image if folder specified
            # print(f"Render mode: {self.render_mode}, Render folder: {self.render_folder}, Done reason: {self._last_info.get('done_reason', 'N/A')}")
            if self.render_folder is not None:
                import os
                if not os.path.exists(self.render_folder):
                    os.makedirs(self.render_folder)
                img_id = len(os.listdir(self.render_folder))
                img_path = os.path.join(self.render_folder, f"trajectory_{img_id:04d}.png")
                from PIL import Image
                im = Image.fromarray(img)
                im.save(img_path)
                print(f"Saved trajectory image to {img_path}")
        else:
            # return blank image
            img = np.zeros((480, 640, 3), dtype=np.uint8)
        return img
    
        
    def render(self):
        # print("Render called." + f" Mode: {self.render_mode}" + f", Folder: {self.render_folder}" + f", Done Reason: {self._last_info.get('done_reason', 'N/A')}") 
        if self.render_mode == "human":
            self._render_text()
        #return traj if is finished and rgb_array
        elif self.render_mode == "rgb_array":
            return self.trajectory_image(self.history, self.action_history)
        return None

    def close(self):
        pass


# ----------------------------
# Quick smoke test (no SB3)
# ----------------------------
if __name__ == "__main__":
    env = HCWSE2Env(render_mode="rgb_array", render_folder="./trajectories")
    obs, info = env.reset(seed=0)
    print("Initial:", obs, info)
    # simple heuristic: thrust toward -position bearing, damp rotation toward 0
    for _ in range(3000):
        x, y, vx, vy, theta, omega = obs
        # aim body toward target (0,0)
        desired_theta = math.atan2(-y, -x)
        theta_err = wrap_pi(desired_theta - theta)
        # PD on attitude to create rotational acceleration command
        u_alpha =0
        # thrust proportional to distance, capped
        dist = math.hypot(x, y)
        u_thrust = 0
        action = np.array([u_thrust, u_alpha], dtype=np.float32)
        obs, reward, term, trunc, info = env.step(action)
        env.render()
        if term or trunc:
            print("Terminated:", info)
            env.reset()
            break
