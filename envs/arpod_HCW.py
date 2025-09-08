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


class HCWSE2Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        n: float = 0.0011,          # [rad/s] mean motion (LEO ~ 90 min period)
        dt: float = 1.0,            # [s] integration step
        m: float = 8.0,             # [kg] small CubeSat mass
        T_max: float = 0.2,        # [N] max thrust (cold-gas class)
        alpha_max: float = 0.02,    # [rad/s^2] max rotational acceleration
        pos_limit: float = 1000.0,  # [m] keep-out bound (episode terminates if exceeded)
        v_limit: float = 50.0,      # [m/s]
        theta_limit: float = math.pi,  # [rad]
        max_steps: int = 3000,
        # Reward weights
        w_p: float = 1.0,
        w_v: float = 0.2,
        w_theta: float = 0.02,
        w_u: float = 0.01,
        # Success tolerances
        pos_tol: float = 1.0,       # [m]
        vel_tol: float = 0.05,      # [m/s]
        theta_tol: float = math.radians(2.0),  # [rad]
        spawn_radius: Tuple[float, float] = (10.0, 50.0),  # [m] min/max spawn distance
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
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
        self.max_steps = int(max_steps)
        self.w_p = float(w_p)
        self.w_v = float(w_v)
        self.w_theta = float(w_theta)
        self.w_u = float(w_u)
        self.pos_tol = float(pos_tol)
        self.vel_tol = float(vel_tol)
        self.theta_tol = float(theta_tol)
        self.spawn_radius = (float(spawn_radius[0]), float(spawn_radius[1]))
        self.render_mode = render_mode

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
        self._last_info: Dict[str, Any] = {}

    # ----------------------
    # Gym API
    # ----------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Random spawn at a ring [r_min, r_max], small random velocity, random attitude
        r_min, r_max = self.spawn_radius
        r = self.np_random.uniform(r_min, r_max)
        phi = self.np_random.uniform(-math.pi, math.pi)
        x0, y0 = r * math.cos(phi), r * math.sin(phi)
        vx0 = self.np_random.uniform(-0.1, 0.1)
        vy0 = self.np_random.uniform(-0.1, 0.1)
        theta0 = self.np_random.uniform(-math.pi, math.pi)
        omega0 = self.np_random.uniform(-0.01, 0.01)

        self.state = np.array([x0, y0, vx0, vy0, theta0, omega0], dtype=np.float64)
        self.steps = 0
        self._last_info = {"success": False, "done_reason": "reset"}

        obs = self.state.astype(np.float32)
        return obs, self._last_info.copy()

    def step(self, action: np.ndarray):
        self.steps += 1
        u = np.asarray(action, dtype=np.float64).copy()
        # Clip action to bounds
        u_thrust = float(np.clip(u[0], 0.0, 1.0))
        u_alpha = float(np.clip(u[1], -1.0, 1.0))

        x, y, vx, vy, theta, omega = self.state

        # Compute commanded accelerations
        a_t = (u_thrust * self.T_max) / self.m  # [m/s^2]
        ax = a_t * math.cos(theta)
        ay = a_t * math.sin(theta)

        # HCW planar dynamics
        x_ddot = 3.0 * self.n**2 * x + 2.0 * self.n * vy + ax
        y_ddot = -2.0 * self.n * vx + ay

        # Semi-implicit Euler integration (symplectic-like for better stability)
        vx = vx + x_ddot * self.dt
        vy = vy + y_ddot * self.dt
        x = x + vx * self.dt
        y = y + vy * self.dt

        # Attitude dynamics
        omega_dot = u_alpha * self.alpha_max
        omega = omega + omega_dot * self.dt
        theta = wrap_pi(theta + omega * self.dt)

        self.state = np.array([x, y, vx, vy, theta, omega], dtype=np.float64)

        # Observations
        obs = self.state.astype(np.float32)

        # Reward shaping
        p_norm = math.hypot(x, y)
        v_norm = math.hypot(vx, vy)
        reward = -(
            self.w_p * p_norm +
            self.w_v * v_norm +
            self.w_theta * abs(theta) +
            self.w_u * (u_thrust**2 + u_alpha**2)
        )

        # Termination conditions
        terminated = False
        success = False
        if (p_norm < self.pos_tol) and (v_norm < self.vel_tol) and (abs(theta) < self.theta_tol):
            terminated = True
            success = True
            done_reason = "success"
        elif (abs(x) > self.pos_limit) or (abs(y) > self.pos_limit) or (abs(vx) > self.v_limit) or (abs(vy) > self.v_limit) or (abs(theta) > self.theta_limit):
            terminated = True
            done_reason = "out_of_bounds"
        elif self.steps >= self.max_steps:
            terminated = True
            done_reason = "timeout"
        else:
            done_reason = "continue"

        truncated = False  # we use 'terminated' only here
        info = {
            "success": success,
            "done_reason": done_reason,
            "p_norm": p_norm,
            "v_norm": v_norm,
            "u_thrust": u_thrust,
            "u_alpha": u_alpha,
        }
        self._last_info = info

        if self.render_mode == "human":
            self._render_text()

        return obs, reward, terminated, truncated, info

    def _render_text(self):
        x, y, vx, vy, theta, omega = self.state
        print(
            f"t={self.steps*self.dt:6.1f}s | r=({x:7.2f},{y:7.2f}) m | v=({vx:6.3f},{vy:6.3f}) m/s | "
            f"theta={theta:+6.3f} rad | omega={omega:+6.3f} rad/s"
        )

    def render(self):
        # Gymnasium calls step's internal render when render_mode=="human".
        pass

    def close(self):
        pass


# ----------------------------
# Quick smoke test (no SB3)
# ----------------------------
if __name__ == "__main__":
    env = HCWSE2Env(render_mode="human")
    obs, info = env.reset(seed=0)
    print("Initial:", obs, info)
    # simple heuristic: thrust toward -position bearing, damp rotation toward 0
    for _ in range(200):
        x, y, vx, vy, theta, omega = obs
        # aim body toward target (0,0)
        desired_theta = math.atan2(-y, -x)
        theta_err = wrap_pi(desired_theta - theta)
        # PD on attitude to create rotational acceleration command
        u_alpha = np.clip(2.0 * theta_err - 0.5 * omega, -1.0, 1.0)
        # thrust proportional to distance, capped
        dist = math.hypot(x, y)
        u_thrust = float(np.clip(dist / 200.0, 0.0, 1.0))
        action = np.array([u_thrust, u_alpha], dtype=np.float32)
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            print("Terminated:", info)
            break
