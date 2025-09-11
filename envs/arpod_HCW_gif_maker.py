"""
Arpod HCW GIF Maker
-------------------
Run a deterministic rollout of a trained SAC model in the HCW environment,
save histories, and generate a 3000x1200 GIF:

- Centered circle at origin with radius = env.pos_tol (p_tol)
- Chaser rendered as an oriented cube; thrust shown on the aft side
- Auto-fit world extents to the full trajectory while preserving equal scale

CLI:
    python -m arpod_HCW_gif_maker --model /path/to/model.zip \
        --out trajectories/run1 --seed 0

Outputs in --out:
- history.npy               (state history [N,6])
- history_actuator.npy      (actuator history [N,2] in SI units)
- reward_history.npy        (reward history [N])
- rollout.gif               (3000x1200 animation)
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Tuple, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from stable_baselines3 import SAC
except Exception as e:  # pragma: no cover
    SAC = None  # Will error later with a clear message

from arpod_HCW import HCWSE2Env


# -------------------------
# Rendering helpers
# -------------------------

def world_to_px(x: float, y: float, cx: float, cy: float, m_per_px: float, width: int, height: int) -> Tuple[int, int]:
    """Convert world meters to pixel coordinates with origin-centered view.
    (cx, cy) are world center in meters; m_per_px is meters per pixel.
    Image origin is top-left; +y in world goes up, so invert for pixels.
    """
    px = int(round((x - cx) / m_per_px + width / 2.0))
    py = int(round((cy - y) / m_per_px + height / 2.0))
    return px, py


def compute_view(history_xy: np.ndarray, p_tol: float, width: int, height: int, margin: float = 0.1) -> Tuple[float, float, float]:
    """Compute view scaling centered at origin, auto-fitting the trajectory and p_tol.
    Returns (m_per_px, cx, cy) with cx=cy=0 to keep origin centered.

    - Preserve aspect ratio: width:height (e.g., 3000:1200)
    - Ensure equal scale on x and y (meters per pixel identical)
    - Include a margin fraction around the furthest extent
    """
    cx = 0.0
    cy = 0.0

    # Max absolute extents from trajectory and p_tol circle
    x_abs = float(np.max(np.abs(history_xy[:, 0]))) if history_xy.size else 0.0
    y_abs = float(np.max(np.abs(history_xy[:, 1]))) if history_xy.size else 0.0
    x_half = max(x_abs, p_tol)
    y_half = max(y_abs, p_tol)

    # Add margin
    x_half *= (1.0 + margin)
    y_half *= (1.0 + margin)

    # Convert to meters per pixel to ensure full coverage within image
    m_per_px_x = x_half / (width / 2.0) if width > 0 else 1.0
    m_per_px_y = y_half / (height / 2.0) if height > 0 else 1.0
    m_per_px = max(m_per_px_x, m_per_px_y, 1e-9)  # same scaling for both axes
    return m_per_px, cx, cy


def rot2d(x: float, y: float, theta: float) -> Tuple[float, float]:
    c, s = math.cos(theta), math.sin(theta)
    return c * x - s * y, s * x + c * y


def draw_cube_with_thrust(draw: ImageDraw.ImageDraw,
                          pos: Tuple[float, float],
                          theta: float,
                          thrust_n: float,
                          t_max: float,
                          cx: float, cy: float, m_per_px: float,
                          width: int, height: int) -> None:
    """Draw a square 'cube' at pos with orientation theta.
    Thrust is a flame on the aft side (opposite body-forward direction) if thrust_n>0.
    """
    # Visual size ~ 2 m cube for readability
    side_m = 2.0
    half = side_m / 2.0

    # Define local square corners (CCW) centered at origin
    local_pts = [(-half, -half), (half, -half), (half, half), (-half, half)]
    # Rotate and translate to world
    world_pts: List[Tuple[float, float]] = []
    for (lx, ly) in local_pts:
        rx, ry = rot2d(lx, ly, theta)
        world_pts.append((pos[0] + rx, pos[1] + ry))

    # Convert to pixels
    px_pts = [world_to_px(x, y, cx, cy, m_per_px, width, height) for (x, y) in world_pts]
    draw.polygon(px_pts, outline=(30, 90, 200), fill=(180, 200, 255))

    # Draw body-forward direction marker (nose) as a short line from center
    nose_len_m = 1.5
    nose_end = (pos[0] + nose_len_m * math.cos(theta), pos[1] + nose_len_m * math.sin(theta))
    p0 = world_to_px(pos[0], pos[1], cx, cy, m_per_px, width, height)
    p1 = world_to_px(nose_end[0], nose_end[1], cx, cy, m_per_px, width, height)
    draw.line([p0, p1], fill=(30, 90, 200), width=max(1, int(round(2 / m_per_px))))

    # Draw thrust flame on aft side when thrust > 0
    if thrust_n > 1e-6 and t_max > 0:
        # Aft direction is opposite forward (-theta)
        aft_dir = theta + math.pi
        # Place flame a bit behind the cube center
        aft_offset = half + 0.2
        base = (pos[0] + aft_offset * math.cos(aft_dir), pos[1] + aft_offset * math.sin(aft_dir))

        # Scale flame length with normalized thrust
        u = float(np.clip(thrust_n / t_max, 0.0, 1.0))
        flame_len = 2.0 + 6.0 * u  # meters
        width_m = 0.6 + 0.6 * u

        # Triangle points: base-left, tip, base-right
        # Build base normal to aft_dir
        nx, ny = -math.sin(aft_dir), math.cos(aft_dir)
        p_base_l = (base[0] + nx * width_m, base[1] + ny * width_m)
        p_base_r = (base[0] - nx * width_m, base[1] - ny * width_m)
        tip = (base[0] + math.cos(aft_dir) * flame_len, base[1] + math.sin(aft_dir) * flame_len)

        tri = [world_to_px(*p_base_l, cx, cy, m_per_px, width, height),
               world_to_px(*tip, cx, cy, m_per_px, width, height),
               world_to_px(*p_base_r, cx, cy, m_per_px, width, height)]
        draw.polygon(tri, fill=(255, 140, 0), outline=(220, 100, 0))


def _nice_step(span: float, target_ticks: int = 8) -> float:
    """Return a 'nice' tick step for a given span and desired tick count."""
    if span <= 0:
        return 1.0
    raw = span / max(1, target_ticks)
    exp = math.floor(math.log10(raw))
    base = 10 ** exp
    for m in (1, 2, 5, 10):
        step = m * base
        if step >= raw - 1e-12:
            return step
    return 10 * base


def draw_axes(draw: ImageDraw.ImageDraw,
              width: int, height: int,
              m_per_px: float,
              cx: float, cy: float) -> None:
    """Draw x/y axes crossing at origin with tick marks and numeric labels (meters)."""
    # Axis lines at y=0 and x=0
    origin_px = world_to_px(0.0, 0.0, cx, cy, m_per_px, width, height)
    axis_color = (180, 180, 180)
    tick_color = (140, 140, 140)
    text_color = (90, 90, 90)
    line_w = max(1, int(round(2 / m_per_px)))
    tick_len_px = max(4, int(round(8 / (m_per_px * 4))))
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Horizontal axis (x-axis)
    draw.line([(0, origin_px[1]), (width, origin_px[1])], fill=axis_color, width=line_w)
    # Vertical axis (y-axis)
    draw.line([(origin_px[0], 0), (origin_px[0], height)], fill=axis_color, width=line_w)

    # Determine visible spans in meters
    x_half_m = (width / 2.0) * m_per_px
    y_half_m = (height / 2.0) * m_per_px
    x_step = _nice_step(2 * x_half_m, target_ticks=10)
    y_step = _nice_step(2 * y_half_m, target_ticks=6)

    # X ticks and labels
    x_val = -math.floor(x_half_m / x_step) * x_step
    while x_val <= x_half_m + 1e-9:
        px, py = world_to_px(x_val, 0.0, cx, cy, m_per_px, width, height)
        draw.line([(px, origin_px[1] - tick_len_px), (px, origin_px[1] + tick_len_px)], fill=tick_color, width=1)
        if abs(x_val) > 1e-12:
            label = f"{int(x_val)}" if abs(x_step) >= 1 else f"{x_val:.1f}"
            if font is not None:
                # Use textbbox when available (newer Pillow); fall back to textsize / font.getsize.
                try:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                except AttributeError:
                    try:
                        tw, th = draw.textsize(label, font=font)
                    except Exception:
                        # Last resort: estimate using font.getsize if available
                        if hasattr(font, "getsize"):
                            tw, th = font.getsize(label)
                        else:
                            tw = len(label) * 6
                            th = 10
                draw.text((px - tw / 2, origin_px[1] + tick_len_px + 2), label, fill=text_color, font=font)
        x_val += x_step

    # Y ticks and labels
    y_val = -math.floor(y_half_m / y_step) * y_step
    while y_val <= y_half_m + 1e-9:
        px, py = world_to_px(0.0, y_val, cx, cy, m_per_px, width, height)
        draw.line([(origin_px[0] - tick_len_px, py), (origin_px[0] + tick_len_px, py)], fill=tick_color, width=1)
        if abs(y_val) > 1e-12:
            label = f"{int(y_val)}" if abs(y_step) >= 1 else f"{y_val:.1f}"
            if font is not None:
                try:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                except AttributeError:
                    try:
                        tw, th = draw.textsize(label, font=font)
                    except Exception:
                        if hasattr(font, "getsize"):
                            tw, th = font.getsize(label)
                        else:
                            tw = len(label) * 6
                            th = 10
                draw.text((origin_px[0] + tick_len_px + 4, py - th / 2), label, fill=text_color, font=font)
        y_val += y_step


def draw_frame(img: Image.Image,
               history: np.ndarray,
               actions_si: np.ndarray,
               step: int,
               p_tol: float,
               t_max: float,
               m_per_px: float,
               cx: float, cy: float) -> None:
    """Draw a single frame up to `step` inclusive onto `img`.
    - history: [N,6] states
    - actions_si: [N,2] actuator history (N, alpha_dot)
    """
    width, height = img.size
    draw = ImageDraw.Draw(img)

    # Background
    draw.rectangle([0, 0, width, height], fill=(250, 250, 250))

    # Axes with ticks/labels
    draw_axes(draw, width, height, m_per_px, cx, cy)

    # Draw target at origin and p_tol circle
    # Draw circle centered at origin with radius p_tol
    r_px = int(round(p_tol / m_per_px))
    center_px = world_to_px(0.0, 0.0, cx, cy, m_per_px, width, height)
    bbox = [center_px[0] - r_px, center_px[1] - r_px, center_px[0] + r_px, center_px[1] + r_px]
    draw.ellipse(bbox, outline=(180, 180, 180), width=max(1, int(round(2 / m_per_px))))
    # Target marker
    cross = 6
    draw.line([(center_px[0] - cross, center_px[1]), (center_px[0] + cross, center_px[1])], fill=(120, 120, 120), width=2)
    draw.line([(center_px[0], center_px[1] - cross), (center_px[0], center_px[1] + cross)], fill=(120, 120, 120), width=2)

    # Trajectory trace up to current step
    pts = [world_to_px(x, y, cx, cy, m_per_px, width, height) for (x, y) in history[: step + 1, :2]]
    if len(pts) >= 2:
        draw.line(pts, fill=(60, 60, 60), width=max(1, int(round(2 / m_per_px))))

    # Current chaser state
    x, y, vx, vy, theta, omega = history[step]
    thrust_n = float(actions_si[step, 0]) if actions_si.shape[0] > step else 0.0
    draw_cube_with_thrust(draw, (x, y), theta, thrust_n, t_max, cx, cy, m_per_px, width, height)


# -------------------------
# Rollout and GIF
# -------------------------

def rollout_and_render(model_path: str,
                       out_dir: str,
                       seed: int = 0,
                       width: int = 3000,
                       height: int = 1200,
                       ms_per_step: int = 100,
                       spawn_radius: Tuple[float, float] = (40.0, 50.0)) -> str:
    if SAC is None:
        raise RuntimeError("stable-baselines3 is required to load the SAC model.")

    os.makedirs(out_dir, exist_ok=True)

    # Create env with desired spawn radius
    env = HCWSE2Env(render_mode=None, spawn_radius=spawn_radius)
    obs, _info = env.reset(seed=seed)

    # Load model
    model = SAC.load(model_path)

    # Deterministic rollout until termination or truncation
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

    # Histories (lists) to arrays
    history = np.array(env.history, dtype=np.float64)
    actions_si = np.array(env.action_history, dtype=np.float64)
    rewards = np.array(env.reward_history, dtype=np.float64)

    # Save histories
    np.save(os.path.join(out_dir, "history.npy"), history)
    np.save(os.path.join(out_dir, "history_actuator.npy"), actions_si)
    np.save(os.path.join(out_dir, "reward_history.npy"), rewards)

    # Compute view scaling
    p_tol = env.pos_tol
    m_per_px, cx, cy = compute_view(history[:, :2], p_tol, width, height)

    # Render frames
    frames: List[Image.Image] = []
    for i in range(history.shape[0]):
        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw_frame(img, history, actions_si, i, p_tol, env.T_max, m_per_px, cx, cy)
        frames.append(img)

    # Save GIF
    gif_path = os.path.join(out_dir, "rollout.gif")
    if len(frames) == 1:
        frames[0].save(gif_path)
    else:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=ms_per_step,
            loop=0,
            optimize=False,
            disposal=2,
        )

    return gif_path


def main():
    parser = argparse.ArgumentParser(description="Generate a 3000x1200 GIF from a SAC model in the HCW env.")
    parser.add_argument("--model", required=True, help="Path to trained SAC model .zip")
    parser.add_argument("--out", default="trajectories/gif_run", help="Output directory for GIF and histories")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for env reset")
    parser.add_argument("--ms-per-step", type=int, default=100, help="GIF frame duration in milliseconds per env step")
    parser.add_argument("--width", type=int, default=1500, help="Output GIF width in pixels")
    parser.add_argument("--height", type=int, default=600, help="Output GIF height in pixels")
    parser.add_argument("--spawn-min", type=float, default=60.0, help="Spawn radius min [m]")
    parser.add_argument("--spawn-max", type=float, default=80.0, help="Spawn radius max [m]")

    args = parser.parse_args()
    spawn_radius = (float(args.spawn_min), float(args.spawn_max))

    gif_path = rollout_and_render(
        model_path=args.model,
        out_dir=args.out,
        seed=args.seed,
        width=args.width,
        height=args.height,
        ms_per_step=args.ms_per_step,
        spawn_radius=spawn_radius,
    )
    print(f"Saved GIF to: {gif_path}")
    print(f"Saved histories to: {args.out}")


if __name__ == "__main__":
    main()
