"""Live Pygame viewer to visualize LunarLander agent landing with interactive gravity control."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
import pygame
from stable_baselines3 import PPO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a trained agent land with interactive gravity.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/ppo_shaped_s0/best_model.zip",
        help="Path to the trained SB3 model (.zip) to visualize.",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=9.8,
        help="Initial gravitational acceleration magnitude (e.g. 9.8 for Earth, 1.62 for Moon, 10.0 for default).",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Follow training checkpoints and reload automatically when modified.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Number of episodes to watch (0 = infinite loop).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional environment seed.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions stochastically instead of deterministically.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Target frames per second (default: 50).",
    )
    return parser.parse_args()


def get_planet_label(g: float) -> str:
    if abs(g - 9.8) < 0.2:
        return "Earth"
    if abs(g - 1.62) < 0.2:
        return "Moon"
    if abs(g - 3.71) < 0.2:
        return "Mars"
    if abs(g - 10.0) < 0.1:
        return "Default"
    return "Custom"


def apply_gravity(env: gym.Env, g_val: float) -> float:
    """Apply gravitation magnitude g_val downwards in Box2D world and env config."""
    # Clamp to a safe physical range
    g_clamped = max(0.5, min(30.0, abs(g_val)))
    unwrapped = env.unwrapped
    unwrapped.gravity = -g_clamped
    if hasattr(unwrapped, "world") and unwrapped.world is not None:
        unwrapped.world.gravity = (0.0, -g_clamped)
    return g_clamped


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model checkpoint not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    model = PPO.load(str(model_path), device="cpu")
    last_mtime = model_path.stat().st_mtime
    last_reload_check = time.time()

    # Initialize Pygame and font
    pygame.init()
    pygame.font.init()
    font_large = pygame.font.SysFont("Consolas", 15, bold=True)
    font_small = pygame.font.SysFont("Consolas", 12)

    viewport_w, viewport_h = 600, 400
    screen = pygame.display.set_mode((viewport_w, viewport_h))
    pygame.display.set_caption("Rocket Lander - Interactive Gravity Visualizer")
    clock = pygame.time.Clock()

    # Create Gym environment with rgb_array so we have total control over the Pygame canvas
    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    # Gravitation state
    active_gravity = apply_gravity(env, args.gravity)
    input_text = f"{active_gravity:.1f}"
    input_active = False

    # UI Element Rectangles (Top-Left)
    hud_rect = pygame.Rect(10, 10, 245, 95)
    input_box_rect = pygame.Rect(20, 32, 60, 24)
    btn_earth = pygame.Rect(20, 62, 50, 20)
    btn_moon = pygame.Rect(75, 62, 48, 20)
    btn_mars = pygame.Rect(128, 62, 48, 20)
    btn_default = pygame.Rect(181, 62, 64, 20)

    episodes_run = 0
    total_reward = 0.0
    steps = 0
    paused = False

    reset_kwargs = {}
    if args.seed is not None:
        reset_kwargs["seed"] = args.seed

    obs, _ = env.reset(**reset_kwargs)
    apply_gravity(env, active_gravity)

    print(
        f"Visualization running with gravity {active_gravity:.1f} m/s² ({get_planet_label(active_gravity)}).\n"
        "Controls:\n"
        "  - Click input box in top-left to enter gravity (e.g. 9.8 for Earth, 1.62 for Moon), then press Enter.\n"
        "  - Click preset buttons [Earth] [Moon] [Mars] [Default].\n"
        "  - Up/Down Arrow keys: adjust gravity by ±0.5 m/s².\n"
        "  - 'R' key: reset/restart episode with current gravity.\n"
        "  - 'Space' key: pause/unpause simulation.\n"
        "  - Press Ctrl+C or close window to exit."
    )

    running = True
    try:
        while running:
            # Handle Pygame Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if input_box_rect.collidepoint(event.pos):
                        input_active = True
                    else:
                        input_active = False
                        # On unfocus, try to apply typed text
                        try:
                            val = float(input_text.strip())
                            active_gravity = apply_gravity(env, val)
                            input_text = f"{active_gravity:.1f}"
                        except ValueError:
                            input_text = f"{active_gravity:.1f}"

                    # Check preset button clicks
                    if btn_earth.collidepoint(event.pos):
                        active_gravity = apply_gravity(env, 9.8)
                        input_text = "9.8"
                    elif btn_moon.collidepoint(event.pos):
                        active_gravity = apply_gravity(env, 1.62)
                        input_text = "1.62"
                    elif btn_mars.collidepoint(event.pos):
                        active_gravity = apply_gravity(env, 3.71)
                        input_text = "3.71"
                    elif btn_default.collidepoint(event.pos):
                        active_gravity = apply_gravity(env, 10.0)
                        input_text = "10.0"

                elif event.type == pygame.KEYDOWN:
                    if input_active:
                        if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                            input_active = False
                            try:
                                val = float(input_text.strip())
                                active_gravity = apply_gravity(env, val)
                                input_text = f"{active_gravity:.1f}"
                                print(f"[Gravity] Updated gravitation to {active_gravity:.2f} m/s² ({get_planet_label(active_gravity)})")
                            except ValueError:
                                input_text = f"{active_gravity:.1f}"
                        elif event.key == pygame.K_BACKSPACE:
                            input_text = input_text[:-1]
                        elif event.key == pygame.K_ESCAPE:
                            input_active = False
                            input_text = f"{active_gravity:.1f}"
                        else:
                            # Only accept numeric, decimal, minus
                            if event.unicode in "0123456789.-":
                                input_text += event.unicode
                    else:
                        if event.key == pygame.K_UP:
                            active_gravity = apply_gravity(env, active_gravity + 0.5)
                            input_text = f"{active_gravity:.1f}"
                            print(f"[Gravity] Adjusted gravitation to {active_gravity:.2f} m/s²")
                        elif event.key == pygame.K_DOWN:
                            active_gravity = apply_gravity(env, max(0.5, active_gravity - 0.5))
                            input_text = f"{active_gravity:.1f}"
                            print(f"[Gravity] Adjusted gravitation to {active_gravity:.2f} m/s²")
                        elif event.key == pygame.K_r:
                            obs, _ = env.reset()
                            apply_gravity(env, active_gravity)
                            total_reward = 0.0
                            steps = 0
                            print("[Simulation] Restarted episode (R pressed).")
                        elif event.key == pygame.K_SPACE:
                            paused = not paused
                            print(f"[Simulation] {'Paused' if paused else 'Resumed'}.")

            if not running:
                break

            # Step simulation if not paused
            if not paused:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, terminated, truncated, info = env.step(int(action))
                total_reward += float(reward)
                steps += 1

                if terminated or truncated:
                    episodes_run += 1
                    outcome = "TERMINATED" if terminated else "TIMEOUT"
                    if reward >= 100:
                        status = "LANDED (SUCCESS)"
                    elif reward <= -100:
                        status = "CRASHED"
                    else:
                        status = outcome

                    print(
                        f"Episode {episodes_run}: {status} | Gravity: {active_gravity:.1f} m/s² ({get_planet_label(active_gravity)}) | "
                        f"Total Reward: {total_reward:+.1f} | Steps: {steps}"
                    )

                    if args.episodes > 0 and episodes_run >= args.episodes:
                        break

                    # Check for model file reload if --follow
                    if args.follow and time.time() - last_reload_check > 5.0:
                        last_reload_check = time.time()
                        try:
                            current_mtime = model_path.stat().st_mtime
                            if current_mtime != last_mtime:
                                print(f"[Follow] Reloading updated checkpoint: {model_path}")
                                model = PPO.load(str(model_path), device="cpu")
                                last_mtime = current_mtime
                        except Exception as err:
                            print(f"[Follow] Warning: Could not reload model: {err}")

                    total_reward = 0.0
                    steps = 0
                    obs, _ = env.reset()
                    apply_gravity(env, active_gravity)

            # Render Environment Frame
            frame = env.render()
            if frame is not None:
                sim_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                screen.blit(sim_surface, (0, 0))

            # --- Draw Top-Left HUD (Interactive Gravitation Control) ---
            # Semi-transparent overlay surface for clean visibility
            overlay = pygame.Surface((hud_rect.width, hud_rect.height), pygame.SRCALPHA)
            overlay.fill((18, 22, 32, 215))  # Dark sleek translucent background
            screen.blit(overlay, (hud_rect.x, hud_rect.y))
            pygame.draw.rect(screen, (70, 85, 110), hud_rect, width=1, border_radius=6)

            # Header / Planet label
            planet_tag = f"Active: {active_gravity:.1f} m/s² ({get_planet_label(active_gravity)})"
            tag_surf = font_small.render(planet_tag, True, (130, 210, 255))
            screen.blit(tag_surf, (hud_rect.x + 10, hud_rect.y + 8))

            # Input box for gravitation
            box_bg = (35, 42, 58) if not input_active else (45, 55, 80)
            box_border = (0, 230, 255) if input_active else (100, 120, 150)
            pygame.draw.rect(screen, box_bg, input_box_rect, border_radius=4)
            pygame.draw.rect(screen, box_border, input_box_rect, width=2 if input_active else 1, border_radius=4)

            # Display typed text inside input box
            cursor = "|" if input_active and (int(time.time() * 2) % 2 == 0) else ""
            txt_surf = font_large.render(input_text + cursor, True, (255, 255, 255))
            screen.blit(txt_surf, (input_box_rect.x + 5, input_box_rect.y + 4))

            # "Gravitation" label next to the input box
            lbl_surf = font_large.render("gravitation", True, (240, 240, 240))
            screen.blit(lbl_surf, (input_box_rect.right + 8, input_box_rect.y + 4))

            # Preset buttons: [Earth 9.8] [Moon] [Mars] [Def]
            for btn, label, active in [
                (btn_earth, "Earth", abs(active_gravity - 9.8) < 0.2),
                (btn_moon, "Moon", abs(active_gravity - 1.62) < 0.2),
                (btn_mars, "Mars", abs(active_gravity - 3.71) < 0.2),
                (btn_default, "Default", abs(active_gravity - 10.0) < 0.1),
            ]:
                b_color = (40, 110, 160) if active else (40, 48, 65)
                pygame.draw.rect(screen, b_color, btn, border_radius=3)
                pygame.draw.rect(screen, (80, 100, 130), btn, width=1, border_radius=3)
                b_text = font_small.render(label, True, (255, 255, 255) if active else (180, 190, 205))
                # Center text in button
                t_rect = b_text.get_rect(center=btn.center)
                screen.blit(b_text, t_rect)

            # Small telemetry in top-right
            if obs is not None:
                telemetry_str = f"Alt: {obs[1]:.2f} | Vy: {obs[3]:.2f} | Angle: {obs[4]:.2f}"
                t_surf = font_small.render(telemetry_str, True, (200, 200, 200))
                screen.blit(t_surf, (viewport_w - t_surf.get_width() - 10, 10))

            if paused:
                pause_surf = font_large.render("PAUSED (Space to Resume)", True, (255, 200, 50))
                screen.blit(pause_surf, (viewport_w // 2 - pause_surf.get_width() // 2, 15))

            pygame.display.flip()
            clock.tick(args.fps)

    except KeyboardInterrupt:
        print("\nExiting visualization.")
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
