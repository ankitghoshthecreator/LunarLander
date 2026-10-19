"""
Enhanced Photorealistic Sci-Fi Renderer for LunarLander Box2D Environment
==========================================================================
Renders genuine Gymnasium LunarLander Box2D physical state onto a high-resolution
sci-fi canvas matching the approved visual concept:
- Deep space starfield with glowing blue Earth in the upper right
- Detailed lunar surface with craters and horizon mountains
- Illuminated circular landing pad with green guidance beacon lights
- Detailed Apollo Lunar Module with golden descent stage, legs, and thruster plumes
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pygame
from pygame import gfxdraw

# Widescreen canvas dimensions (16:9 aspect ratio)
CANVAS_W = 1024
CANVAS_H = 576

# Box2D LunarLander coordinate space:
# W = 600 / 30 = 20.0 world units wide
# H = 400 / 30 = 13.33 world units high
WORLD_W = 20.0
WORLD_H = 13.33


class SciFiLunarRenderer:
    """High-resolution sci-fi renderer for Box2D LunarLander."""

    def __init__(self, width: int = CANVAS_W, height: int = CANVAS_H) -> None:
        self.width = width
        self.height = height

        # Ensure headless pygame
        pygame.init()
        self.surf = pygame.Surface((self.width, self.height))

        # Pre-generate static background (stars + Earth + lunar horizon)
        self.bg_surf = pygame.Surface((self.width, self.height))
        self._build_static_background()

    def _build_static_background(self) -> None:
        """Create static starry space background with glowing Earth."""
        self.bg_surf.fill((3, 6, 14))

        # 1. Starfield (deterministic seed)
        rng = np.random.default_rng(seed=1337)
        for _ in range(160):
            sx = int(rng.integers(0, self.width))
            sy = int(rng.integers(0, int(self.height * 0.72)))
            brightness = int(rng.integers(120, 255))
            color = (brightness, brightness, int(min(255, brightness + 20)))
            size = rng.choice([1, 1, 1, 2])
            if size == 1:
                self.bg_surf.set_at((sx, sy), color)
            else:
                pygame.draw.circle(self.bg_surf, color, (sx, sy), 1)

        # 2. Glowing Earth in upper right (around x=780, y=140, radius=90)
        earth_x = int(self.width * 0.78)
        earth_y = int(self.height * 0.28)
        earth_r = int(self.height * 0.18)

        # Atmosphere glow
        for r_offset in range(14, 0, -2):
            glow_alpha = int(35 * (1.0 - r_offset / 14.0))
            glow_surf = pygame.Surface((earth_r * 4, earth_r * 4), pygame.SRCALPHA)
            pygame.draw.circle(
                glow_surf,
                (0, 160, 255, glow_alpha),
                (earth_r * 2, earth_r * 2),
                earth_r + r_offset,
            )
            self.bg_surf.blit(glow_surf, (earth_x - earth_r * 2, earth_y - earth_r * 2))

        # Earth base sphere (ocean blue)
        pygame.draw.circle(self.bg_surf, (15, 65, 145), (earth_x, earth_y), earth_r)

        # Earth continents & cloud bands (procedural circles/arcs)
        cloud_surf = pygame.Surface((earth_r * 2, earth_r * 2), pygame.SRCALPHA)
        # Continents (greenish brown)
        for cx, cy, cr in [(0.4, 0.45, 0.35), (0.6, 0.3, 0.28), (0.35, 0.65, 0.25)]:
            px = int(cx * earth_r * 2)
            py = int(cy * earth_r * 2)
            pygame.draw.circle(cloud_surf, (45, 115, 75, 200), (px, py), int(cr * earth_r))
        # White swirl clouds
        for cx, cy, cr in [(0.5, 0.4, 0.4), (0.3, 0.25, 0.3), (0.65, 0.6, 0.35), (0.45, 0.75, 0.28)]:
            px = int(cx * earth_r * 2)
            py = int(cy * earth_r * 2)
            pygame.draw.circle(cloud_surf, (240, 245, 255, 180), (px, py), int(cr * earth_r))

        # Night-side shadow mask (crescent/half-earth illumination from left)
        shadow_surf = pygame.Surface((earth_r * 2, earth_r * 2), pygame.SRCALPHA)
        for y_idx in range(earth_r * 2):
            dy = (y_idx - earth_r) / float(earth_r)
            if abs(dy) < 1.0:
                dx = math.sqrt(1.0 - dy * dy) * earth_r
                # shadow on right
                shadow_start_x = int(earth_r + dx * 0.15)
                shadow_end_x = int(earth_r + dx + 2)
                if shadow_end_x > shadow_start_x:
                    pygame.draw.line(
                        shadow_surf,
                        (3, 6, 14, 230),
                        (shadow_start_x, y_idx),
                        (shadow_end_x, y_idx),
                    )

        # Mask circle to earth radius
        mask_surf = pygame.Surface((earth_r * 2, earth_r * 2), pygame.SRCALPHA)
        pygame.draw.circle(mask_surf, (255, 255, 255, 255), (earth_r, earth_r), earth_r)
        cloud_surf.blit(shadow_surf, (0, 0))
        cloud_surf.blit(mask_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        self.bg_surf.blit(cloud_surf, (earth_x - earth_r, earth_y - earth_r))

    def _world_to_canvas(self, wx: float, wy: float) -> tuple[int, int]:
        """Map Box2D world coordinate to canvas pixel coordinate."""
        # Box2D: x in [-10, 10] centered at 0 (or [0, 20]), y in [0, 13.33] from ground up.
        # Canvas: x in [0, width], y in [0, height] with y=0 at top.
        scale_x = self.width / WORLD_W
        scale_y = (self.height * 0.88) / WORLD_H

        cx = int(wx * scale_x)
        cy = int(self.height * 0.94 - wy * scale_y)
        return cx, cy

    def render(self, env: Any, last_action: int | None = None) -> np.ndarray:
        """Render the complete scene using genuine Box2D physics objects."""
        # 1. Blit static space background
        self.surf.blit(self.bg_surf, (0, 0))

        unwrapped = getattr(env, "unwrapped", env)
        if not hasattr(unwrapped, "moon") or unwrapped.moon is None:
            # Fallback to direct surfarray if environment not initialized
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2))

        # 2. Render Lunar Terrain from Box2D moon fixtures
        # In Gymnasium LunarLander, unwrapped.moon has edge fixtures defining ground
        terrain_points: list[tuple[int, int]] = []
        for fixture in unwrapped.moon.fixtures:
            edge = fixture.shape
            v1 = edge.vertices[0]
            v2 = edge.vertices[1]
            p1 = self._world_to_canvas(v1[0], v1[1])
            p2 = self._world_to_canvas(v2[0], v2[1])
            if not terrain_points:
                terrain_points.append(p1)
            terrain_points.append(p2)

        # Close polygon at bottom of screen
        if terrain_points:
            poly = list(terrain_points)
            poly.append((self.width + 10, self.height + 10))
            poly.append((-10, self.height + 10))
            # Base lunar surface
            pygame.draw.polygon(self.surf, (68, 72, 80), poly)
            # Surface lighting edge
            for idx in range(len(terrain_points) - 1):
                p_a = terrain_points[idx]
                p_b = terrain_points[idx + 1]
                pygame.draw.line(self.surf, (155, 160, 170), p_a, p_b, 3)

        # 3. Illuminated Circular Landing Pad
        h_x1 = getattr(unwrapped, "helipad_x1", 8.0)
        h_x2 = getattr(unwrapped, "helipad_x2", 12.0)
        h_y = getattr(unwrapped, "helipad_y", 3.33)
        p1 = self._world_to_canvas(h_x1, h_y)
        p2 = self._world_to_canvas(h_x2, h_y)
        pad_center_x = (p1[0] + p2[0]) // 2
        pad_center_y = (p1[1] + p2[1]) // 2 + 6
        pad_radius_x = int((p2[0] - p1[0]) * 0.72)
        pad_radius_y = int(pad_radius_x * 0.32)

        # Outer illuminated rim
        pad_rect = pygame.Rect(
            pad_center_x - pad_radius_x,
            pad_center_y - pad_radius_y,
            pad_radius_x * 2,
            pad_radius_y * 2,
        )
        pygame.draw.ellipse(self.surf, (30, 35, 45), pad_rect)
        pygame.draw.ellipse(self.surf, (0, 255, 136), pad_rect, 3)

        # Inner landing bullseye
        inner_rect = pygame.Rect(
            pad_center_x - int(pad_radius_x * 0.55),
            pad_center_y - int(pad_radius_y * 0.55),
            int(pad_radius_x * 1.1),
            int(pad_radius_y * 1.1),
        )
        pygame.draw.ellipse(self.surf, (0, 255, 136, 120), inner_rect, 2)

        # Guidance beacon lights
        for ang in [0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi, 5 * math.pi / 4, 3 * math.pi / 2, 7 * math.pi / 4]:
            bx = int(pad_center_x + pad_radius_x * math.cos(ang))
            by = int(pad_center_y + pad_radius_y * math.sin(ang))
            pygame.draw.circle(self.surf, (0, 255, 136), (bx, by), 3)

        # Landing pad marker flags (Gymnasium flags)
        for fx in [p1[0] - 12, p2[0] + 12]:
            pygame.draw.line(self.surf, (220, 225, 235), (fx, p1[1]), (fx, p1[1] - 32), 2)
            pygame.draw.polygon(
                self.surf,
                (255, 215, 0),
                [(fx, p1[1] - 32), (fx + 16, p1[1] - 24), (fx, p1[1] - 16)],
            )

        # 4. Box2D Active Thruster Particles (Exhaust Flame & Smoke)
        particles = getattr(unwrapped, "particles", [])
        for part in particles:
            if hasattr(part, "fixtures") and part.fixtures:
                pos = part.fixtures[0].body.position
                px, py = self._world_to_canvas(pos.x, pos.y)
                color = getattr(part, "color1", (255, 140, 20))
                pygame.draw.circle(self.surf, color, (px, py), 3)

        # 5. Apollo Lunar Module (Lander Body & Legs)
        if hasattr(unwrapped, "lander") and unwrapped.lander is not None:
            lander_body = unwrapped.lander
            lx, ly = self._world_to_canvas(lander_body.position.x, lander_body.position.y)
            angle = float(lander_body.angle)

            # Thruster Fire Plumes when Main Engine is Firing (action == 2)
            if last_action == 2 or (hasattr(unwrapped, "action") and unwrapped.action == 2):
                plume_surf = pygame.Surface((60, 80), pygame.SRCALPHA)
                plume_pts = [(30, 0), (12, 60), (30, 75), (48, 60)]
                pygame.draw.polygon(plume_surf, (255, 120, 0, 220), plume_pts)
                inner_pts = [(30, 0), (20, 40), (30, 55), (40, 40)]
                pygame.draw.polygon(plume_surf, (255, 240, 80, 255), inner_pts)
                rot_plume = pygame.transform.rotate(plume_surf, math.degrees(-angle))
                rect = rot_plume.get_rect(center=(lx, ly + 25))
                self.surf.blit(rot_plume, rect.topleft)

            # Draw Apollo Lander Sprite (Cabin + Gold stage + Struts)
            lander_w = 46
            lander_h = 42
            lsurf = pygame.Surface((lander_w * 2, lander_h * 2), pygame.SRCALPHA)
            cx = lander_w
            cy = lander_h

            # Gold Foil Descent Stage (Octagonal base)
            gold_color = (205, 155, 45)
            gold_rect = pygame.Rect(cx - 16, cy - 4, 32, 18)
            pygame.draw.rect(lsurf, gold_color, gold_rect, border_radius=3)
            pygame.draw.rect(lsurf, (235, 185, 75), gold_rect, width=2, border_radius=3)

            # White Ascent Stage (Crew Cabin)
            cabin_pts = [
                (cx - 12, cy - 4),
                (cx - 14, cy - 18),
                (cx - 7, cy - 26),
                (cx + 7, cy - 26),
                (cx + 14, cy - 18),
                (cx + 12, cy - 4),
            ]
            pygame.draw.polygon(lsurf, (240, 242, 248), cabin_pts)
            pygame.draw.polygon(lsurf, (180, 190, 205), cabin_pts, width=2)

            # Cockpit Dark Windows
            win_left = [(cx - 9, cy - 18), (cx - 3, cy - 18), (cx - 4, cy - 12), (cx - 9, cy - 14)]
            win_right = [(cx + 3, cy - 18), (cx + 9, cy - 18), (cx + 9, cy - 14), (cx + 4, cy - 12)]
            pygame.draw.polygon(lsurf, (15, 25, 45), win_left)
            pygame.draw.polygon(lsurf, (15, 25, 45), win_right)

            # Landing Struts / Legs
            leg_color = (215, 220, 230)
            # Left leg
            pygame.draw.line(lsurf, leg_color, (cx - 14, cy + 6), (cx - 26, cy + 24), 3)
            pygame.draw.line(lsurf, (235, 185, 75), (cx - 30, cy + 24), (cx - 22, cy + 24), 3)
            # Right leg
            pygame.draw.line(lsurf, leg_color, (cx + 14, cy + 6), (cx + 26, cy + 24), 3)
            pygame.draw.line(lsurf, (235, 185, 75), (cx + 22, cy + 24), (cx + 30, cy + 24), 3)

            # Rotate lander with physics angle
            rot_lander = pygame.transform.rotate(lsurf, math.degrees(-angle))
            lrect = rot_lander.get_rect(center=(lx, ly))
            self.surf.blit(rot_lander, lrect.topleft)

        # Convert to numpy uint8 RGB array (H, W, 3)
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2))
