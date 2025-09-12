"""
Advanced Axarion Engine Renderer
High-performance 2D rendering with advanced effects and optimizations
"""

import pygame
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from .game_object import GameObject
from .camera import Camera

class Renderer:
    """Advanced 2D renderer with effects, lighting, and optimization"""

    def __init__(self, width: int, height: int, surface=None):
        self.width = width
        self.height = height

        # Initialize display
        if surface is None:
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Axarion Engine")
        else:
            self.screen = surface

        # Rendering surfaces for different layers
        self.background_surface = pygame.Surface((width, height))
        self.game_surface = pygame.Surface((width, height))
        self.effect_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.ui_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.lighting_surface = pygame.Surface((width, height), pygame.SRCALPHA)

        # Camera system
        self.camera = Camera(0, 0, width, height)

        # Rendering settings
        self.background_color = (30, 30, 40)
        self.debug_mode = False
        self.vsync_enabled = True

        # Advanced rendering features
        self.lighting_enabled = True
        self.post_processing_enabled = True
        self.particle_effects_enabled = True
        self.shader_effects_enabled = True

        # Performance optimization
        self.frustum_culling_enabled = True
        self.sprite_batching_enabled = True
        self.render_layers = {}
        self.render_queue = []

        # Lighting system
        self.lights = []
        self.ambient_light = 0.3
        self.shadows_enabled = True

        # Post-processing effects
        self.post_effects = []
        self.bloom_enabled = False
        self.chromatic_aberration = False
        self.screen_shake = {"intensity": 0, "duration": 0, "time": 0}

        # Fonts
        pygame.font.init()
        self.default_font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 16)
        self.large_font = pygame.font.Font(None, 48)

        # Performance tracking
        self.render_stats = {
            "objects_rendered": 0,
            "objects_culled": 0,
            "draw_calls": 0,
            "triangles": 0
        }

        # Render targets for effects
        self.render_targets = {}
        self.create_render_target("main", width, height)
        self.create_render_target("bloom", width // 2, height // 2)
        self.create_render_target("lighting", width, height)

    def create_render_target(self, name: str, width: int, height: int):
        """Create a render target for effects"""
        self.render_targets[name] = pygame.Surface((width, height), pygame.SRCALPHA)

    def set_camera(self, camera: Camera):
        """Set the active camera"""
        self.camera = camera

    def get_camera(self) -> Tuple[float, float]:
        """Get camera position"""
        return self.camera.position

    def set_background_color(self, color: Tuple[int, int, int]):
        """Set background color"""
        self.background_color = color

    def clear(self):
        """Clear all render surfaces"""
        self.screen.fill(self.background_color)
        self.background_surface.fill(self.background_color)
        self.game_surface.fill((0, 0, 0, 0))
        self.effect_surface.fill((0, 0, 0, 0))
        self.ui_surface.fill((0, 0, 0, 0))
        self.lighting_surface.fill((0, 0, 0, 0))

        # Reset stats
        self.render_stats = {
            "objects_rendered": 0,
            "objects_culled": 0,
            "draw_calls": 0,
            "triangles": 0
        }

    def is_on_screen(self, game_object: GameObject) -> bool:
        """Check if object is visible on screen (frustum culling)"""
        if not self.frustum_culling_enabled:
            return True

        bounds = game_object.get_bounds()
        cam_x, cam_y = self.camera.position

        # Check if object bounds intersect with camera view
        return (bounds[2] >= cam_x and bounds[0] <= cam_x + self.width and
                bounds[3] >= cam_y and bounds[1] <= cam_y + self.height)

    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        cam_x, cam_y = self.camera.position
        screen_x = int(world_x - cam_x)
        screen_y = int(world_y - cam_y)
        return (screen_x, screen_y)

    def screen_to_world(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates"""
        cam_x, cam_y = self.camera.position
        world_x = screen_x + cam_x
        world_y = screen_y + cam_y
        return (world_x, world_y)

    # LIGHTING SYSTEM
    def add_light(self, x: float, y: float, radius: float, color: Tuple[int, int, int], 
                  intensity: float = 1.0, light_type: str = "point"):
        """Add dynamic light source"""
        light = {
            "position": (x, y),
            "radius": radius,
            "color": color,
            "intensity": intensity,
            "type": light_type,
            "enabled": True
        }
        self.lights.append(light)
        return light

    def remove_light(self, light: Dict):
        """Remove light source"""
        if light in self.lights:
            self.lights.remove(light)

    def render_lighting(self):
        """Render dynamic lighting"""
        if not self.lighting_enabled:
            return

        # Clear lighting surface
        self.lighting_surface.fill((int(255 * self.ambient_light),) * 3)

        # Render each light
        for light in self.lights:
            if not light["enabled"]:
                continue

            x, y = light["position"]
            screen_x, screen_y = self.world_to_screen(x, y)
            radius = light["radius"]
            color = light["color"]
            intensity = light["intensity"]

            # Create light gradient
            if light["type"] == "point":
                self.draw_radial_light(screen_x, screen_y, radius, color, intensity)
            elif light["type"] == "spot":
                # TODO: Implement spotlight
                pass

    def draw_radial_light(self, x: int, y: int, radius: float, 
                         color: Tuple[int, int, int], intensity: float):
        """Draw radial light with falloff"""
        # Create light surface
        light_size = int(radius * 2)
        light_surface = pygame.Surface((light_size, light_size), pygame.SRCALPHA)

        # Draw light gradient
        for r in range(int(radius)):
            alpha = int(255 * intensity * (1 - r / radius))
            if alpha > 0:
                light_color = (*color, alpha)
                pygame.draw.circle(light_surface, light_color, 
                                 (light_size // 2, light_size // 2), 
                                 int(radius - r))

        # Blit to lighting surface
        self.lighting_surface.blit(light_surface, 
                                 (x - light_size // 2, y - light_size // 2),
                                 special_flags=pygame.BLEND_ADD)

    # ADVANCED DRAWING METHODS
    def draw_game_object(self, game_object: GameObject):
        """Draw game object with advanced rendering"""
        if not game_object.visible:
            return

        # Frustum culling
        if not self.is_on_screen(game_object):
            self.render_stats["objects_culled"] += 1
            return

        x, y = game_object.position
        screen_x, screen_y = self.world_to_screen(x, y)

        # Get layer for depth sorting
        layer = getattr(game_object, 'layer', 0)
        if layer not in self.render_layers:
            self.render_layers[layer] = []

        # Add to render queue instead of immediate rendering for batching
        self.render_queue.append({
            "object": game_object,
            "screen_pos": (screen_x, screen_y),
            "layer": layer
        })

        self.render_stats["objects_rendered"] += 1

    def flush_render_queue(self):
        """Render all queued objects with optimal batching"""
        # Sort by layer for proper depth
        self.render_queue.sort(key=lambda x: x["layer"])

        # Group by object type for batching
        batches = {}
        for item in self.render_queue:
            obj_type = item["object"].object_type
            if obj_type not in batches:
                batches[obj_type] = []
            batches[obj_type].append(item)

        # Render each batch
        for obj_type, batch in batches.items():
            self.render_batch(obj_type, batch)

        # Clear queue
        self.render_queue.clear()

    def render_batch(self, obj_type: str, batch: List[Dict]):
        """Render a batch of similar objects efficiently"""
        if obj_type == "sprite":
            self.render_sprite_batch(batch)
        elif obj_type in ["rectangle", "square"]:
            self.render_rect_batch(batch)
        elif obj_type == "circle":
            self.render_circle_batch(batch)
        else:
            # Fallback to individual rendering
            for item in batch:
                self.render_individual_object(item["object"], item["screen_pos"])

    def render_sprite_batch(self, batch: List[Dict]):
        """Optimized sprite batch rendering"""
        for item in batch:
            obj = item["object"]
            screen_x, screen_y = item["screen_pos"]

            # Get sprite surface
            sprite = obj.get_current_sprite()
            if sprite:
                # Apply transformations
                if obj.rotation != 0 or obj.scale != (1.0, 1.0):
                    sprite = self.apply_transformations(sprite, obj.rotation, obj.scale)

                # Apply opacity
                opacity = getattr(obj, 'opacity', 1.0)
                if opacity < 1.0:
                    sprite = sprite.copy()
                    sprite.set_alpha(int(255 * opacity))

                self.game_surface.blit(sprite, (screen_x, screen_y))
                self.render_stats["draw_calls"] += 1

    def render_rect_batch(self, batch: List[Dict]):
        """Optimized rectangle batch rendering"""
        for item in batch:
            obj = item["object"]
            screen_x, screen_y = item["screen_pos"]

            width = obj.get_property("width", 50)
            height = obj.get_property("height", 50)
            color = obj.get_property("color", (255, 255, 255))

            rect = pygame.Rect(screen_x, screen_y, width, height)
            pygame.draw.rect(self.game_surface, color, rect)
            self.render_stats["draw_calls"] += 1

    def render_circle_batch(self, batch: List[Dict]):
        """Optimized circle batch rendering"""
        for item in batch:
            obj = item["object"]
            screen_x, screen_y = item["screen_pos"]

            radius = obj.get_property("radius", 25)
            color = obj.get_property("color", (255, 255, 255))

            pygame.draw.circle(self.game_surface, color, 
                             (int(screen_x + radius), int(screen_y + radius)), 
                             int(radius))
            self.render_stats["draw_calls"] += 1

    def render_individual_object(self, game_object: GameObject, screen_pos: Tuple[int, int]):
        """Render individual object (fallback method)"""
        screen_x, screen_y = screen_pos
        obj_type = game_object.object_type

        if obj_type == "rectangle":
            width = game_object.get_property("width", 50)
            height = game_object.get_property("height", 50)
            color = game_object.get_property("color", (255, 255, 255))
            pygame.draw.rect(self.game_surface, color, 
                           pygame.Rect(screen_x, screen_y, width, height))

        elif obj_type == "square":
            size = game_object.get_property("size", 50)
            color = game_object.get_property("color", (255, 255, 255))
            pygame.draw.rect(self.game_surface, color, 
                           pygame.Rect(screen_x, screen_y, size, size))

        elif obj_type == "circle":
            radius = game_object.get_property("radius", 25)
            color = game_object.get_property("color", (255, 255, 255))
            pygame.draw.circle(self.game_surface, color, 
                             (int(screen_x + radius), int(screen_y + radius)), 
                             int(radius))

        elif obj_type == "triangle":
            size = game_object.get_property("size", 50)
            color = game_object.get_property("color", (255, 255, 255))
            points = [
                (screen_x + size // 2, screen_y),
                (screen_x, screen_y + size),
                (screen_x + size, screen_y + size)
            ]
            pygame.draw.polygon(self.game_surface, color, points)

        elif obj_type == "star":
            self.draw_star(screen_x, screen_y, game_object)

        elif obj_type == "hexagon":
            self.draw_hexagon(screen_x, screen_y, game_object)

        elif obj_type == "ellipse":
            width = game_object.get_property("width", 80)
            height = game_object.get_property("height", 50)
            color = game_object.get_property("color", (255, 255, 255))
            pygame.draw.ellipse(self.game_surface, color, 
                              pygame.Rect(screen_x, screen_y, width, height))

        elif obj_type == "polygon":
            points = game_object.get_property("points", [(0, 0), (50, 0), (25, 50)])
            color = game_object.get_property("color", (255, 255, 255))
            world_points = [(screen_x + p[0], screen_y + p[1]) for p in points]
            if len(world_points) >= 3:
                pygame.draw.polygon(self.game_surface, color, world_points)

        self.render_stats["draw_calls"] += 1

    def apply_transformations(self, surface: pygame.Surface, rotation: float, 
                            scale: Tuple[float, float]) -> pygame.Surface:
        """Apply rotation and scaling transformations"""
        # Scale first
        if scale != (1.0, 1.0):
            new_width = int(surface.get_width() * scale[0])
            new_height = int(surface.get_height() * scale[1])
            surface = pygame.transform.scale(surface, (new_width, new_height))

        # Then rotate
        if rotation != 0:
            surface = pygame.transform.rotate(surface, rotation)

        return surface

    def draw_star(self, x: int, y: int, game_object: GameObject):
        """Draw star shape"""
        outer_radius = game_object.get_property("outer_radius", 30)
        inner_radius = game_object.get_property("inner_radius", 15)
        points = game_object.get_property("points", 5)
        color = game_object.get_property("color", (255, 255, 0))

        center_x = x + outer_radius
        center_y = y + outer_radius

        star_points = []
        for i in range(points * 2):
            angle = (i * math.pi) / points
            if i % 2 == 0:
                radius = outer_radius
            else:
                radius = inner_radius

            px = center_x + radius * math.cos(angle - math.pi / 2)
            py = center_y + radius * math.sin(angle - math.pi / 2)
            star_points.append((px, py))

        pygame.draw.polygon(self.game_surface, color, star_points)

    def draw_hexagon(self, x: int, y: int, game_object: GameObject):
        """Draw hexagon shape"""
        radius = game_object.get_property("radius", 30)
        color = game_object.get_property("color", (255, 255, 255))

        center_x = x + radius
        center_y = y + radius

        hex_points = []
        for i in range(6):
            angle = (i * math.pi) / 3
            px = center_x + radius * math.cos(angle)
            py = center_y + radius * math.sin(angle)
            hex_points.append((px, py))

        pygame.draw.polygon(self.game_surface, color, hex_points)

    # POST-PROCESSING EFFECTS
    def add_screen_shake(self, intensity: float, duration: float):
        """Add screen shake effect"""
        self.screen_shake = {
            "intensity": intensity,
            "duration": duration,
            "time": 0
        }

    def update_screen_shake(self, delta_time: float):
        """Update screen shake effect"""
        if self.screen_shake["duration"] > 0:
            self.screen_shake["time"] += delta_time

            if self.screen_shake["time"] >= self.screen_shake["duration"]:
                self.screen_shake["duration"] = 0
                self.camera.shake_offset = (0, 0)
            else:
                # Calculate shake offset
                progress = self.screen_shake["time"] / self.screen_shake["duration"]
                intensity = self.screen_shake["intensity"] * (1 - progress)

                import random
                offset_x = random.uniform(-intensity, intensity)
                offset_y = random.uniform(-intensity, intensity)
                self.camera.shake_offset = (offset_x, offset_y)

    def apply_bloom_effect(self):
        """Apply bloom post-processing effect"""
        if not self.bloom_enabled:
            return

        # Create bloom texture
        bloom_surface = self.render_targets["bloom"]
        bloom_surface.fill((0, 0, 0))

        # Extract bright areas
        bright_surface = pygame.Surface((self.width, self.height))
        bright_surface.fill((0, 0, 0))

        # Simple bloom extraction (threshold-based)
        # This is a simplified version - real bloom would use proper HDR
        for y in range(0, self.height, 4):
            for x in range(0, self.width, 4):
                try:
                    pixel = self.game_surface.get_at((x, y))
                    brightness = (pixel[0] + pixel[1] + pixel[2]) / 3
                    if brightness > 200:  # Threshold for bloom
                        pygame.draw.circle(bright_surface, pixel[:3], (x, y), 3)
                except:
                    pass

        # Blur and add back to main surface
        # Note: This is a simplified blur - real implementation would use gaussian blur
        self.game_surface.blit(bright_surface, (0, 0), special_flags=pygame.BLEND_ADD)

    def apply_chromatic_aberration(self):
        """Apply chromatic aberration effect"""
        if not self.chromatic_aberration:
            return

        # Create separate color channels
        red_surface = pygame.Surface((self.width, self.height))
        green_surface = pygame.Surface((self.width, self.height))
        blue_surface = pygame.Surface((self.width, self.height))

        # Extract color channels with slight offset
        offset = 2

        # This is a simplified version
        red_surface.blit(self.game_surface, (offset, 0))
        green_surface.blit(self.game_surface, (0, 0))
        blue_surface.blit(self.game_surface, (-offset, 0))

        # Combine channels
        self.game_surface.fill((0, 0, 0))
        self.game_surface.blit(red_surface, (0, 0), special_flags=pygame.BLEND_ADD)
        self.game_surface.blit(green_surface, (0, 0), special_flags=pygame.BLEND_ADD)
        self.game_surface.blit(blue_surface, (0, 0), special_flags=pygame.BLEND_ADD)

    # TEXT RENDERING
    def draw_text(self, text: str, x: int, y: int, color: Tuple[int, int, int] = (255, 255, 255), 
                  font: pygame.font.Font = None):
        """Draw text with specified font and color"""
        if font is None:
            font = self.default_font

        text_surface = font.render(str(text), True, color)
        self.ui_surface.blit(text_surface, (x, y))

    def draw_text_centered(self, text: str, x: int, y: int, color: Tuple[int, int, int] = (255, 255, 255),
                          font: pygame.font.Font = None):
        """Draw centered text"""
        if font is None:
            font = self.default_font

        text_surface = font.render(str(text), True, color)
        text_rect = text_surface.get_rect(center=(x, y))
        self.ui_surface.blit(text_surface, text_rect)

    def draw_text_with_shadow(self, text: str, x: int, y: int, color: Tuple[int, int, int] = (255, 255, 255),
                             shadow_color: Tuple[int, int, int] = (0, 0, 0), shadow_offset: int = 2,
                             font: pygame.font.Font = None):
        """Draw text with drop shadow"""
        if font is None:
            font = self.default_font

        # Draw shadow first
        shadow_surface = font.render(str(text), True, shadow_color)
        self.ui_surface.blit(shadow_surface, (x + shadow_offset, y + shadow_offset))

        # Draw main text
        text_surface = font.render(str(text), True, color)
        self.ui_surface.blit(text_surface, (x, y))

    # PRIMITIVE DRAWING
    def draw_line(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int], 
                  color: Tuple[int, int, int], width: int = 1):
        """Draw line"""
        pygame.draw.line(self.game_surface, color, start_pos, end_pos, width)

    def draw_rect(self, x: int, y: int, width: int, height: int, 
                  color: Tuple[int, int, int], filled: bool = True):
        """Draw rectangle"""
        rect = pygame.Rect(x, y, width, height)
        if filled:
            pygame.draw.rect(self.game_surface, color, rect)
        else:
            pygame.draw.rect(self.game_surface, color, rect, 1)

    def draw_circle(self, x: int, y: int, radius: int, color: Tuple[int, int, int], filled: bool = True):
        """Draw circle"""
        if filled:
            pygame.draw.circle(self.game_surface, color, (x, y), radius)
        else:
            pygame.draw.circle(self.game_surface, color, (x, y), radius, 1)

    def draw_sprite(self, x: int, y: int, sprite: pygame.Surface):
        """Draw sprite at position"""
        self.game_surface.blit(sprite, (x, y))

    # FINAL RENDERING
    def present(self):
        """Present final frame to screen"""
        # Update camera effects
        self.update_screen_shake(1/60)  # Assume 60 FPS for now

        # Flush render queue
        self.flush_render_queue()

        # Apply post-processing effects
        if self.post_processing_enabled:
            self.apply_bloom_effect()
            self.apply_chromatic_aberration()

        # Composite all layers
        self.screen.blit(self.background_surface, (0, 0))
        self.screen.blit(self.game_surface, (0, 0))

        # Apply lighting
        if self.lighting_enabled:
            self.render_lighting()
            self.screen.blit(self.lighting_surface, (0, 0), special_flags=pygame.BLEND_MULT)

        # Add effects and UI on top
        self.screen.blit(self.effect_surface, (0, 0))
        self.screen.blit(self.ui_surface, (0, 0))

        # Apply camera shake if active
        if hasattr(self.camera, 'shake_offset') and self.camera.shake_offset != (0, 0):
            # Create temporary surface for shake effect
            shake_surface = self.screen.copy()
            self.screen.fill(self.background_color)
            self.screen.blit(shake_surface, self.camera.shake_offset)

        # Update display
        pygame.display.flip()

    def set_vsync(self, enabled: bool):
        """Enable/disable vertical sync"""
        self.vsync_enabled = enabled

    def get_render_stats(self) -> Dict[str, int]:
        """Get rendering performance statistics"""
        return self.render_stats.copy()

    def enable_advanced_effects(self, bloom: bool = False, chromatic: bool = False, 
                               lighting: bool = True, shadows: bool = False):
        """Enable advanced rendering effects"""
        self.bloom_enabled = bloom
        self.chromatic_aberration = chromatic
        self.lighting_enabled = lighting
        self.shadows_enabled = shadows

        print(f"Advanced effects enabled: Bloom={bloom}, Chromatic={chromatic}, Lighting={lighting}, Shadows={shadows}")

    def cleanup(self):
        """Clean up renderer resources"""
        # Clear all surfaces
        self.render_queue.clear()
        self.render_layers.clear()
        self.lights.clear()
        self.post_effects.clear()