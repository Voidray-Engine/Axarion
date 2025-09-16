
"""
Advanced Camera System for Axarion Engine
Provides comprehensive camera control, effects, and transitions
"""

import pygame
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable

class CameraTransition:
    """Smooth camera transitions between positions/targets"""
    
    def __init__(self, start_pos: Tuple[float, float], end_pos: Tuple[float, float], 
                 duration: float, easing_func: Optional[Callable] = None):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.duration = duration
        self.elapsed = 0.0
        self.easing_func = easing_func or self._ease_in_out_cubic
        self.active = True
    
    def _ease_in_out_cubic(self, t: float) -> float:
        """Smooth cubic easing function"""
        if t < 0.5:
            return 4 * t * t * t
        else:
            p = 2 * t - 2
            return 1 + p * p * p / 2
    
    def update(self, delta_time: float) -> Tuple[float, float]:
        """Update transition and return current position"""
        if not self.active:
            return self.end_pos
            
        self.elapsed += delta_time
        progress = min(self.elapsed / self.duration, 1.0)
        
        if progress >= 1.0:
            self.active = False
            return self.end_pos
        
        eased_progress = self.easing_func(progress)
        
        x = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * eased_progress
        y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * eased_progress
        
        return (x, y)

class CameraShake:
    """Camera shake effect system"""
    
    def __init__(self):
        self.intensity = 0.0
        self.duration = 0.0
        self.elapsed = 0.0
        self.frequency = 30.0
        self.offset = (0.0, 0.0)
        self.decay_rate = 0.8
    
    def start_shake(self, intensity: float, duration: float, frequency: float = 30.0):
        """Start camera shake effect"""
        self.intensity = intensity
        self.duration = duration
        self.frequency = frequency
        self.elapsed = 0.0
    
    def update(self, delta_time: float) -> Tuple[float, float]:
        """Update shake and return offset"""
        if self.intensity <= 0:
            return (0.0, 0.0)
        
        self.elapsed += delta_time
        
        # Calculate shake offset
        shake_x = math.sin(self.elapsed * self.frequency) * self.intensity
        shake_y = math.cos(self.elapsed * self.frequency * 1.5) * self.intensity
        
        # Decay intensity over time
        decay_factor = max(0, 1 - (self.elapsed / self.duration))
        shake_x *= decay_factor
        shake_y *= decay_factor
        
        if decay_factor <= 0:
            self.intensity = 0
            return (0.0, 0.0)
        
        self.offset = (shake_x, shake_y)
        return self.offset

class CameraZone:
    """Camera behavior zones (trigger areas for camera changes)"""
    
    def __init__(self, rect: pygame.Rect, camera_pos: Tuple[float, float], 
                 zoom: float = 1.0, priority: int = 0):
        self.rect = rect
        self.camera_pos = camera_pos
        self.zoom = zoom
        self.priority = priority
        self.active = True

class Camera:
    """Advanced 2D Camera with multiple features"""
    
    def __init__(self, width: int, height: int, world_width: int = None, world_height: int = None):
        # Basic camera properties
        self.position = [0.0, 0.0]
        self.width = width
        self.height = height
        self.zoom = 1.0
        self.rotation = 0.0
        
        # World boundaries
        self.world_width = world_width or width * 4
        self.world_height = world_height or height * 4
        self.constrain_to_world = False
        
        # Following system
        self.target = None
        self.follow_speed = 5.0
        self.follow_offset = (0.0, 0.0)
        self.follow_deadzone = pygame.Rect(0, 0, 100, 100)
        self.follow_lookahead = True
        self.lookahead_distance = 100.0
        
        # Advanced features
        self.shake = CameraShake()
        self.transition = None
        self.zones = []
        self.current_zone = None
        
        # Viewport and effects
        self.viewport = pygame.Rect(0, 0, width, height)
        self.parallax_layers = []
        
        # Smoothing and prediction
        self.smooth_movement = True
        self.smooth_zoom = True
        self.smooth_rotation = True
        self.movement_smoothing = 0.1
        self.zoom_smoothing = 0.05
        self.rotation_smoothing = 0.1
        
        # Target values for smooth transitions
        self._target_position = [0.0, 0.0]
        self._target_zoom = 1.0
        self._target_rotation = 0.0
        
        # Culling optimization
        self.culling_enabled = True
        self.culling_margin = 100  # Extra margin for culling
        
        # Multi-target support
        self.multiple_targets = []
        self.multi_target_mode = False
        self.multi_target_padding = 200.0
    
    def set_position(self, x: float, y: float, smooth: bool = False):
        """Set camera position"""
        if smooth and self.smooth_movement:
            self._target_position = [x, y]
        else:
            self.position = [x, y]
            self._target_position = [x, y]
    
    def move(self, dx: float, dy: float):
        """Move camera by offset"""
        self.position[0] += dx
        self.position[1] += dy
        self._target_position[0] += dx
        self._target_position[1] += dy
    
    def set_zoom(self, zoom: float, smooth: bool = False):
        """Set camera zoom level"""
        zoom = max(0.1, min(zoom, 10.0))  # Reasonable zoom limits
        if smooth and self.smooth_zoom:
            self._target_zoom = zoom
        else:
            self.zoom = zoom
            self._target_zoom = zoom
    
    def zoom_in(self, factor: float = 1.2):
        """Zoom in by factor"""
        self.set_zoom(self.zoom * factor, smooth=True)
    
    def zoom_out(self, factor: float = 1.2):
        """Zoom out by factor"""
        self.set_zoom(self.zoom / factor, smooth=True)
    
    def set_rotation(self, angle: float, smooth: bool = False):
        """Set camera rotation in degrees"""
        if smooth and self.smooth_rotation:
            self._target_rotation = angle
        else:
            self.rotation = angle
            self._target_rotation = angle
    
    def follow(self, target, speed: float = 5.0, offset: Tuple[float, float] = (0, 0)):
        """Follow a target object"""
        self.target = target
        self.follow_speed = speed
        self.follow_offset = offset
    
    def unfollow(self):
        """Stop following target"""
        self.target = None
    
    def set_follow_deadzone(self, width: int, height: int):
        """Set deadzone for following (camera won't move if target is in this zone)"""
        center_x = self.width // 2
        center_y = self.height // 2
        self.follow_deadzone = pygame.Rect(
            center_x - width // 2,
            center_y - height // 2,
            width, height
        )
    
    def add_multiple_targets(self, targets: List[Any]):
        """Follow multiple targets by framing all of them"""
        self.multiple_targets = targets
        self.multi_target_mode = True
    
    def clear_multiple_targets(self):
        """Clear multiple target following"""
        self.multiple_targets.clear()
        self.multi_target_mode = False
    
    def apply_shake(self, intensity: float, duration: float, frequency: float = 30.0):
        """Add camera shake effect"""
        self.shake.start_shake(intensity, duration, frequency)
    
    def transition_to(self, x: float, y: float, duration: float, 
                     easing_func: Optional[Callable] = None):
        """Smooth transition to position"""
        self.transition = CameraTransition(
            tuple(self.position), (x, y), duration, easing_func
        )
    
    def add_zone(self, rect: pygame.Rect, camera_pos: Tuple[float, float], 
                 zoom: float = 1.0, priority: int = 0):
        """Add camera zone for automatic camera behavior"""
        zone = CameraZone(rect, camera_pos, zoom, priority)
        self.zones.append(zone)
        self.zones.sort(key=lambda z: z.priority, reverse=True)
        return zone
    
    def add_parallax_layer(self, layer_objects: List[Any], scroll_factor: float):
        """Add parallax scrolling layer"""
        self.parallax_layers.append({
            'objects': layer_objects,
            'scroll_factor': scroll_factor
        })
    
    def update(self, delta_time: float):
        """Update camera system"""
        # Handle transitions
        if self.transition and self.transition.active:
            new_pos = self.transition.update(delta_time)
            self.position = list(new_pos)
        
        # Handle target following
        if self.target and hasattr(self.target, 'position'):
            self._update_following(delta_time)
        
        # Handle multiple target following
        if self.multi_target_mode and self.multiple_targets:
            self._update_multiple_target_following(delta_time)
        
        # Check camera zones
        self._update_zones(delta_time)
        
        # Apply smooth movements
        if self.smooth_movement:
            self._apply_smooth_movement(delta_time)
        
        if self.smooth_zoom:
            self._apply_smooth_zoom(delta_time)
        
        if self.smooth_rotation:
            self._apply_smooth_rotation(delta_time)
        
        # Update shake effect
        shake_offset = self.shake.update(delta_time)
        
        # Apply world constraints
        if self.constrain_to_world:
            self._constrain_to_world()
        
        # Update viewport with shake
        self.viewport.x = self.position[0] - self.width // 2 + shake_offset[0]
        self.viewport.y = self.position[1] - self.height // 2 + shake_offset[1]
    
    def _update_following(self, delta_time: float):
        """Update target following logic"""
        if not self.target:
            return
        
        target_pos = getattr(self.target, 'position', (0, 0))
        if hasattr(target_pos, '__iter__') and len(target_pos) >= 2:
            target_x, target_y = target_pos[0], target_pos[1]
        else:
            return
        
        # Add offset
        target_x += self.follow_offset[0]
        target_y += self.follow_offset[1]
        
        # Lookahead prediction
        if self.follow_lookahead and hasattr(self.target, 'velocity'):
            vel = getattr(self.target, 'velocity', (0, 0))
            if hasattr(vel, '__iter__') and len(vel) >= 2:
                target_x += vel[0] * self.lookahead_distance / 100.0
                target_y += vel[1] * self.lookahead_distance / 100.0
        
        # Check deadzone
        screen_target_x = target_x - self.position[0] + self.width // 2
        screen_target_y = target_y - self.position[1] + self.height // 2
        
        if not self.follow_deadzone.collidepoint(screen_target_x, screen_target_y):
            # Move towards target
            dx = target_x - self.position[0]
            dy = target_y - self.position[1]
            
            move_speed = self.follow_speed * delta_time
            self._target_position[0] += dx * move_speed
            self._target_position[1] += dy * move_speed
    
    def _update_multiple_target_following(self, delta_time: float):
        """Update multiple target following"""
        if not self.multiple_targets:
            return
        
        # Get bounding box of all targets
        positions = []
        for target in self.multiple_targets:
            if hasattr(target, 'position'):
                pos = getattr(target, 'position')
                if hasattr(pos, '__iter__') and len(pos) >= 2:
                    positions.append((pos[0], pos[1]))
        
        if len(positions) < 2:
            return
        
        # Calculate bounding box
        min_x = min(pos[0] for pos in positions)
        max_x = max(pos[0] for pos in positions)
        min_y = min(pos[1] for pos in positions)
        max_y = max(pos[1] for pos in positions)
        
        # Center camera on bounding box center
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # Calculate required zoom to fit all targets
        box_width = max_x - min_x + self.multi_target_padding
        box_height = max_y - min_y + self.multi_target_padding
        
        zoom_x = self.width / box_width if box_width > 0 else 1.0
        zoom_y = self.height / box_height if box_height > 0 else 1.0
        required_zoom = min(zoom_x, zoom_y, 2.0)  # Cap zoom at 2x
        
        # Smoothly adjust position and zoom
        move_speed = self.follow_speed * delta_time
        self._target_position[0] += (center_x - self.position[0]) * move_speed
        self._target_position[1] += (center_y - self.position[1]) * move_speed
        self._target_zoom += (required_zoom - self.zoom) * move_speed
    
    def _update_zones(self, delta_time: float):
        """Update camera zones"""
        if self.target and hasattr(self.target, 'position'):
            target_pos = getattr(self.target, 'position', (0, 0))
            if hasattr(target_pos, '__iter__') and len(target_pos) >= 2:
                target_point = (target_pos[0], target_pos[1])
                
                # Find highest priority active zone containing target
                active_zone = None
                for zone in self.zones:
                    if zone.active and zone.rect.collidepoint(target_point):
                        active_zone = zone
                        break
                
                if active_zone != self.current_zone:
                    self.current_zone = active_zone
                    if active_zone:
                        # Transition to zone camera position
                        self.transition_to(
                            active_zone.camera_pos[0],
                            active_zone.camera_pos[1],
                            1.0  # 1 second transition
                        )
                        self.set_zoom(active_zone.zoom, smooth=True)
    
    def _apply_smooth_movement(self, delta_time: float):
        """Apply smooth movement interpolation"""
        lerp_factor = 1 - math.exp(-self.movement_smoothing * 60 * delta_time)
        
        self.position[0] += (self._target_position[0] - self.position[0]) * lerp_factor
        self.position[1] += (self._target_position[1] - self.position[1]) * lerp_factor
    
    def _apply_smooth_zoom(self, delta_time: float):
        """Apply smooth zoom interpolation"""
        lerp_factor = 1 - math.exp(-self.zoom_smoothing * 60 * delta_time)
        self.zoom += (self._target_zoom - self.zoom) * lerp_factor
    
    def _apply_smooth_rotation(self, delta_time: float):
        """Apply smooth rotation interpolation"""
        lerp_factor = 1 - math.exp(-self.rotation_smoothing * 60 * delta_time)
        
        # Handle angle wrapping
        angle_diff = self._target_rotation - self.rotation
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
        
        self.rotation += angle_diff * lerp_factor
    
    def _constrain_to_world(self):
        """Constrain camera to world boundaries"""
        half_width = (self.width / self.zoom) // 2
        half_height = (self.height / self.zoom) // 2
        
        self.position[0] = max(half_width, min(self.position[0], self.world_width - half_width))
        self.position[1] = max(half_height, min(self.position[1], self.world_height - half_height))
    
    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        # Apply camera transformation
        screen_x = (world_x - self.position[0]) * self.zoom + self.width // 2
        screen_y = (world_y - self.position[1]) * self.zoom + self.height // 2
        
        # Apply rotation if needed
        if self.rotation != 0:
            cos_r = math.cos(math.radians(self.rotation))
            sin_r = math.sin(math.radians(self.rotation))
            
            center_x, center_y = self.width // 2, self.height // 2
            rx = screen_x - center_x
            ry = screen_y - center_y
            
            screen_x = center_x + rx * cos_r - ry * sin_r
            screen_y = center_y + rx * sin_r + ry * cos_r
        
        return (int(screen_x), int(screen_y))
    
    def screen_to_world(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates"""
        # Reverse rotation if needed
        if self.rotation != 0:
            cos_r = math.cos(math.radians(-self.rotation))
            sin_r = math.sin(math.radians(-self.rotation))
            
            center_x, center_y = self.width // 2, self.height // 2
            rx = screen_x - center_x
            ry = screen_y - center_y
            
            screen_x = center_x + rx * cos_r - ry * sin_r
            screen_y = center_y + rx * sin_r + ry * cos_r
        
        # Apply inverse camera transformation
        world_x = (screen_x - self.width // 2) / self.zoom + self.position[0]
        world_y = (screen_y - self.height // 2) / self.zoom + self.position[1]
        
        return (world_x, world_y)
    
    def is_visible(self, world_x: float, world_y: float, width: float, height: float) -> bool:
        """Check if object is visible in camera viewport (for culling)"""
        if not self.culling_enabled:
            return True
        
        # Expand viewport by culling margin
        left = self.position[0] - (self.width / self.zoom) // 2 - self.culling_margin
        right = self.position[0] + (self.width / self.zoom) // 2 + self.culling_margin
        top = self.position[1] - (self.height / self.zoom) // 2 - self.culling_margin
        bottom = self.position[1] + (self.height / self.zoom) // 2 + self.culling_margin
        
        # Check if object overlaps with viewport
        return not (world_x + width < left or world_x > right or 
                   world_y + height < top or world_y > bottom)
    
    def get_visible_area(self) -> pygame.Rect:
        """Get the visible world area as a rectangle"""
        half_width = (self.width / self.zoom) // 2
        half_height = (self.height / self.zoom) // 2
        
        return pygame.Rect(
            self.position[0] - half_width,
            self.position[1] - half_height,
            half_width * 2,
            half_height * 2
        )
    
    def reset(self):
        """Reset camera to default state"""
        self.position = [0.0, 0.0]
        self._target_position = [0.0, 0.0]
        self.zoom = 1.0
        self._target_zoom = 1.0
        self.rotation = 0.0
        self._target_rotation = 0.0
        self.target = None
        self.transition = None
        self.shake.intensity = 0
        self.current_zone = None
    
    def save_state(self) -> Dict[str, Any]:
        """Save camera state"""
        return {
            'position': self.position.copy(),
            'zoom': self.zoom,
            'rotation': self.rotation,
            'follow_speed': self.follow_speed,
            'follow_offset': self.follow_offset
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load camera state"""
        self.position = state.get('position', [0.0, 0.0]).copy()
        self._target_position = self.position.copy()
        self.zoom = state.get('zoom', 1.0)
        self._target_zoom = self.zoom
        self.rotation = state.get('rotation', 0.0)
        self._target_rotation = self.rotation
        self.follow_speed = state.get('follow_speed', 5.0)
        self.follow_offset = state.get('follow_offset', (0.0, 0.0))

# Camera manager for multiple cameras
class CameraManager:
    """Manage multiple cameras and camera switching"""
    
    def __init__(self):
        self.cameras = {}
        self.active_camera = None
        self.transition_time = 0.0
        self.transition_duration = 1.0
        self.transitioning = False
        self.from_camera = None
        self.to_camera = None
    
    def add_camera(self, name: str, camera: Camera):
        """Add a camera to the manager"""
        self.cameras[name] = camera
        if self.active_camera is None:
            self.active_camera = camera
    
    def remove_camera(self, name: str):
        """Remove a camera"""
        if name in self.cameras:
            if self.cameras[name] == self.active_camera:
                self.active_camera = None
            del self.cameras[name]
    
    def switch_camera(self, name: str, transition_duration: float = 1.0):
        """Switch to different camera with transition"""
        if name in self.cameras and self.cameras[name] != self.active_camera:
            self.from_camera = self.active_camera
            self.to_camera = self.cameras[name]
            self.transition_duration = transition_duration
            self.transition_time = 0.0
            self.transitioning = True
    
    def update(self, delta_time: float):
        """Update camera manager"""
        if self.active_camera:
            self.active_camera.update(delta_time)
        
        # Handle camera transitions
        if self.transitioning and self.from_camera and self.to_camera:
            self.transition_time += delta_time
            progress = min(self.transition_time / self.transition_duration, 1.0)
            
            if progress >= 1.0:
                self.active_camera = self.to_camera
                self.transitioning = False
                self.from_camera = None
                self.to_camera = None
            else:
                # Interpolate between cameras during transition
                # This is a simplified version - you might want more sophisticated blending
                self.active_camera = self.to_camera
    
    def get_active_camera(self) -> Optional[Camera]:
        """Get the currently active camera"""
        return self.active_camera
