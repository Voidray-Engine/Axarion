"""
Axarion Engine Physics System - Enhanced
Advanced 2D physics simulation with optimizations
"""

import math
from typing import Dict, List, Tuple, Optional, Set
from .game_object import GameObject

class PhysicsSystem:
    """Enhanced 2D physics system with optimizations"""

    def __init__(self):
        self.gravity: Tuple[float, float] = (0.0, 800.0)
        self.enabled = True
        self.collision_enabled = True

        # Enhanced physics constants
        self.min_velocity = 0.1
        self.restitution = 0.3
        self.air_resistance = 0.98
        self.friction_coefficient = 0.7

        # Advanced collision detection
        self.spatial_grid_size = 64
        self.spatial_grid: Dict[str, List[GameObject]] = {}
        self.collision_layers: Dict[str, List[GameObject]] = {}

        # Performance optimizations
        self.use_spatial_partitioning = True
        self.use_broad_phase = True
        self.max_collision_checks = 500
        self.collision_cache: Dict[Tuple[int, int], bool] = {}

        # World bounds
        self.world_bounds = (0, 0, 800, 600)
        self.constrain_to_world = True

        # Advanced features
        self.physics_materials = {}
        self.joint_systems = []
        self.force_fields = []
        self.gravity_wells = []
        self.magnetic_fields = []

        # Performance tracking
        self.collision_checks = 0
        self.broad_phase_culls = 0

    def update(self, delta_time: float):
        """Enhanced physics update with optimizations"""
        if not self.enabled:
            return

        # Reset performance counters
        self.collision_checks = 0
        self.broad_phase_culls = 0

        # Update spatial grid for optimization
        if self.use_spatial_partitioning:
            self._update_spatial_grid()

        # Clean up collision cache periodically
        if len(self.collision_cache) > 1000:
            self.collision_cache.clear()

    def _update_spatial_grid(self):
        """Update spatial partitioning grid for fast collision detection"""
        self.spatial_grid.clear()

        for layer_objects in self.collision_layers.values():
            for obj in layer_objects:
                if not obj.active or obj.destroyed:
                    continue

                bounds = obj.get_bounds()
                grid_keys = self._get_grid_keys(bounds)

                for key in grid_keys:
                    if key not in self.spatial_grid:
                        self.spatial_grid[key] = []
                    self.spatial_grid[key].append(obj)

    def _get_grid_keys(self, bounds: Tuple[float, float, float, float]) -> List[str]:
        """Get spatial grid keys for object bounds"""
        left, top, right, bottom = bounds
        keys = []

        start_x = int(left // self.spatial_grid_size)
        end_x = int(right // self.spatial_grid_size)
        start_y = int(top // self.spatial_grid_size)
        end_y = int(bottom // self.spatial_grid_size)

        for gx in range(start_x, end_x + 1):
            for gy in range(start_y, end_y + 1):
                keys.append(f"{gx},{gy}")

        return keys

    def apply_gravity(self, game_object: GameObject, delta_time: float):
        """Enhanced gravity application"""
        if not self.enabled or game_object.is_static or game_object.gravity_scale <= 0:
            return

        gx, gy = self.gravity
        ax, ay = game_object.acceleration

        # Apply gravity with material modifications
        material = self._get_material(game_object)
        gravity_modifier = material.get('gravity_modifier', 1.0) if material else 1.0

        game_object.acceleration = (
            ax + gx * game_object.gravity_scale * gravity_modifier,
            ay + gy * game_object.gravity_scale * gravity_modifier
        )

    def check_collision(self, obj1: GameObject, obj2: GameObject) -> bool:
        """Enhanced collision detection with caching"""
        if not self.collision_enabled:
            return False

        if not obj1.collision_enabled or not obj2.collision_enabled:
            return False

        if obj1 == obj2:
            return False

        # Check cache first
        cache_key = (id(obj1), id(obj2))
        if cache_key in self.collision_cache:
            return self.collision_cache[cache_key]

        # Broad phase collision detection
        if self.use_broad_phase and not self._broad_phase_collision(obj1, obj2):
            self.broad_phase_culls += 1
            self.collision_cache[cache_key] = False
            return False

        # Detailed collision detection
        result = self._detailed_collision_check(obj1, obj2)
        self.collision_cache[cache_key] = result
        self.collision_checks += 1

        return result

    def _broad_phase_collision(self, obj1: GameObject, obj2: GameObject) -> bool:
        """Fast broad phase collision check using AABB"""
        bounds1 = obj1.get_bounds()
        bounds2 = obj2.get_bounds()

        # Add margin for broad phase
        margin = 10

        return (bounds1[0] - margin < bounds2[2] and bounds1[2] + margin > bounds2[0] and
                bounds1[1] - margin < bounds2[3] and bounds1[3] + margin > bounds2[1])

    def _detailed_collision_check(self, obj1: GameObject, obj2: GameObject) -> bool:
        """Detailed collision detection"""
        if obj1.object_type == "circle" and obj2.object_type == "circle":
            return self._circle_circle_collision(obj1, obj2)
        elif obj1.object_type == "circle" or obj2.object_type == "circle":
            return self._circle_rect_collision(obj1, obj2)
        else:
            return self._rect_rect_collision(obj1, obj2)

    def _circle_circle_collision(self, obj1: GameObject, obj2: GameObject) -> bool:
        """Circle-to-circle collision"""
        x1, y1 = obj1.position
        x2, y2 = obj2.position
        r1 = obj1.properties.get("radius", 25)
        r2 = obj2.properties.get("radius", 25)

        distance_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
        radius_sum_sq = (r1 + r2) ** 2

        return distance_sq <= radius_sum_sq

    def _circle_rect_collision(self, obj1: GameObject, obj2: GameObject) -> bool:
        """Circle-to-rectangle collision"""
        if obj1.object_type == "circle":
            circle, rect = obj1, obj2
        else:
            circle, rect = obj2, obj1

        cx, cy = circle.position
        radius = circle.properties.get("radius", 25)

        bounds = rect.get_bounds()

        # Find closest point on rectangle to circle center
        closest_x = max(bounds[0], min(cx, bounds[2]))
        closest_y = max(bounds[1], min(cy, bounds[3]))

        # Calculate distance from circle center to closest point
        distance_sq = (cx - closest_x) ** 2 + (cy - closest_y) ** 2

        return distance_sq <= radius ** 2

    def _rect_rect_collision(self, obj1: GameObject, obj2: GameObject) -> bool:
        """Rectangle-to-rectangle collision (AABB)"""
        bounds1 = obj1.get_bounds()
        bounds2 = obj2.get_bounds()

        return (bounds1[0] < bounds2[2] and bounds1[2] > bounds2[0] and
                bounds1[1] < bounds2[3] and bounds1[3] > bounds2[1])

    def resolve_collision(self, obj1: GameObject, obj2: GameObject, delta_time: float):
        """Enhanced collision resolution with materials"""
        if not self.check_collision(obj1, obj2):
            return

        # Get material properties
        material1 = self._get_material(obj1)
        material2 = self._get_material(obj2)

        combined_restitution = self._combine_materials(material1, material2, 'restitution')
        combined_friction = self._combine_materials(material1, material2, 'friction')

        if obj1.object_type == "circle" and obj2.object_type == "circle":
            self._resolve_circle_circle_collision(obj1, obj2, combined_restitution)
        else:
            self._resolve_aabb_collision(obj1, obj2, combined_restitution, combined_friction)

    def _resolve_circle_circle_collision(self, obj1: GameObject, obj2: GameObject, restitution: float):
        """Resolve collision between two circles"""
        x1, y1 = obj1.position
        x2, y2 = obj2.position
        r1 = obj1.properties.get("radius", 25)
        r2 = obj2.properties.get("radius", 25)

        # Calculate collision normal
        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx * dx + dy * dy)

        if distance == 0:
            return

        # Normalize
        nx = dx / distance
        ny = dy / distance

        # Separate objects
        overlap = (r1 + r2) - distance
        separation = overlap / 2

        if not obj1.is_static:
            obj1.position = (x1 - nx * separation, y1 - ny * separation)
        if not obj2.is_static:
            obj2.position = (x2 + nx * separation, y2 + ny * separation)

        # Apply impulse for velocity resolution
        self._apply_collision_impulse(obj1, obj2, nx, ny, restitution)

    def _resolve_aabb_collision(self, obj1: GameObject, obj2: GameObject, restitution: float, friction: float):
        """Enhanced AABB collision resolution"""
        bounds1 = obj1.get_bounds()
        bounds2 = obj2.get_bounds()

        # Calculate overlap
        overlap_x = min(bounds1[2], bounds2[2]) - max(bounds1[0], bounds2[0])
        overlap_y = min(bounds1[3], bounds2[3]) - max(bounds1[1], bounds2[1])

        if overlap_x <= 0 or overlap_y <= 0:
            return

        # Determine collision direction
        center1_x = (bounds1[0] + bounds1[2]) / 2
        center1_y = (bounds1[1] + bounds1[3]) / 2
        center2_x = (bounds2[0] + bounds2[2]) / 2
        center2_y = (bounds2[1] + bounds2[3]) / 2

        # Resolve based on smallest overlap
        if overlap_x < overlap_y:
            # Horizontal collision
            if center1_x < center2_x:
                self._resolve_horizontal_collision(obj1, obj2, -1, overlap_x, restitution, friction)
            else:
                self._resolve_horizontal_collision(obj1, obj2, 1, overlap_x, restitution, friction)
        else:
            # Vertical collision
            if center1_y < center2_y:
                self._resolve_vertical_collision(obj1, obj2, -1, overlap_y, restitution, friction)
            else:
                self._resolve_vertical_collision(obj1, obj2, 1, overlap_y, restitution, friction)

    def _resolve_horizontal_collision(self, obj1: GameObject, obj2: GameObject, 
                                    direction: int, overlap: float, restitution: float, friction: float):
        """Resolve horizontal collision"""
        separation = overlap / 2

        if not obj1.is_static:
            obj1.position = (obj1.position[0] - direction * separation, obj1.position[1])
            vx, vy = obj1.velocity
            obj1.velocity = (-direction * abs(vx) * restitution, vy * friction)

        if not obj2.is_static:
            obj2.position = (obj2.position[0] + direction * separation, obj2.position[1])
            vx, vy = obj2.velocity
            obj2.velocity = (direction * abs(vx) * restitution, vy * friction)

    def _resolve_vertical_collision(self, obj1: GameObject, obj2: GameObject, 
                                  direction: int, overlap: float, restitution: float, friction: float):
        """Resolve vertical collision with ground detection"""
        separation = overlap / 2

        if not obj1.is_static:
            obj1.position = (obj1.position[0], obj1.position[1] - direction * separation)
            vx, vy = obj1.velocity

            if direction == -1 and vy > 0:  # Landing on ground
                obj1.velocity = (vx * friction, 0)
                obj1._on_ground = True
            else:
                obj1.velocity = (vx * friction, -direction * abs(vy) * restitution)

        if not obj2.is_static:
            obj2.position = (obj2.position[0], obj2.position[1] + direction * separation)
            vx, vy = obj2.velocity
            obj2.velocity = (vx * friction, direction * abs(vy) * restitution)

    def _apply_collision_impulse(self, obj1: GameObject, obj2: GameObject, 
                               nx: float, ny: float, restitution: float):
        """Apply collision impulse for realistic physics"""
        if obj1.is_static and obj2.is_static:
            return

        # Calculate relative velocity
        v1x, v1y = obj1.velocity if not obj1.is_static else (0, 0)
        v2x, v2y = obj2.velocity if not obj2.is_static else (0, 0)

        rel_vx = v2x - v1x
        rel_vy = v2y - v1y

        # Calculate relative velocity along normal
        vel_along_normal = rel_vx * nx + rel_vy * ny

        # Don't resolve if velocities are separating
        if vel_along_normal > 0:
            return

        # Calculate impulse scalar
        impulse_scalar = -(1 + restitution) * vel_along_normal

        # Apply mass ratio
        if not obj1.is_static and not obj2.is_static:
            impulse_scalar /= (1/obj1.mass + 1/obj2.mass)
        elif obj1.is_static:
            impulse_scalar /= (1/obj2.mass)
        else:
            impulse_scalar /= (1/obj1.mass)

        # Apply impulse
        impulse_x = impulse_scalar * nx
        impulse_y = impulse_scalar * ny

        if not obj1.is_static:
            obj1.velocity = (v1x - impulse_x/obj1.mass, v1y - impulse_y/obj1.mass)
        if not obj2.is_static:
            obj2.velocity = (v2x + impulse_x/obj2.mass, v2y + impulse_y/obj2.mass)

    def check_all_collisions(self, objects: List[GameObject], delta_time: float):
        """Optimized collision detection using spatial partitioning"""
        if not self.collision_enabled:
            return

        active_objects = [obj for obj in objects if obj.active and not obj.destroyed and obj.collision_enabled]

        if self.use_spatial_partitioning and len(active_objects) > 20:
            self._check_collisions_spatial(active_objects, delta_time)
        else:
            self._check_collisions_brute_force(active_objects, delta_time)

    def _check_collisions_spatial(self, objects: List[GameObject], delta_time: float):
        """Spatial partitioning collision detection"""
        checked_pairs: Set[Tuple[int, int]] = set()
        collision_count = 0

        for grid_objects in self.spatial_grid.values():
            if collision_count > self.max_collision_checks:
                break

            for i, obj1 in enumerate(grid_objects):
                for obj2 in grid_objects[i+1:]:
                    if collision_count > self.max_collision_checks:
                        break

                    pair = (min(id(obj1), id(obj2)), max(id(obj1), id(obj2)))
                    if pair in checked_pairs:
                        continue

                    checked_pairs.add(pair)

                    if self.check_collision(obj1, obj2):
                        self.resolve_collision(obj1, obj2, delta_time)

                    collision_count += 1

    def _check_collisions_brute_force(self, objects: List[GameObject], delta_time: float):
        """Brute force collision detection for small object counts"""
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                if self.check_collision(obj1, obj2):
                    self.resolve_collision(obj1, obj2, delta_time)

    def _get_material(self, obj: GameObject) -> Optional[Dict]:
        """Get physics material for object"""
        material_name = obj.get_property("material")
        return self.physics_materials.get(material_name) if material_name else None

    def _combine_materials(self, mat1: Optional[Dict], mat2: Optional[Dict], property_name: str) -> float:
        """Combine material properties"""
        default_values = {
            'restitution': self.restitution,
            'friction': self.friction_coefficient
        }

        val1 = mat1.get(property_name, default_values[property_name]) if mat1 else default_values[property_name]
        val2 = mat2.get(property_name, default_values[property_name]) if mat2 else default_values[property_name]

        # Use geometric mean for combining
        return math.sqrt(val1 * val2)

    def create_material(self, name: str, restitution: float = 0.3, friction: float = 0.7, 
                       density: float = 1.0, **properties) -> Dict:
        """Create physics material"""
        material = {
            'restitution': restitution,
            'friction': friction,
            'density': density,
            **properties
        }
        self.physics_materials[name] = material
        return material

    def apply_force(self, game_object: GameObject, force_x: float, force_y: float):
        """Enhanced force application"""
        if game_object.mass <= 0 or game_object.is_static:
            return

        ax, ay = game_object.acceleration
        game_object.acceleration = (ax + force_x / game_object.mass,
                                   ay + force_y / game_object.mass)

    def apply_impulse(self, game_object: GameObject, impulse_x: float, impulse_y: float):
        """Enhanced impulse application"""
        if game_object.mass <= 0 or game_object.is_static:
            return

        vx, vy = game_object.velocity
        game_object.velocity = (vx + impulse_x / game_object.mass,
                               vy + impulse_y / game_object.mass)

    def constrain_to_bounds(self, game_object: GameObject, bounds: Tuple[float, float, float, float] = None):
        """Enhanced bounds constraint with material bounce"""
        if not self.constrain_to_world:
            return

        if bounds is None:
            bounds = self.world_bounds

        obj_bounds = game_object.get_bounds()
        min_x, min_y, max_x, max_y = bounds
        x, y = game_object.position

        obj_width = obj_bounds[2] - obj_bounds[0]
        obj_height = obj_bounds[3] - obj_bounds[1]

        new_x, new_y = x, y
        vx, vy = game_object.velocity

        # Get material bounce
        material = self._get_material(game_object)
        bounce = material.get('restitution', self.restitution) if material else self.restitution

        # Check horizontal bounds
        if obj_bounds[0] < min_x:
            new_x = min_x
            vx = abs(vx) * bounce
        elif obj_bounds[2] > max_x:
            new_x = max_x - obj_width
            vx = -abs(vx) * bounce

        # Check vertical bounds
        if obj_bounds[1] < min_y:
            new_y = min_y
            vy = abs(vy) * bounce
        elif obj_bounds[3] > max_y:
            new_y = max_y - obj_height
            vy = -abs(vy) * bounce
            game_object._on_ground = True

        game_object.position = (new_x, new_y)
        game_object.velocity = (vx, vy)

    def set_gravity(self, gx: float, gy: float):
        """Set gravity vector"""
        self.gravity = (gx, gy)

    def set_world_bounds(self, left: float, top: float, right: float, bottom: float):
        """Set world bounds"""
        self.world_bounds = (left, top, right, bottom)

    def enable_physics(self, enabled: bool):
        """Enable or disable physics"""
        self.enabled = enabled

    def enable_collision(self, enabled: bool):
        """Enable or disable collision detection"""
        self.collision_enabled = enabled

    def add_to_collision_layer(self, layer_name: str, game_object: GameObject):
        """Add object to collision layer"""
        if layer_name not in self.collision_layers:
            self.collision_layers[layer_name] = []

        if game_object not in self.collision_layers[layer_name]:
            self.collision_layers[layer_name].append(game_object)

        game_object.collision_layer = layer_name

    def remove_from_collision_layer(self, layer_name: str, game_object: GameObject):
        """Remove object from collision layer"""
        if layer_name in self.collision_layers:
            if game_object in self.collision_layers[layer_name]:
                self.collision_layers[layer_name].remove(game_object)

    def get_objects_in_layer(self, layer_name: str) -> List[GameObject]:
        """Get all objects in collision layer"""
        return self.collision_layers.get(layer_name, [])

    def get_performance_stats(self) -> Dict:
        """Get physics performance statistics"""
        return {
            'collision_checks': self.collision_checks,
            'broad_phase_culls': self.broad_phase_culls,
            'spatial_grid_cells': len(self.spatial_grid),
            'cache_size': len(self.collision_cache)
        }

    def cleanup(self):
        """Clean up physics system"""
        self.collision_layers.clear()
        self.spatial_grid.clear()
        self.collision_cache.clear()
        self.physics_materials.clear()

    def check_bounds_collision(self, game_object: GameObject, 
                              bounds: Tuple[float, float, float, float] = None) -> bool:
        """Check if object is colliding with bounds"""
        if bounds is None:
            bounds = self.world_bounds

        obj_bounds = game_object.get_bounds()
        min_x, min_y, max_x, max_y = bounds

        return (obj_bounds[0] < min_x or obj_bounds[2] > max_x or
                obj_bounds[1] < min_y or obj_bounds[3] > max_y)

    def enable_world_bounds(self, enabled: bool):
        """Enable or disable world bounds constraint"""
        self.constrain_to_world = enabled

    def check_layer_collisions(self, layer1: str, layer2: str, delta_time: float):
        """Check collisions between two layers"""
        objects1 = self.get_objects_in_layer(layer1)
        objects2 = self.get_objects_in_layer(layer2)

        for obj1 in objects1:
            for obj2 in objects2:
                if self.check_collision(obj1, obj2):
                    self.resolve_collision(obj1, obj2, delta_time)

    def get_objects_in_range(self, center_x: float, center_y: float, 
                            radius: float, objects: List[GameObject]) -> List[GameObject]:
        """Get all objects within a certain range"""
        result = []
        for obj in objects:
            if not obj.active or obj.destroyed:
                continue

            x, y = obj.position
            distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if distance <= radius:
                result.append(obj)
        return result

    def raycast(self, start_x: float, start_y: float, 
                end_x: float, end_y: float, 
                objects: List[GameObject]) -> Optional[GameObject]:
        """Simple raycast to find first object hit"""
        # This is a simplified raycast implementation
        steps = 100
        dx = (end_x - start_x) / steps
        dy = (end_y - start_y) / steps

        for i in range(steps):
            x = start_x + dx * i
            y = start_y + dy * i

            for obj in objects:
                if obj.active and not obj.destroyed and obj.contains_point(x, y):
                    return obj

        return None

    def circle_cast(self, start_x: float, start_y: float, 
                   end_x: float, end_y: float, radius: float,
                   objects: List[GameObject]) -> List[GameObject]:
        """Cast a circle along a path and return all hit objects"""
        hits = []
        steps = int(math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2) / 5) + 1
        dx = (end_x - start_x) / steps
        dy = (end_y - start_y) / steps

        for i in range(steps):
            x = start_x + dx * i
            y = start_y + dy * i

            nearby = self.get_objects_in_range(x, y, radius, objects)
            for obj in nearby:
                if obj not in hits:
                    hits.append(obj)

        return hits

    def apply_explosion_force(self, center_x: float, center_y: float, 
                            force: float, radius: float, objects: List[GameObject]):
        """Apply explosion force to objects in range"""
        affected_objects = self.get_objects_in_range(center_x, center_y, radius, objects)

        for obj in affected_objects:
            if obj.is_static:
                continue

            x, y = obj.position
            dx = x - center_x
            dy = y - center_y
            distance = math.sqrt(dx * dx + dy * dy)

            if distance == 0:
                continue

            # Calculate force based on distance (inverse square law)
            force_magnitude = force * (1.0 - distance / radius)
            if force_magnitude <= 0:
                continue

            # Normalize direction and apply force
            nx = dx / distance
            ny = dy / distance

            self.apply_impulse(obj, nx * force_magnitude, ny * force_magnitude)

    def step_simulation(self, objects: List[GameObject], delta_time: float):
        """Step the physics simulation for all objects"""
        if not self.enabled:
            return

        # Apply physics to all objects
        for obj in objects:
            if obj.active and not obj.destroyed and not obj.is_static:
                # Apply gravity
                self.apply_gravity(obj, delta_time)

                # Apply air resistance
                vx, vy = obj.velocity
                obj.velocity = (vx * self.air_resistance, vy * self.air_resistance)

                # Constrain to world bounds
                self.constrain_to_bounds(obj)

        # Check all collisions
        self.check_all_collisions(objects, delta_time)

        # Update advanced physics
        #self.update_force_fields(objects, delta_time) #No implementation
        #self.update_joints(delta_time) # No implementation

    # PATHFINDING SYSTEM
    def create_pathfinding_grid(self, width, height, cell_size=32):
        """Create pathfinding grid"""
        self.pathfinding_grid = {
            'width': width,
            'height': height,
            'cell_size': cell_size,
            'obstacles': set()
        }

    def add_obstacle(self, x, y):
        """Add obstacle to pathfinding grid"""
        if hasattr(self, 'pathfinding_grid'):
            grid_x = int(x // self.pathfinding_grid['cell_size'])
            grid_y = int(y // self.pathfinding_grid['cell_size'])
            self.pathfinding_grid['obstacles'].add((grid_x, grid_y))

    def find_path(self, start_x, start_y, end_x, end_y):
        """Find path using A* algorithm"""
        if not hasattr(self, 'pathfinding_grid'):
            return []

        grid = self.pathfinding_grid
        cell_size = grid['cell_size']

        # Convert world coordinates to grid coordinates
        start_grid = (int(start_x // cell_size), int(start_y // cell_size))
        end_grid = (int(end_x // cell_size), int(end_y // cell_size))

        # Simple A* implementation
        from heapq import heappush, heappop

        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self._heuristic(start_grid, end_grid)}

        while open_set:
            current = heappop(open_set)[1]

            if current == end_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append((current[0] * cell_size, current[1] * cell_size))
                    current = came_from[current]
                path.reverse()
                return path

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)

                # Check bounds and obstacles
                if (neighbor[0] < 0 or neighbor[0] >= grid['width'] or
                    neighbor[1] < 0 or neighbor[1] >= grid['height'] or
                    neighbor in grid['obstacles']):
                    continue

                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, end_grid)
                    heappush(open_set, (f_score[neighbor], neighbor))

        return []  # No path found

    def _heuristic(self, a, b):
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])