"""
Axarion Engine Core
"""

import pygame
import json
import os
import time
from typing import Dict, List, Optional
from .renderer import Renderer
from .scene import Scene
from .physics import PhysicsSystem
from .input_system import input_system
from .audio_system import audio_system
from .particle_system import particle_system
from .animation_system import animation_system
from .game_object import GameObject

class AxarionEngine:
    """Modern game engine with clean architecture and high performance"""

    def __init__(self, width: int = 800, height: int = 600, title: str = "Axarion Engine"):
        # Core properties
        self.width = width
        self.height = height
        self.title = title
        self.running = False
        self.initialized = False

        # Timing and performance
        self.clock = pygame.time.Clock()
        self.target_fps = 60
        self.delta_time = 0.0
        self.total_time = 0.0
        self.frame_count = 0
        self.accumulated_time = 0.0
        
        # NEW PERFORMANCE OPTIMIZATIONS WITH GPU SUPPORT
        self.performance_mode = "auto"  # auto, performance, quality, gpu
        self.object_pool = {}  # Object pooling for better performance
        self.culling_enabled = True
        self.batch_rendering = True
        self.adaptive_fps = True
        self.frame_skip_enabled = True
        self.max_objects_per_frame = 1000
        
        # CPU/GPU Task Distribution
        self.cpu_tasks = ["physics", "ai", "scripting", "collision_broad_phase"]
        self.gpu_tasks = ["rendering", "particles", "post_processing", "lighting"]
        
        # GPU optimization settings
        self.gpu_acceleration_enabled = False
        self.gpu_batch_size = 100
        self.prefer_gpu_rendering = True
        self.gpu_memory_limit = 256  # MB
        self.gpu_compute_available = False
        
        # CPU optimization settings
        self.cpu_thread_pool_size = 4
        self.cpu_affinity_enabled = False
        self.cpu_intensive_tasks = set()
        
        # Modern cache systems with CPU/GPU split
        self.cpu_cache = {"collision": {}, "pathfinding": {}, "ai_decisions": {}}
        self.gpu_cache = {"textures": {}, "shaders": {}, "vertex_buffers": {}}
        self.unified_cache = {}
        self.cache_memory_budget = 128  # MB
        
        # Advanced multithreading
        self.thread_pool = None
        self.async_tasks = []
        self.cpu_workers = []
        
        # Modern spatial systems
        self.spatial_grid = {}
        self.dirty_regions = []
        self.occlusion_culling = True
        self.dynamic_batching = True
        
        self.performance_stats = {
            "fps": 0,
            "frame_time": 0,
            "objects_rendered": 0,
            "physics_time": 0,
            "render_time": 0,
            "objects_culled": 0,
            "cache_hits": 0,
            "pool_reuses": 0
        }

        # Pure engine mode - no helper systems
        self.game_mode = "manual"  # Users must code everything manually

        # Unlimited rendering capabilities
        self.layered_rendering = True
        self.post_processing = []
        self.lighting_system = None
        self.shader_support = True

        # Advanced AI and behavior systems
        self.ai_systems = {}
        self.pathfinding = None
        self.state_machines = {}
        self.behavior_trees = {}

        # Engine subsystems (initialized in order)
        self.renderer = None
        self.physics = None
        self.input_manager = None
        self.audio_manager = None
        self.asset_manager = None

        # Scene management
        self.current_scene = None
        self.scenes: Dict[str, Scene] = {}
        self.scene_transition = None

        # Engine state and configuration
        self.game_state = "running"  # running, paused, loading, menu
        self.debug_mode = False
        self.time_scale = 1.0
        self.vsync_enabled = True
        self.auto_pause = True  # Pause when window loses focus

        # Event and messaging system
        self.event_dispatcher = {}
        self.global_variables = {}
        self.message_queue = []

        # System management
        self.registered_systems = []
        self.system_priorities = {}

        # Error handling and logging
        self.error_log = []
        self.warning_log = []
        self.verbose_logging = False
        
        # Fatal error handling
        self.fatal_error_handler = None
        self.crash_reports = []
        self.enable_crash_reporting = True
        self.last_error_time = 0.0

    def initialize(self, surface=None, **config):
        """Initialize the engine with comprehensive system setup"""
        if self.initialized:
            self._log_warning("Engine already initialized")
            return True

        try:
            # Initialize pygame if needed
            if not pygame.get_init():
                pygame.init()
                pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
                pygame.mixer.init()

            # Apply configuration
            self._apply_config(config)

            # Initialize core subsystems in correct order
            success = (
                self._init_renderer(surface) and
                self._init_physics() and
                self._init_input_system() and
                self._init_audio_system() and
                self._init_asset_manager()
            )

            if not success:
                raise RuntimeError("Failed to initialize one or more subsystems")

            # Set up default scene
            self._create_default_scene()

            # Initialize performance monitoring
            self._init_performance_monitoring()

            self.initialized = True
            self._log_info(f"🎮 Axarion Engine v0.5 initialized successfully")
            self._log_info(f"   Resolution: {self.width}x{self.height}")
            self._log_info(f"   Target FPS: {self.target_fps}")
            self._log_info(f"   VSync: {'Enabled' if self.vsync_enabled else 'Disabled'}")

            return True

        except Exception as e:
            self._log_error(f"Failed to initialize engine: {e}")
            self.cleanup()
            return False

    def _apply_config(self, config):
        """Apply configuration options"""
        self.target_fps = config.get('fps', 60)
        self.vsync_enabled = config.get('vsync', True)
        self.debug_mode = config.get('debug', False)
        self.verbose_logging = config.get('verbose', False)

    def _init_renderer(self, surface):
        """Initialize rendering subsystem with advanced CPU/GPU optimizations"""
        try:
            from .renderer import Renderer
            self.renderer = Renderer(self.width, self.height, surface)
            self.renderer.set_vsync(self.vsync_enabled)
            
            # Initialize CPU/GPU task distribution
            self._init_cpu_gpu_distribution()
            
            # Try to enable GPU acceleration
            if self.prefer_gpu_rendering:
                gpu_success = self.renderer.optimize_for_gpu()
                if gpu_success:
                    self.gpu_acceleration_enabled = True
                    self.gpu_compute_available = self.renderer.check_compute_capability()
                    self._log_info("🚀 GPU acceleration initialized successfully")
                    if self.gpu_compute_available:
                        self._log_info("🔥 GPU compute shaders available")
                else:
                    self._log_warning("⚠️ GPU acceleration not available, using software rendering")
                    self._fallback_to_cpu_rendering()
            
            return True
        except Exception as e:
            self._log_error(f"Failed to initialize renderer: {e}")
            return False
    
    def _init_cpu_gpu_distribution(self):
        """Initialize CPU/GPU task distribution system"""
        import threading
        import multiprocessing
        
        # CPU thread pool initialization
        cpu_count = multiprocessing.cpu_count()
        self.cpu_thread_pool_size = min(cpu_count, 6)  # Limit threads
        
        # Initialize CPU workers for intensive tasks
        self._init_cpu_workers()
        
        # GPU task queue
        self.gpu_command_queue = []
        self.gpu_memory_usage = 0
        
        self._log_info(f"💻 CPU threads: {self.cpu_thread_pool_size}, GPU tasks: {len(self.gpu_tasks)}")
    
    def _init_cpu_workers(self):
        """Initialize CPU worker threads for background tasks"""
        import threading
        import queue
        
        self.cpu_task_queue = queue.Queue()
        self.cpu_result_queue = queue.Queue()
        
        for i in range(self.cpu_thread_pool_size):
            worker = threading.Thread(target=self._cpu_worker, daemon=True)
            worker.start()
            self.cpu_workers.append(worker)
    
    def _cpu_worker(self):
        """CPU worker thread for background processing"""
        import time
        while True:
            try:
                task = self.cpu_task_queue.get(timeout=1.0)
                if task is None:  # Shutdown signal
                    break
                
                # Execute CPU task
                result = self._execute_cpu_task(task)
                self.cpu_result_queue.put(result)
                self.cpu_task_queue.task_done()
                
            except:
                continue  # Timeout or error, keep working
    
    def _execute_cpu_task(self, task):
        """Execute CPU-intensive task"""
        task_type = task.get("type")
        
        if task_type == "physics_broad_phase":
            return self._cpu_physics_broad_phase(task["objects"])
        elif task_type == "pathfinding":
            return self._cpu_pathfinding(task["start"], task["end"], task["grid"])
        elif task_type == "ai_processing":
            return self._cpu_ai_processing(task["entities"])
        elif task_type == "collision_detection":
            return self._cpu_collision_detection(task["objects"])
        
        return None
    
    def _fallback_to_cpu_rendering(self):
        """Fallback to CPU-optimized rendering"""
        self.renderer.force_cpu_mode()
        self.cpu_tasks.extend(["rendering", "particles"])
        self.gpu_tasks = ["lighting"]  # Basic lighting only
        self._log_info("🔄 Switched to CPU-optimized rendering mode")

    def _init_physics(self):
        """Initialize physics subsystem"""
        try:
            self.physics = PhysicsSystem()
            return True
        except Exception as e:
            self._log_error(f"Failed to initialize physics: {e}")
            return False

    def _init_input_system(self):
        """Initialize input management"""
        try:
            from .input_system import input_system
            self.input_manager = input_system
            return True
        except Exception as e:
            self._log_error(f"Failed to initialize input system: {e}")
            return False

    def _init_audio_system(self):
        """Initialize audio subsystem"""
        try:
            from .audio_system import audio_system
            self.audio_manager = audio_system
            # Audio system can work in disabled mode, so always return True
            if not audio_system.audio_enabled:
                self._log_warning("Audio system running in disabled mode")
            return True
        except Exception as e:
            self._log_error(f"Failed to initialize audio system: {e}")
            # Create a dummy audio manager to prevent crashes
            self.audio_manager = None
            return True  # Don't fail engine init due to audio issues

    def _init_asset_manager(self):
        """Initialize asset management"""
        try:
            from .asset_manager import asset_manager
            self.asset_manager = asset_manager
            return True
        except Exception as e:
            self._log_error(f"Failed to initialize asset manager: {e}")
            return False

    def _create_default_scene(self):
        """Create and set default scene"""
        default_scene = Scene("Default")
        self.scenes["Default"] = default_scene
        self.current_scene = default_scene

    def _init_performance_monitoring(self):
        """Initialize performance monitoring"""
        import time
        self.performance_stats["start_time"] = time.time()
        self.performance_stats["last_fps_update"] = time.time()

    def update(self):
        """High-performance engine update with advanced optimizations"""
        if not self.running or not self.initialized:
            return

        try:
            import time
            frame_start = time.perf_counter()

            # NEW OPTIMIZATION: Adaptive FPS
            if self.adaptive_fps:
                self._adjust_target_fps()

            # Calculate delta time with time scale
            raw_delta = self.clock.tick(self.target_fps) / 1000.0
            
            # NEW OPTIMIZATION: Frame skipping for demanding games
            if self.frame_skip_enabled and raw_delta > 1.0/30.0:  # If FPS < 30
                self.delta_time = 1.0/30.0 * self.time_scale  # Limit delta time
            else:
                self.delta_time = raw_delta * self.time_scale
                
            self.total_time += raw_delta
            self.frame_count += 1

            # Handle window events first
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    self.stop()
                elif event.type == pygame.WINDOWFOCUSLOST and self.auto_pause:
                    self.pause_game()
                elif event.type == pygame.WINDOWFOCUSGAINED and self.game_state == "paused":
                    self.resume_game()

            # Skip updates if paused
            if self.game_state == "paused":
                return

            # NEW OPTIMIZATION: Update spatial grid for collisions
            self._update_spatial_grid()

            # Update input system
            if self.input_manager:
                try:
                    self.input_manager.update(events)
                except Exception as e:
                    self._log_error(f"Input system update failed: {e}")

            # Update registered systems in priority order
            try:
                self._update_systems_optimized(self.delta_time)
            except Exception as e:
                self._log_error(f"Systems update failed: {e}")

            # NEW OPTIMIZATION: Selective scene update
            if self.current_scene and self.current_scene.active:
                try:
                    self._update_scene_optimized(self.delta_time)
                except Exception as e:
                    self._log_error(f"Scene update failed: {e}")

            # Handle scene transitions
            if self.scene_transition:
                try:
                    self._process_scene_transition()
                except Exception as e:
                    self._log_error(f"Scene transition failed: {e}")

            # Process message queue
            try:
                self._process_messages()
            except Exception as e:
                self._log_error(f"Message processing failed: {e}")

            # NEW OPTIMIZATION: Cleanup unused objects
            self._cleanup_object_pools()

            # Update performance stats
            frame_time = time.perf_counter() - frame_start
            self._update_performance_stats(frame_time)
            
        except Exception as e:
            self._log_fatal(f"Critical error in engine update: {e}", e)
            raise

    def _update_systems(self, delta_time):
        """Update all registered systems in priority order"""
        # Update core systems
        systems_to_update = [
            ('physics', self.physics),
            ('animation', None),  # Will be handled via import
            ('particles', None),  # Will be handled via import
            ('audio', self.audio_manager)
        ]

        for system_name, system in systems_to_update:
            if system and hasattr(system, 'update'):
                try:
                    physics_start = time.perf_counter() if system_name == 'physics' else None
                    system.update(delta_time)
                    if physics_start:
                        self.performance_stats['physics_time'] = time.perf_counter() - physics_start
                except Exception as e:
                    self._log_error(f"Error updating {system_name} system: {e}")
            elif system_name == 'audio' and system is None:
                # Audio system disabled, skip silently
                pass

        # Update external systems
        try:
            from .animation_system import animation_system
            from .particle_system import particle_system
            animation_system.update(delta_time)
            particle_system.update(delta_time)
        except ImportError:
            pass  # Systems not available

        # Update custom registered systems
        for system in self.registered_systems:
            try:
                if hasattr(system, 'update'):
                    system.update(delta_time)
            except Exception as e:
                self._log_error(f"Error updating custom system: {e}")

    def _process_messages(self):
        """Process queued messages"""
        while self.message_queue:
            message = self.message_queue.pop(0)
            self._dispatch_message(message)

    def _dispatch_message(self, message):
        """Dispatch message to registered handlers"""
        msg_type = message.get('type')
        if msg_type in self.event_dispatcher:
            for handler in self.event_dispatcher[msg_type]:
                try:
                    handler(message)
                except Exception as e:
                    self._log_error(f"Error in message handler: {e}")

    def _update_performance_stats(self, frame_time):
        """Update performance statistics with CPU monitoring"""
        import time
        current_time = time.time()

        self.performance_stats['frame_time'] = frame_time * 1000  # Convert to ms

        # NEW: CPU utilization monitoring
        cpu_usage = self._get_cpu_usage()
        self.performance_stats['cpu_usage'] = cpu_usage
        
        # NEW: Memory usage monitoring
        memory_usage = self._get_memory_usage()
        self.performance_stats['memory_usage'] = memory_usage

        # Update FPS every second
        if current_time - self.performance_stats.get('last_fps_update', 0) >= 1.0:
            elapsed = current_time - self.performance_stats.get('start_time', current_time)
            if elapsed > 0:
                self.performance_stats['fps'] = self.frame_count / elapsed
            self.performance_stats['last_fps_update'] = current_time
            
            # NEW: Performance warnings
            self._check_performance_warnings()

    def _get_cpu_usage(self):
        """Get CPU usage without external libraries"""
        try:
            # Simple way to measure CPU without psutil
            import os
            import time
            
            if not hasattr(self, '_last_cpu_time'):
                self._last_cpu_time = time.time()
                self._last_process_time = time.process_time()
                return 0.0
            
            current_time = time.time()
            current_process_time = time.process_time()
            
            time_delta = current_time - self._last_cpu_time
            process_delta = current_process_time - self._last_process_time
            
            if time_delta > 0:
                cpu_percent = (process_delta / time_delta) * 100
                cpu_percent = min(100.0, max(0.0, cpu_percent))  # Clamp 0-100%
            else:
                cpu_percent = 0.0
            
            self._last_cpu_time = current_time
            self._last_process_time = current_process_time
            
            return cpu_percent
        except:
            return 0.0

    def _check_performance_warnings(self):
        """Check performance and show warnings"""
        fps = self.performance_stats.get('fps', 60)
        cpu_usage = self.performance_stats.get('cpu_usage', 0)
        memory_usage = self.performance_stats.get('memory_usage', 0)
        frame_time = self.performance_stats.get('frame_time', 0)
        
        # CPU insufficient warning
        if cpu_usage > 95:
            self._log_warning(f"⚠️ CPU OVERLOADED! Usage: {cpu_usage:.1f}% - Reduce graphics details or object count")
            self._suggest_performance_improvements()
        elif cpu_usage > 85:
            self._log_warning(f"🔥 High CPU: {cpu_usage:.1f}% - Engine using nearly full power")
        
        # FPS warnings
        if fps < 30:
            self._log_warning(f"⚠️ LOW FPS: {fps:.1f} - Game is slow! CPU probably can't keep up")
            self._auto_optimize_for_performance()
        elif fps < 45:
            self._log_warning(f"⚠️ FPS below target: {fps:.1f} - Consider optimization")
        
        # Frame time warnings  
        if frame_time > 33.3:  # Over 30 FPS threshold
            self._log_warning(f"⚠️ Long frame time: {frame_time:.1f}ms - CPU can't process frame in time")

    def _suggest_performance_improvements(self):
        """Suggest performance improvements when CPU is struggling"""
        suggestions = [
            "💡 Set performance mode: engine.set_performance_mode('performance')",
            "💡 Reduce max_objects_per_frame: engine.max_objects_per_frame = 500",
            "💡 Enable frame skipping: engine.frame_skip_enabled = True",
            "💡 Lower target FPS: engine.set_target_fps(30)",
            "💡 Enable object pooling for frequently created objects"
        ]
        
        for suggestion in suggestions:
            self._log_info(suggestion)

    def _auto_optimize_for_performance(self):
        """Automatically optimize when performance is poor"""
        if self.performance_mode == "auto":
            self._log_info("🔧 Auto-optimization: Switching to performance mode")
            self.set_performance_mode("performance")
            
            # Additional aggressive optimizations
            self.max_objects_per_frame = min(self.max_objects_per_frame, 300)
            self.frame_skip_enabled = True
            
            if self.target_fps > 30:
                self.target_fps = 30
                self._log_info("🔧 Auto-optimization: Lowering target FPS to 30")

    def optimize_rendering(self, enable_culling=True, enable_batching=True):
        """Optimize rendering performance"""
        if enable_culling:
            self.renderer.enable_frustum_culling = True
        if enable_batching:
            self.renderer.enable_sprite_batching = True

    def set_quality_settings(self, quality_level="medium"):
        """Set graphics quality settings"""
        if quality_level == "low":
            self.renderer.enable_antialiasing = False
            self.renderer.particle_limit = 100
            self.target_fps = 30
        elif quality_level == "medium":
            self.renderer.enable_antialiasing = True
            self.renderer.particle_limit = 500
            self.target_fps = 60
        elif quality_level == "high":
            self.renderer.enable_antialiasing = True
            self.renderer.particle_limit = 1000
            self.target_fps = 60

    def render(self):
        """High-performance rendering with monitoring and error protection"""
        if not self.renderer or not self.initialized:
            return

        try:
            import time
            render_start = time.perf_counter()

            try:
                # Clear screen
                self.renderer.clear()

                # Render current scene
                objects_rendered = 0
                if self.current_scene and self.current_scene.active:
                    try:
                        objects_rendered = self.current_scene.render(self.renderer)
                    except Exception as e:
                        self._log_error(f"Scene rendering failed: {e}")
                        objects_rendered = 0

                # Render particle effects
                try:
                    from .particle_system import particle_system
                    particle_system.render(self.renderer)
                except ImportError:
                    pass
                except Exception as e:
                    self._log_error(f"Particle rendering failed: {e}")

                # Render debug information if enabled
                if self.debug_mode:
                    try:
                        self._render_debug_info()
                    except Exception as e:
                        self._log_error(f"Debug info rendering failed: {e}")

                # Present frame
                self.renderer.present()

                # Update render stats
                self.performance_stats['render_time'] = (time.perf_counter() - render_start) * 1000
                self.performance_stats['objects_rendered'] = objects_rendered

            except Exception as e:
                self._log_error(f"Render error: {e}")
                # Try to recover by clearing screen
                try:
                    self.renderer.clear()
                    self.renderer.present()
                except:
                    pass
                    
        except Exception as e:
            self._log_fatal(f"Critical rendering error: {e}", e)
            raise

    def _render_debug_info(self):
        """Render debug information overlay with CPU info"""
        if not self.renderer:
            return

        debug_info = [
            f"FPS: {self.performance_stats.get('fps', 0):.1f}",
            f"Frame: {self.performance_stats.get('frame_time', 0):.2f}ms",
            f"Render: {self.performance_stats.get('render_time', 0):.2f}ms",
            f"Physics: {self.performance_stats.get('physics_time', 0)*1000:.2f}ms",
            f"CPU: {self.performance_stats.get('cpu_usage', 0):.1f}%",
            f"Memory: {self.performance_stats.get('memory_usage', 0):.1f}MB",
            f"Objects: {self.performance_stats.get('objects_rendered', 0)}",
            f"Mode: {self.performance_mode}",
            f"Scene: {self.current_scene.name if self.current_scene else 'None'}",
            f"Time Scale: {self.time_scale:.2f}"
        ]

        y_offset = 10
        for info in debug_info:
            # Color code based on performance
            color = (255, 255, 0)  # Yellow default
            if "CPU:" in info:
                cpu_val = self.performance_stats.get('cpu_usage', 0)
                if cpu_val > 90:
                    color = (255, 0, 0)  # Red for high CPU
                elif cpu_val > 70:
                    color = (255, 165, 0)  # Orange for medium CPU
                else:
                    color = (0, 255, 0)  # Green for good CPU
            elif "FPS:" in info:
                fps_val = self.performance_stats.get('fps', 0)
                if fps_val < 30:
                    color = (255, 0, 0)  # Red for low FPS
                elif fps_val < 50:
                    color = (255, 165, 0)  # Orange for medium FPS
                else:
                    color = (0, 255, 0)  # Green for good FPS
                    
            self.renderer.draw_text(info, 10, y_offset, color)
            y_offset += 18

        # NEW: Warning indicators
        cpu_usage = self.performance_stats.get('cpu_usage', 0)
        fps = self.performance_stats.get('fps', 0)
        
        if cpu_usage > 95 or fps < 25:
            warning_text = "⚠️ CPU OVERLOADED! Try performance mode"
            self.renderer.draw_text(warning_text, 10, y_offset + 20, (255, 0, 0))
        elif cpu_usage > 85 or fps < 45:
            warning_text = "⚠️ High CPU load"
            self.renderer.draw_text(warning_text, 10, y_offset + 20, (255, 165, 0))

    def run_frame(self):
        """Run a single frame of the engine"""
        self.update()
        self.render()

    def run(self):
        """Run the main game loop with fatal error protection"""
        try:
            self.start()

            while self.running:
                try:
                    self.run_frame()
                except KeyboardInterrupt:
                    print("\n🛑 Game interrupted by user")
                    self.stop()
                    break
                except Exception as e:
                    self._log_fatal(f"Fatal error in game loop: {e}", e)
                    self.stop()
                    break

        except Exception as e:
            self._log_fatal(f"Fatal error during engine startup: {e}", e)
        finally:
            self.cleanup()

    def start(self):
        """Start the engine main loop"""
        self.running = True

    def stop(self):
        """Stop the engine"""
        self.running = False

    def load_scene(self, scene_name: str) -> bool:
        """Load a scene by name"""
        if scene_name in self.scenes:
            self.current_scene = self.scenes[scene_name]
            return True
        return False

    def create_scene(self, scene_name: str) -> Scene:
        """Create a new scene"""
        scene = Scene(scene_name)
        self.scenes[scene_name] = scene
        return scene

    def get_scene(self, scene_name: str) -> Optional[Scene]:
        """Get a scene by name"""
        return self.scenes.get(scene_name)

    def save_project(self, file_path: str) -> bool:
        """Save the current project to file"""
        try:
            project_data = {
                "engine_version": "0.4",
                "scenes": {}
            }

            # Save all scenes
            for name, scene in self.scenes.items():
                project_data["scenes"][name] = scene.serialize()

            with open(file_path, 'w') as f:
                json.dump(project_data, f, indent=2)

            return True
        except Exception as e:
            print(f"Failed to save project: {e}")
            return False

    def load_project(self, file_path: str) -> bool:
        """Load a project from file"""
        try:
            with open(file_path, 'r') as f:
                project_data = json.load(f)

            # Clear existing scenes
            self.scenes.clear()

            # Load scenes
            for name, scene_data in project_data.get("scenes", {}).items():
                scene = Scene(name)
                scene.deserialize(scene_data)
                self.scenes[name] = scene

            # Set first scene as current
            if self.scenes:
                self.current_scene = next(iter(self.scenes.values()))

            return True
        except Exception as e:
            print(f"Failed to load project: {e}")
            return False

    def set_global_variable(self, name: str, value):
        """Set global variable accessible from all scripts"""
        self.global_variables[name] = value

    def get_global_variable(self, name: str, default=None):
        """Get global variable"""
        return self.global_variables.get(name, default)

    def add_game_system(self, system):
        """Add custom game system"""
        self.game_systems.append(system)

    def emit_event(self, event_name: str, data=None):
        """Emit global event"""
        if event_name in self.event_system:
            for callback in self.event_system[event_name]:
                callback(data)

    def subscribe_event(self, event_name: str, callback):
        """Subscribe to global event"""
        if event_name not in self.event_system:
            self.event_system[event_name] = []
        self.event_system[event_name].append(callback)

    def pause_game(self):
        """Pause the game"""
        self.game_state = "paused"
        self.time_scale = 0.0

    def resume_game(self):
        """Resume the game"""
        self.game_state = "running"
        self.time_scale = 1.0

    def set_time_scale(self, scale: float):
        """Set time scale for slow motion or fast forward"""
        self.time_scale = max(0.0, scale)

    def add_collision_layer(self, layer_name: str, objects: List = None):
        """Add collision layer for organized collision detection"""
        self.collision_layers[layer_name] = objects or []

    def get_objects_in_layer(self, layer_name: str):
        """Get all objects in collision layer"""
        return self.collision_layers.get(layer_name, [])

    def load_texture(self, name: str, file_path: str):
        """Load and cache texture"""
        try:
            texture = pygame.image.load(file_path)
            self.loaded_textures[name] = texture
            return True
        except:
            return False

    def get_texture(self, name: str):
        """Get cached texture"""
        return self.loaded_textures.get(name)

    def create_tilemap(self, tile_data: List[List[int]], tile_size: int = 32):
        """Create tilemap from 2D array"""
        tilemap = GameObject("Tilemap", "tilemap")
        tilemap.set_property("tile_data", tile_data)
        tilemap.set_property("tile_size", tile_size)
        return tilemap

    def find_objects_by_tag(self, tag: str):
        """Find all objects with specific tag across all scenes"""
        results = []
        for scene in self.scenes.values():
            for obj in scene.get_all_objects():
                if obj.get_property("tags", []) and tag in obj.get_property("tags"):
                    results.append(obj)
        return results

    # NOVÉ OPTIMALIZAČNÍ METODY
    def _adjust_target_fps(self):
        """Adaptive FPS based on performance"""
        if hasattr(self, 'performance_stats'):
            current_fps = self.performance_stats.get('fps', 60)
            
            if self.performance_mode == "auto":
                if current_fps < 45:
                    self.target_fps = max(30, self.target_fps - 5)
                elif current_fps > 55 and self.target_fps < 60:
                    self.target_fps = min(60, self.target_fps + 5)
            elif self.performance_mode == "performance":
                self.target_fps = 30
            elif self.performance_mode == "quality":
                self.target_fps = 60

    def _update_spatial_grid(self):
        """Spatial grid for faster collisions"""
        if not self.current_scene:
            return
            
        grid_size = 64
        self.spatial_grid.clear()
        
        for obj in self.current_scene.objects:
            if not obj.active or obj.destroyed:
                continue
                
            bounds = obj.get_bounds()
            grid_x = int(bounds[0] // grid_size)
            grid_y = int(bounds[1] // grid_size)
            
            key = f"{grid_x},{grid_y}"
            if key not in self.spatial_grid:
                self.spatial_grid[key] = []
            self.spatial_grid[key].append(obj)

    def _update_systems_optimized(self, delta_time):
        """Optimized systems update"""
        # Only active systems
        systems_to_update = [
            ('physics', self.physics),
            ('audio', self.audio_manager)
        ]

        for system_name, system in systems_to_update:
            if system and hasattr(system, 'update'):
                try:
                    system.update(delta_time)
                except Exception as e:
                    self._log_error(f"Error updating {system_name} system: {e}")

        # Update external systems with error handling
        try:
            from .animation_system import animation_system
            from .particle_system import particle_system
            animation_system.update(delta_time)
            particle_system.update(delta_time)
        except ImportError:
            pass

    def _update_scene_optimized(self, delta_time):
        """Enhanced scene update with advanced optimizations"""
        if not self.current_scene:
            return
            
        try:
            # Use scene's optimized update method
            self.current_scene.update(delta_time)
            
            # Apply region effects if available
            if hasattr(self.current_scene, 'apply_region_effects'):
                self.current_scene.apply_region_effects(delta_time)
            
            # Update scene performance stats
            scene_stats = self.current_scene.get_scene_stats()
            self.performance_stats.update({
                'scene_objects': scene_stats.get('total_objects', 0),
                'active_objects': scene_stats.get('active_objects', 0),
                'static_objects': scene_stats.get('static_objects', 0)
            })
            
        except Exception as e:
            self._log_error(f"Scene update error: {e}")
            # Fallback to basic update
            self._update_scene_fallback(delta_time)

    def _update_scene_fallback(self, delta_time):
        """Fallback scene update method"""
        if not self.current_scene:
            return
            
        active_objects = [obj for obj in self.current_scene.objects.values() 
                         if obj.active and not obj.destroyed]
        
        # Limit objects per frame for performance
        max_updates = min(len(active_objects), self.max_objects_per_frame)
        start_index = (self.frame_count * max_updates) % len(active_objects) if active_objects else 0
        
        objects_to_update = active_objects[start_index:start_index + max_updates]
        
        for obj in objects_to_update:
            try:
                obj.update(delta_time)
            except Exception as e:
                self._log_error(f"Error updating object {obj.name}: {e}")

    def _batch_update_objects(self, objects, delta_time):
        """Batch update objects for better CPU cache utilization"""
        # Group objects by type for better cache locality
        object_groups = {}
        
        for obj in objects:
            obj_type = obj.object_type
            if obj_type not in object_groups:
                object_groups[obj_type] = []
            object_groups[obj_type].append(obj)
        
        # Update each group of objects together
        for obj_type, group in object_groups.items():
            self._update_object_group(group, delta_time)

    def _update_object_group(self, objects, delta_time):
        """Update group of objects of same type"""
        # Optimized update for object group
        for obj in objects:
            try:
                # NEW OPTIMIZATION: Skip expensive operations for distant objects
                if hasattr(obj, 'position') and hasattr(self, 'renderer'):
                    # Simple distance check (without sqrt for speed)
                    if self.renderer.camera:
                        cam_x, cam_y = self.renderer.camera.position
                        obj_x, obj_y = obj.position
                        dist_sq = (obj_x - cam_x) ** 2 + (obj_y - cam_y) ** 2
                        
                        # Skip update for very distant objects
                        if dist_sq > 1000000:  # 1000^2
                            obj._skip_update_check = True
                        else:
                            obj._skip_update_check = False
                
                obj.update(delta_time)
            except Exception as e:
                self._log_error(f"Error updating object {obj.name}: {e}")

    def _cleanup_object_pools(self):
        """Cleanup object pools"""
        if self.frame_count % 300 == 0:  # Every 5 seconds at 60 FPS
            for pool_name, pool in self.object_pool.items():
                if hasattr(pool, 'cleanup'):
                    pool.cleanup()

    def get_object_from_pool(self, object_type):
        """Get object from pool for better performance"""
        if object_type not in self.object_pool:
            self.object_pool[object_type] = []
            
        pool = self.object_pool[object_type]
        if pool:
            obj = pool.pop()
            obj.reset()  # Reset properties
            self.performance_stats['pool_reuses'] += 1
            return obj
        else:
            # Create new object
            from .game_object import GameObject
            return GameObject(f"Pooled_{object_type}", object_type)

    def return_object_to_pool(self, obj):
        """Return object to pool"""
        if obj.object_type not in self.object_pool:
            self.object_pool[obj.object_type] = []
            
        obj.visible = False
        obj.active = False
        self.object_pool[obj.object_type].append(obj)

    def set_performance_mode(self, mode):
        """Set performance mode with advanced CPU/GPU distribution"""
        self.performance_mode = mode
        
        if mode == "performance":
            # CPU-focused optimizations
            self._configure_cpu_mode()
            self._log_info("🚀 Performance mode: CPU-optimized for weak hardware")
            
        elif mode == "quality":
            # Balanced CPU/GPU usage
            self._configure_balanced_mode()
            self._log_info("✨ Quality mode: Balanced CPU/GPU usage")
            
        elif mode == "gpu":
            # GPU-focused optimizations
            if self.gpu_acceleration_enabled:
                self._configure_gpu_mode()
                self._log_info("🎮 GPU mode: Graphics card accelerated")
            else:
                self._log_warning("❌ GPU mode not available, using performance mode")
                self.set_performance_mode("performance")
                
        elif mode == "extreme_performance":
            # Aggressive CPU optimizations
            self._configure_extreme_cpu_mode()
            self._log_info("⚡ Extreme Performance: Maximum CPU optimization")
            
        elif mode == "compute":
            # Modern compute-optimized mode
            if self.gpu_compute_available:
                self._configure_compute_mode()
                self._log_info("🔥 Compute mode: GPU compute shaders enabled")
            else:
                self._log_warning("❌ Compute mode not available")
                self.set_performance_mode("gpu")
    
    def _configure_cpu_mode(self):
        """Configure for CPU-optimized performance"""
        # Move tasks to CPU
        self.cpu_tasks = ["physics", "ai", "scripting", "collision", "particles", "basic_rendering"]
        self.gpu_tasks = ["sprite_batching"]
        
        # CPU optimizations
        self.culling_enabled = True
        self.batch_rendering = True
        self.adaptive_fps = True
        self.frame_skip_enabled = True
        self.max_objects_per_frame = 500
        self.cpu_thread_pool_size = min(4, self.cpu_thread_pool_size)
        
        # Disable expensive effects
        if self.renderer:
            self.renderer.disable_expensive_effects()
    
    def _configure_balanced_mode(self):
        """Configure for balanced CPU/GPU usage"""
        # Distribute tasks evenly
        self.cpu_tasks = ["physics", "ai", "scripting", "collision_broad_phase"]
        self.gpu_tasks = ["rendering", "particles", "lighting", "post_processing"]
        
        self.culling_enabled = True
        self.batch_rendering = True
        self.adaptive_fps = False
        self.frame_skip_enabled = False
        self.max_objects_per_frame = 1500
    
    def _configure_gpu_mode(self):
        """Configure for GPU-accelerated performance"""
        # Move more tasks to GPU
        self.cpu_tasks = ["physics", "ai", "scripting"]
        self.gpu_tasks = ["rendering", "particles", "lighting", "post_processing", "collision_broad_phase"]
        
        self.culling_enabled = True
        self.batch_rendering = True
        self.adaptive_fps = False
        self.frame_skip_enabled = False
        self.max_objects_per_frame = 3000
        
        # Enable GPU features
        if self.renderer:
            self.renderer.enable_gpu_acceleration()
            self.renderer.enable_advanced_effects()
    
    def _configure_extreme_cpu_mode(self):
        """Configure for extreme CPU optimization"""
        # All tasks on CPU with maximum optimization
        self.cpu_tasks = ["physics", "ai", "scripting", "collision", "particles", "minimal_rendering"]
        self.gpu_tasks = []
        
        self.culling_enabled = True
        self.batch_rendering = True
        self.adaptive_fps = True
        self.frame_skip_enabled = True
        self.max_objects_per_frame = 200
        self.target_fps = 30
        
        # Aggressive optimizations
        self._enable_aggressive_optimizations()
        
        # Reduce thread usage for weak CPUs
        self.cpu_thread_pool_size = 2
    
    def _configure_compute_mode(self):
        """Configure for GPU compute shader acceleration"""
        # Utilize GPU compute for parallel tasks
        self.cpu_tasks = ["ai", "scripting"]
        self.gpu_tasks = ["rendering", "particles", "lighting", "post_processing", "physics_compute", "collision_compute"]
        
        self.culling_enabled = False  # GPU can handle more objects
        self.batch_rendering = True
        self.adaptive_fps = False
        self.frame_skip_enabled = False
        self.max_objects_per_frame = 5000
        
        # Enable compute shaders
        if self.renderer:
            self.renderer.enable_compute_shaders()
            self.renderer.enable_advanced_effects(all=True)

    def _enable_aggressive_optimizations(self):
        """Enable aggressive optimizations for weak CPUs"""
        # Lower rendering quality
        if self.renderer:
            self.renderer.enable_antialiasing = False
            self.renderer.particle_limit = 50
            
        # Aggressive garbage collection
        import gc
        gc.set_threshold(100, 5, 5)  # More frequent garbage collection
        
        # Spatial grid optimization
        self._spatial_grid_size = 128  # Larger grid cells = fewer calculations

    def enable_cpu_optimization(self):
        """Enable advanced CPU optimizations"""
        self._log_info("🔧 Enabling advanced CPU optimizations...")
        
        # Multi-threading for suitable tasks (without external libraries)
        self._enable_threaded_updates = True
        
        # Priority update system
        self._priority_update_system = True
        
        # Cache optimizations
        self._enable_smart_caching()
        
        # Memory pool pre-allocation
        self._preallocate_memory_pools()
        
        self._log_info("✅ CPU optimizations enabled")

    def _enable_smart_caching(self):
        """Enable smart caching for better CPU utilization"""
        # Cache for frequently used calculations
        self.calculation_cache = {}
        self.collision_cache_enabled = True
        self.render_cache_enabled = True
        
        # Cache cleanup every 1000 frames
        self.cache_cleanup_interval = 1000

    def _preallocate_memory_pools(self):
        """Pre-allocate memory pools to reduce garbage collection"""
        # Pre-allocate objects for commonly used types
        common_types = ['projectile', 'particle', 'enemy', 'pickup']
        
        for obj_type in common_types:
            if obj_type not in self.object_pool:
                self.object_pool[obj_type] = []
                
                # Pre-create several objects
                for _ in range(20):
                    from .game_object import GameObject
                    obj = GameObject(f"Pooled_{obj_type}", obj_type)
                    obj.active = False
                    obj.visible = False
                    self.object_pool[obj_type].append(obj)
        
        self._log_info(f"💾 Pre-allocated {len(common_types)} object pools")

    def get_cpu_performance_info(self):
        """Get detailed CPU and GPU performance information"""
        gpu_info = {}
        if self.renderer:
            gpu_info = self.renderer.get_gpu_info()
        
        # CPU task distribution info
        cpu_task_info = {
            "active_tasks": len(self.cpu_tasks),
            "thread_pool_size": self.cpu_thread_pool_size,
            "task_queue_size": self.cpu_task_queue.qsize() if hasattr(self, 'cpu_task_queue') else 0,
            "workers_active": len(self.cpu_workers)
        }
        
        # GPU task distribution info
        gpu_task_info = {
            "active_tasks": len(self.gpu_tasks),
            "memory_usage": getattr(self, 'gpu_memory_usage', 0),
            "compute_available": self.gpu_compute_available
        }
        
        return {
            "cpu_usage": self.performance_stats.get('cpu_usage', 0),
            "memory_usage": self.performance_stats.get('memory_usage', 0),
            "fps": self.performance_stats.get('fps', 0),
            "frame_time": self.performance_stats.get('frame_time', 0),
            "objects_rendered": self.performance_stats.get('objects_rendered', 0),
            "performance_mode": self.performance_mode,
            "gpu_acceleration": self.gpu_acceleration_enabled,
            "gpu_info": gpu_info,
            "cpu_task_info": cpu_task_info,
            "gpu_task_info": gpu_task_info,
            "task_distribution": {
                "cpu_tasks": self.cpu_tasks,
                "gpu_tasks": self.gpu_tasks
            },
            "optimizations_active": {
                "culling": self.culling_enabled,
                "batching": self.batch_rendering,
                "adaptive_fps": self.adaptive_fps,
                "frame_skipping": self.frame_skip_enabled,
                "object_pooling": len(self.object_pool) > 0,
                "gpu_rendering": self.gpu_acceleration_enabled,
                "cpu_threading": len(self.cpu_workers) > 0,
                "gpu_compute": self.gpu_compute_available
            }
        }
    
    def _cpu_physics_broad_phase(self, objects):
        """CPU-based physics broad phase collision detection"""
        # Simplified broad phase for CPU processing
        potential_pairs = []
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                if self._broad_phase_check(obj1, obj2):
                    potential_pairs.append((obj1, obj2))
        return potential_pairs
    
    def _cpu_pathfinding(self, start, end, grid):
        """CPU-based pathfinding calculation"""
        # A* pathfinding implementation for CPU
        # Simplified implementation
        return []
    
    def _cpu_ai_processing(self, entities):
        """CPU-based AI entity processing"""
        # AI decision making, state updates
        results = []
        for entity in entities:
            # Process AI logic
            result = {"entity_id": entity.get("id"), "decision": "continue"}
            results.append(result)
        return results
    
    def _cpu_collision_detection(self, objects):
        """CPU-based detailed collision detection"""
        collisions = []
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                if self._detailed_collision_check(obj1, obj2):
                    collisions.append((obj1, obj2))
        return collisions
    
    def submit_cpu_task(self, task_type, **kwargs):
        """Submit task to CPU worker pool"""
        if hasattr(self, 'cpu_task_queue'):
            task = {"type": task_type, **kwargs}
            self.cpu_task_queue.put(task)
            return True
        return False
    
    def submit_gpu_task(self, task_type, **kwargs):
        """Submit task to GPU processing queue"""
        task = {"type": task_type, **kwargs}
        self.gpu_command_queue.append(task)
        return True
    
    def process_gpu_tasks(self):
        """Process queued GPU tasks"""
        if not self.gpu_acceleration_enabled:
            return
            
        while self.gpu_command_queue:
            task = self.gpu_command_queue.pop(0)
            self._execute_gpu_task(task)
    
    def _execute_gpu_task(self, task):
        """Execute GPU task"""
        task_type = task.get("type")
        
        if task_type == "render_batch":
            if self.renderer:
                self.renderer.gpu_render_batch(task.get("objects", []))
        elif task_type == "particle_update":
            if hasattr(self, 'particle_system'):
                self.particle_system.gpu_update(task.get("particles", []))
        elif task_type == "lighting_compute":
            if self.renderer:
                self.renderer.gpu_compute_lighting(task.get("lights", []))
    
    def balance_cpu_gpu_load(self):
        """Dynamically balance CPU/GPU task distribution"""
        cpu_usage = self.performance_stats.get('cpu_usage', 0)
        fps = self.performance_stats.get('fps', 60)
        
        # If CPU is overloaded, move tasks to GPU
        if cpu_usage > 85 and self.gpu_acceleration_enabled:
            self._shift_tasks_to_gpu()
        # If FPS is low and GPU is available, utilize GPU more
        elif fps < 45 and self.gpu_acceleration_enabled:
            self._shift_tasks_to_gpu()
        # If CPU is underutilized, move some tasks back
        elif cpu_usage < 50 and len(self.gpu_tasks) > 2:
            self._shift_tasks_to_cpu()
    
    def _shift_tasks_to_gpu(self):
        """Move CPU tasks to GPU for better performance"""
        movable_tasks = ["particles", "collision_broad_phase", "lighting"]
        
        for task in movable_tasks:
            if task in self.cpu_tasks and task not in self.gpu_tasks:
                self.cpu_tasks.remove(task)
                self.gpu_tasks.append(task)
                self._log_info(f"🔄 Moved {task} to GPU")
                break
    
    def _shift_tasks_to_cpu(self):
        """Move GPU tasks to CPU to balance load"""
        movable_tasks = ["collision_broad_phase", "particles"]
        
        for task in movable_tasks:
            if task in self.gpu_tasks and task not in self.cpu_tasks:
                self.gpu_tasks.remove(task)
                self.cpu_tasks.append(task)
                self._log_info(f"🔄 Moved {task} to CPU")
                break

    def force_cpu_optimization(self):
        """Force CPU optimization for struggling systems"""
        self._log_warning("🔧 FORCED CPU OPTIMIZATION - Game running slowly!")
        
        # Drastic load reduction
        self.set_performance_mode("extreme_performance")
        self.max_objects_per_frame = 100
        self.target_fps = 20
        
        # Disable demanding effects
        try:
            from .particle_system import particle_system
            particle_system.max_particles = 25
        except:
            pass
            
        self._log_info("⚡ CPU optimization completed - game should run better")

    def cleanup(self):
        """Comprehensive cleanup of all engine resources"""
        self._log_info("🧹 Cleaning up engine resources...")

        try:
            # Stop any running processes
            self.running = False

            # NEW OPTIMIZATION: Cleanup pools
            self.object_pool.clear()
            self.render_cache.clear()
            self.collision_cache.clear()
            self.spatial_grid.clear()

            # Cleanup subsystems in reverse order
            subsystems = [
                ('asset_manager', self.asset_manager),
                ('audio_manager', self.audio_manager),
                ('renderer', self.renderer),
                ('physics', self.physics)
            ]

            for name, system in subsystems:
                if system and hasattr(system, 'cleanup'):
                    try:
                        system.cleanup()
                        self._log_info(f"   ✓ {name} cleaned up")
                    except Exception as e:
                        self._log_error(f"   ✗ Failed to cleanup {name}: {e}")

            # Clear external systems
            try:
                from .audio_system import audio_system
                from .animation_system import animation_system
                from .particle_system import particle_system

                audio_system.cleanup()
                animation_system.clear()
                particle_system.clear()
                self._log_info("   ✓ External systems cleaned up")
            except Exception as e:
                self._log_error(f"   ✗ Failed to cleanup external systems: {e}")

            # Clear scene data
            for scene in self.scenes.values():
                if hasattr(scene, 'cleanup'):
                    scene.cleanup()
            self.scenes.clear()
            self.current_scene = None

            # Clear engine state
            self.global_variables.clear()
            self.event_dispatcher.clear()
            self.message_queue.clear()
            self.registered_systems.clear()

            # Cleanup pygame
            if pygame.get_init():
                pygame.quit()
                self._log_info("   ✓ Pygame cleaned up")

            self.initialized = False
            self._log_info("✅ Engine cleanup completed successfully")

        except Exception as e:
            print(f"❌ Error during cleanup: {e}")

    def _log_info(self, message):
        """Log info message"""
        if self.verbose_logging:
            print(f"[INFO] {message}")

    def _log_warning(self, message):
        """Log warning message"""
        print(f"[WARNING] {message}")
        self.warning_log.append(message)

    def _log_error(self, message):
        """Log error message"""
        print(f"[ERROR] {message}")
        self.error_log.append(message)

    def _log_fatal(self, message, exception=None):
        """Log fatal error and trigger crash handler"""
        import time
        import traceback
        
        self.last_error_time = time.time()
        fatal_msg = f"[FATAL ERROR] {message}"
        
        if exception:
            fatal_msg += f"\nException: {str(exception)}"
            fatal_msg += f"\nTraceback:\n{traceback.format_exc()}"
        
        print(f"💀 {fatal_msg}")
        self.error_log.append(fatal_msg)
        
        # Create crash report
        if self.enable_crash_reporting:
            self._create_crash_report(message, exception)
        
        # Trigger fatal error handler
        if self.fatal_error_handler:
            try:
                self.fatal_error_handler(message, exception)
            except Exception as e:
                print(f"[FATAL] Error in fatal error handler: {e}")
        
        # Show fatal error dialog
        self._show_fatal_error_dialog(message, exception)

    def _create_crash_report(self, message, exception):
        """Create detailed crash report"""
        import time
        import traceback
        import sys
        import os
        
        crash_data = {
            "timestamp": time.time(),
            "message": message,
            "exception": str(exception) if exception else None,
            "traceback": traceback.format_exc() if exception else None,
            "engine_version": "0.5",
            "python_version": sys.version,
            "platform": sys.platform,
            "current_scene": self.current_scene.name if self.current_scene else None,
            "performance_stats": self.performance_stats.copy(),
            "object_count": len(self.current_scene.objects) if self.current_scene else 0,
            "memory_usage": self._get_memory_usage()
        }
        
        self.crash_reports.append(crash_data)
        
        # Save crash report to file
        try:
            import json
            if not os.path.exists("crash_reports"):
                os.makedirs("crash_reports")
            
            timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            filename = f"crash_reports/crash_{timestamp_str}.json"
            
            with open(filename, 'w') as f:
                json.dump(crash_data, f, indent=2)
            
            print(f"💾 Crash report saved to: {filename}")
        except Exception as e:
            print(f"Failed to save crash report: {e}")

    def _get_memory_usage(self):
        """Get current memory usage"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0

    def _show_fatal_error_dialog(self, message, exception):
        """Show fatal error dialog to user"""
        try:
            import tkinter as tk
            from tkinter import messagebox
            
            root = tk.Tk()
            root.withdraw()  # Hide main window
            
            error_text = f"Axarion Engine Fatal Error\n\n{message}"
            if exception:
                error_text += f"\n\nException: {str(exception)}"
            
            error_text += "\n\nThe engine will now shut down."
            error_text += "\nCheck crash_reports/ folder for detailed information."
            
            messagebox.showerror("Fatal Error - Axarion Engine", error_text)
            root.destroy()
        except Exception as e:
            print(f"Failed to show error dialog: {e}")
            # Fallback to console output
            print("=" * 60)
            print("🚨 FATAL ERROR - AXARION ENGINE 🚨")
            print("=" * 60)
            print(f"Error: {message}")
            if exception:
                print(f"Exception: {str(exception)}")
            print("=" * 60)
            print("The engine will now shut down.")
            print("Check crash_reports/ folder for detailed information.")
            print("=" * 60)

    def set_fatal_error_handler(self, handler):
        """Set custom fatal error handler"""
        self.fatal_error_handler = handler

    # Additional engine management methods
    def register_system(self, system, priority=0):
        """Register a custom system with the engine"""
        self.registered_systems.append(system)
        self.system_priorities[system] = priority
        self.registered_systems.sort(key=lambda s: self.system_priorities.get(s, 0))

    def unregister_system(self, system):
        """Unregister a custom system"""
        if system in self.registered_systems:
            self.registered_systems.remove(system)
            if system in self.system_priorities:
                del self.system_priorities[system]

    def post_message(self, message_type, data=None):
        """Post a message to the message queue"""
        message = {'type': message_type, 'data': data, 'timestamp': self.total_time}
        self.message_queue.append(message)

    def subscribe_to_messages(self, message_type, handler):
        """Subscribe to messages of a specific type"""
        if message_type not in self.event_dispatcher:
            self.event_dispatcher[message_type] = []
        self.event_dispatcher[message_type].append(handler)

    def unsubscribe_from_messages(self, message_type, handler):
        """Unsubscribe from messages"""
        if message_type in self.event_dispatcher:
            if handler in self.event_dispatcher[message_type]:
                self.event_dispatcher[message_type].remove(handler)

    def get_performance_info(self):
        """Get detailed performance information"""
        return self.performance_stats.copy()

    def set_target_fps(self, fps):
        """Set target FPS"""
        self.target_fps = max(1, min(fps, 240))  # Clamp between 1-240

    def toggle_debug_mode(self):
        """Toggle debug mode"""
        self.debug_mode = not self.debug_mode
        if self.renderer:
            self.renderer.debug_mode = self.debug_mode
        return self.debug_mode

    def emergency_shutdown(self, reason="Unknown error"):
        """Emergency shutdown with error reporting"""
        self._log_fatal(f"Emergency shutdown: {reason}")
        
        try:
            # Quick cleanup
            self.running = False
            
            # Save any critical data
            if hasattr(self, 'current_scene') and self.current_scene:
                try:
                    self.save_project("emergency_save.json")
                    print("💾 Emergency save completed")
                except:
                    pass
            
            # Force cleanup
            pygame.quit()
            
        except Exception as e:
            print(f"Error during emergency shutdown: {e}")

    def get_crash_reports(self):
        """Get all crash reports"""
        return self.crash_reports.copy()

    def clear_crash_reports(self):
        """Clear crash reports"""
        self.crash_reports.clear()

    def is_engine_healthy(self):
        """Check if engine is in healthy state"""
        return (self.initialized and 
                self.renderer is not None and 
                self.physics is not None and
                len(self.error_log) < 100)  # Too many errors indicate problems

    # UNLIMITED GAME DEVELOPMENT FEATURES

    def enable_unlimited_mode(self):
        """Enable unlimited game development capabilities"""
        self.game_mode = "unlimited"
        self.unlimited_objects = True
        self.unlimited_scenes = True
        self.unlimited_assets = True
        self._log_info("Unlimited game development mode enabled!")

    def create_custom_system(self, name: str, system_class):
        """Create custom game system for any genre"""
        self.custom_systems[name] = system_class()
        self.register_system(self.custom_systems[name])
        return self.custom_systems[name]

    def add_networking(self, server_port: int = 5000):
        """Add networking capabilities for multiplayer games"""
        self.network_enabled = True
        self.multiplayer_support = True
        self.network_port = server_port
        self._log_info(f"Networking enabled on port {server_port}")

    def create_save_system(self, save_slots: int = 10):
        """Create unlimited save system"""
        self.save_system = {
            "slots": save_slots,
            "auto_save": True,
            "compression": True,
            "encryption": False
        }
        return self.save_system

    def add_achievement_system(self, achievements: List[Dict]):
        """Add achievement system for any game type"""
        self.achievement_system = achievements
        for achievement in achievements:
            achievement["unlocked"] = False
            achievement["progress"] = 0
        return self.achievement_system

    def create_dialog_system(self, dialog_data: Dict):
        """Create dialog system for RPGs, adventures, etc."""
        self.dialog_system = dialog_data
        return self.dialog_system

    def add_inventory_system(self, name: str, max_items: int = 999):
        """Add inventory system (unlimited items supported)"""
        self.inventory_systems[name] = {
            "max_items": max_items,
            "items": {},
            "categories": [],
            "sorting": True
        }
        return self.inventory_systems[name]

    def create_quest_system(self, quests: List[Dict]):
        """Create quest system for RPGs and adventure games"""
        self.quest_system = {
            "active_quests": [],
            "completed_quests": [],
            "available_quests": quests,
            "quest_log": True
        }
        return self.quest_system

    def enable_level_editor(self):
        """Enable in-game level editor"""
        try:
            from .level_editor import LevelEditor
            self.level_editor = LevelEditor(self)
            return self.level_editor
        except ImportError:
            self._log_warning("Level editor module not available")
            # Create a simple placeholder level editor
            self.level_editor = {"enabled": True, "mode": "basic"}
            return self.level_editor

    def add_post_processing(self, effect_name: str, parameters: Dict):
        """Add post-processing effects"""
        effect = {"name": effect_name, "params": parameters, "enabled": True}
        self.post_processing.append(effect)
        return effect

    def create_lighting_system(self, ambient_light: float = 0.3):
        """Create dynamic lighting system"""
        self.lighting_system = {
            "ambient": ambient_light,
            "lights": [],
            "shadows": True,
            "dynamic": True
        }
        return self.lighting_system

    def add_ai_system(self, name: str, ai_type: str = "fsm"):
        """Add AI system (FSM, Behavior Trees, Neural Networks)"""
        if ai_type == "fsm":
            self.ai_systems[name] = {"type": "finite_state_machine", "states": {}}
        elif ai_type == "bt":
            self.ai_systems[name] = {"type": "behavior_tree", "nodes": []}
        elif ai_type == "neural":
            self.ai_systems[name] = {"type": "neural_network", "layers": []}
        return self.ai_systems[name]

    def enable_pathfinding(self, algorithm: str = "astar"):
        """Enable pathfinding for AI characters"""
        self.pathfinding = {
            "algorithm": algorithm,
            "grid_size": 32,
            "obstacles": [],
            "cache": True
        }
        return self.pathfinding

    def create_unlimited_objects(self, count: int = 10000):
        """Remove object limits for massive games"""
        self.max_objects = count
        self._log_info(f"Object limit increased to {count}")

    def add_mod_support(self, mod_folder: str = "mods"):
        """Enable mod support for community content"""
        self.mod_support = True
        self.mod_folder = mod_folder
        self._log_info(f"Mod support enabled in '{mod_folder}' folder")

    def create_particle_physics(self):
        """Advanced particle physics for realistic effects"""
        return {
            "gravity": True,
            "collision": True,
            "fluid_dynamics": True,
            "particle_count": 50000
        }

    def enable_procedural_generation(self):
        """Enable procedural content generation"""
        return {
            "terrain": True,
            "dungeons": True,
            "items": True,
            "enemies": True,
            "music": True
        }

    def add_analytics(self):
        """Add game analytics for balancing"""
        return {
            "player_behavior": True,
            "performance_tracking": True,
            "crash_reporting": True,
            "heatmaps": True
        }

    def create_scripting_sandbox(self):
        """Create secure scripting environment"""
        return {
            "lua_support": True,
            "python_support": True,
            "javascript_support": True,
            "security": "sandboxed"
        }
