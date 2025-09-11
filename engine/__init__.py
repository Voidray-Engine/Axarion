__version__ = "1.0.2"
__author__ = "Raynix Studio"

# Core engine imports
from .core import AxarionEngine
from .game_object import GameObject
from .scene import Scene
from .renderer import Renderer
from .physics import PhysicsSystem

# System imports
try:
    from .input_system import input_system
    from .audio_system import audio_system  
    from .animation_system import animation_system
    from .particle_system import particle_system
    from .asset_manager import asset_manager
except ImportError as e:
    print(f"Warning: Some engine systems may not be available: {e}")

# Camera and advanced features
try:
    from .camera import Camera
    from .tilemap import Tilemap
    from .state_machine import StateMachine
except ImportError:
    pass

# Export main classes for easy import
__all__ = [
    'AxarionEngine',
    'GameObject', 
    'Scene',
    'Renderer',
    'PhysicsSystem',
    'Camera',
    'Tilemap',
    'StateMachine'
]

# Convenience functions for quick game setup
def create_engine(width=800, height=600, title="Axarion Game", **kwargs):
    """
    Quick engine creation with automatic initialization
    
    Args:
        width: Screen width
        height: Screen height  
        title: Window title
        **kwargs: Additional engine configuration
        
    Returns:
        Initialized AxarionEngine instance
    """
    engine = AxarionEngine(width, height, title)
    
    # Apply any additional configuration
    for key, value in kwargs.items():
        if hasattr(engine, key):
            setattr(engine, key, value)
    
    # Initialize engine
    if not engine.initialize():
        print("Warning: Engine initialization had issues, but continuing...")
    
    return engine

def quick_game_setup(width=800, height=600, title="My Game"):
    """
    Ultra-quick game setup - creates engine and default scene
    
    Returns:
        tuple: (engine, scene) ready for immediate use
    """
    engine = create_engine(width, height, title)
    scene = engine.create_scene("MainScene")
    engine.current_scene = scene
    
    return engine, scene

# Version info
def get_version():
    """Get engine version"""
    return __version__

def get_info():
    """Get engine info"""
    return {
        "name": "Axarion Engine",
        "version": __version__,
        "author": __author__,
        "description": "Modern 2D game engine for Python",
        "standalone": True
    }

# Initialize message
print(f"Axarion Engine v{__version__} - Standalone Mode")
print("Ready for game development!")

