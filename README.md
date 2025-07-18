# 🎮 Axarion Engine
**Professional 2D Game Engine for Code-First Development**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-GPL--3.0-red)
![Version](https://img.shields.io/badge/version-0.8.9-green)
![Status](https://img.shields.io/badge/status-Beta--stable-brightgreen)

*Proudly developed by a Czech indie studio*

---

## 🌟 Overview

Axarion Engine is a cutting-edge 2D game development framework designed specifically for programmers who prefer code-first workflows over visual editors. Born from years of experience in game development and refined through countless iterations, Axarion represents the evolution of modern indie game creation tools.

**🇨🇿 Made in Czech Republic** - Developed with passion by a small Czech development studio committed to empowering indie developers worldwide.

## ✨ Why Axarion Engine?

### 🚀 **Performance-First Design**
- **Lightning-Fast Rendering**: Optimized graphics pipeline with intelligent batching
- **Memory Efficient**: Advanced asset management reduces memory footprint by 50%
- **Smooth Physics**: Stable collision detection with realistic simulations
- **Frame-Perfect Animation**: Advanced interpolation for buttery-smooth motion

### 💡 **Developer-Centric Features**
- **Pure Code Approach**: No complex GUI editors - just clean, readable code
- **Rapid Prototyping**: From idea to playable demo in minutes
- **Professional Debugging**: Visual collision bounds, performance profiling, error tracking
- **Cross-Platform**: Windows, macOS, and Linux support out of the box

### 🎵 **Complete Multimedia Support**
- **Advanced Audio System**: Full sound effects, background music, and spatial audio
- **Particle Effects**: Built-in systems for explosions, trails, weather, and custom effects
- **Animation Pipeline**: Frame-by-frame sprite animations with blend modes
- **Asset Management**: Automatic loading and optimization of images, sounds, and fonts

### 🏗️ **Professional Architecture**
- **Scene Management**: Organize games into levels, menus, and states
- **Component System**: Modular, reusable game object components
- **Event System**: Decoupled communication between game systems
- **State Machines**: Easy AI and game state management

## 🎯 Perfect For

- **Indie Game Developers** seeking full creative control
- **Programming Students** learning game development fundamentals
- **Game Jams** and rapid prototyping competitions
- **Educational Projects** teaching game programming concepts
- **Professional Studios** requiring lightweight, customizable tools

## 🚀 Quick Start

### Your First Game (5 Minutes)
```python
from engine.core import AxarionEngine
from engine.game_object import GameObject
import pygame

# Initialize pygame first
pygame.init()

# Create engine
engine = AxarionEngine(800, 600, "My First Game")

# Initialize with error handling
try:
    if not engine.initialize():
        print("⚠️ Engine didn't initialize correctly, but continuing...")
except Exception as e:
    print(f"⚠️ Initialization error: {e}")

# Create scene
scene = engine.create_scene("GameScene")
engine.current_scene = scene

# Create player object
player = GameObject("Player", "rectangle")
player.position = (100, 100)
player.set_property("width", 40)
player.set_property("height", 40)
player.set_property("color", (100, 200, 255))
player.is_static = True

# Add to scene
scene.add_object(player)

# Game loop
clock = pygame.time.Clock()
speed = 200

while engine.running:
    delta_time = clock.tick(60) / 1000.0
    
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            engine.stop()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                engine.stop()
    
    # Handle input
    keys = pygame.key.get_pressed()
    x, y = player.position
    
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        player.position = (x, y - speed * delta_time)
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        player.position = (x, y + speed * delta_time)
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        player.position = (x - speed * delta_time, y)
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        player.position = (x + speed * delta_time, y)
    
    # Update and render
    if engine.current_scene:
        engine.current_scene.update(delta_time)
    
    if engine.renderer:
        engine.renderer.clear()
        if engine.current_scene:
            engine.current_scene.render(engine.renderer)
        engine.renderer.present()

engine.cleanup()
```

## 🎮 Core Systems

### Game Object Types
| Type | Description | Use Case |
|------|-------------|----------|
| `rectangle` | Rectangular collision shapes | Platforms, walls, UI elements |
| `circle` | Circular collision shapes | Balls, coins, projectiles |
| `sprite` | Image-based objects | Characters, items, decorations |
| `animated_sprite` | Frame-animated objects | Walking characters, spinning coins |

### Core Engine Features
```python
# Movement & Physics
player.move(dx, dy)                    # Relative movement
player.position = (x, y)               # Absolute positioning
player.apply_force(fx, fy)             # Physics-based forces
player.velocity = (vx, vy)             # Direct velocity control

# Input Handling
engine.input.is_key_pressed("w")       # Check if key is held
engine.input.is_key_just_pressed(" ")  # Check for single key press
engine.input.is_mouse_clicked(0)       # Left mouse button
engine.input.get_mouse_position()      # Current mouse coordinates

# Audio & Effects
engine.audio.play_sound("explosion")   # Play sound effect
engine.audio.play_music("background")  # Background music
engine.particles.create_explosion(x, y) # Particle explosion
engine.particles.create_trail(x, y, color) # Particle trail

# Game Logic
scene.find_objects_by_tag("enemy")     # Find objects by tag
player.is_colliding_with(enemy)        # Collision detection
engine.math.distance(x1, y1, x2, y2)   # Calculate distance
scene.create_object("enemy", x, y)     # Spawn new objects
```

## 📁 Project Structure

```
your-game/
├── assets/                     # Game assets
│   ├── images/                 # Sprites and textures
│   ├── sounds/                 # Audio files
│   ├── animations/             # Frame animations
│   └── fonts/                  # Typography
├── engine/                     # Core engine files
├── utils/                      # Utility modules
├── DOCS/                       # Documentation
├── main.py                     # Your game entry point
└── README.md                   # Project documentation
```

## 🛠️ Advanced Features

### Performance Optimization
- **Sprite Batching**: Automatic rendering optimization
- **Culling System**: Only render visible objects
- **Asset Streaming**: Dynamic loading for large games
- **Memory Pooling**: Efficient object recycling

### Development Tools
- **Live Reloading**: See changes instantly during development
- **Performance Profiler**: Identify bottlenecks and optimize
- **Visual Debugger**: See collision bounds and object states
- **Error Recovery**: Graceful handling of script errors

### Extensibility
- **Plugin System**: Extend engine functionality
- **Custom Components**: Create reusable game systems
- **Mod Support**: Allow players to modify your games
- **Export Pipeline**: Deploy to multiple platforms

## 📚 Learning Resources

### Documentation
- 📖 **[Getting Started Guide](DOCS/GETTING_STARTED.md)** - Complete beginner tutorial
- ⚡ **[Quick Reference](DOCS/QUICK_REFERENCE.md)** - Function reference and examples
- 🎮 **[Complete Game Tutorial](DOCS/TUTORIAL_COMPLETE_GAME.md)** - Build a full game step-by-step
- 🎨 **[Asset Creation Guide](DOCS/ASSET_CREATION.md)** - Creating game art and audio

### Example Projects
- **Space Shooter** - Classic arcade-style game
- **Platformer** - Jump and run mechanics
- **Puzzle Game** - Logic-based gameplay
- **RPG Demo** - Turn-based combat system

## 🌍 Community & Support

### Czech Studio Background
Axarion Engine was born from the passion of Czech indie developers who understand the challenges of creating games with limited resources. Our studio focuses on:

- **Accessible Tools**: Making game development approachable for everyone
- **Performance First**: Ensuring smooth gameplay on all hardware
- **Developer Happiness**: Tools that inspire creativity, not frustration
- **Open Source Spirit**: Sharing knowledge and fostering community growth

### Getting Help
- **GitHub Issues**: Report bugs and request features
- **Community Discord**: Connect with other developers
- **Documentation**: Comprehensive guides and tutorials
- **Email Support**: Direct contact with the development team

## 📜 License & Attribution

Axarion Engine is released under the GPL License. While attribution is not required, we appreciate recognition of our work. See our [Attribution Guide](DOCS/Axarion_Attribution_Guide.md) for optional ways to credit the engine.

## 🚀 Get Started Today

Ready to create your next masterpiece? Join thousands of developers who've chosen Axarion Engine for their creative projects. From weekend game jams to commercial releases, Axarion provides the power and flexibility you need.

**Axarion Engine - Where Code Meets Creativity** 🎮✨

---

*Made with ❤️ in Czech Republic | Empowering Developers Worldwide*
