
"""
Axarion Engine Particle System
Creates visual effects like explosions, fire, smoke, etc.
"""

import pygame
import math
import random
from typing import List, Tuple, Optional
from .game_object import GameObject

class Particle:
    """Individual particle"""
    
    def __init__(self, x: float, y: float, velocity: Tuple[float, float], 
                 lifetime: float, color: Tuple[int, int, int], size: float = 2.0):
        self.position = [x, y]
        self.velocity = list(velocity)
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.color = list(color)
        self.size = size
        self.alpha = 255
        
    def update(self, delta_time: float, gravity: float = 0.0):
        """Update particle"""
        # Update position
        self.position[0] += self.velocity[0] * delta_time
        self.position[1] += self.velocity[1] * delta_time
        
        # Apply gravity
        self.velocity[1] += gravity * delta_time
        
        # Update lifetime
        self.lifetime -= delta_time
        
        # Fade out
        life_ratio = self.lifetime / self.max_lifetime
        self.alpha = int(255 * life_ratio)
        
        return self.lifetime > 0

class ParticleEmitter:
    """Particle emitter that spawns particles"""
    
    def __init__(self, x: float, y: float):
        self.position = [x, y]
        self.particles: List[Particle] = []
        
        # Emission properties
        self.emission_rate = 50  # particles per second
        self.emission_timer = 0.0
        self.active = True
        
        # Particle properties
        self.lifetime_range = (1.0, 3.0)
        self.velocity_range = ((-100, -100), (100, 100))
        self.color_range = ((255, 0, 0), (255, 255, 0))  # Red to yellow
        self.size_range = (1.0, 4.0)
        self.gravity = 100.0
        
    def update(self, delta_time: float):
        """Update emitter and all particles"""
        # Emit new particles
        if self.active:
            self.emission_timer += delta_time
            particles_to_emit = int(self.emission_timer * self.emission_rate)
            self.emission_timer -= particles_to_emit / self.emission_rate
            
            for _ in range(particles_to_emit):
                self.emit_particle()
        
        # Update existing particles
        active_particles = []
        for particle in self.particles:
            if particle.update(delta_time, self.gravity):
                active_particles.append(particle)
        
        self.particles = active_particles
    
    def emit_particle(self):
        """Emit a single particle"""
        # Random properties within ranges
        lifetime = random.uniform(*self.lifetime_range)
        
        vel_x = random.uniform(self.velocity_range[0][0], self.velocity_range[1][0])
        vel_y = random.uniform(self.velocity_range[0][1], self.velocity_range[1][1])
        
        # Interpolate color
        t = random.random()
        color = [
            int(self.color_range[0][i] * (1-t) + self.color_range[1][i] * t)
            for i in range(3)
        ]
        
        size = random.uniform(*self.size_range)
        
        particle = Particle(
            self.position[0], self.position[1],
            (vel_x, vel_y), lifetime, color, size
        )
        
        self.particles.append(particle)
    
    def render(self, renderer):
        """Render all particles"""
        for particle in self.particles:
            color = (*particle.color, particle.alpha)
            renderer.draw_circle(
                particle.position[0], particle.position[1],
                particle.size, particle.color[:3]
            )

class ParticleSystem:
    """Manages all particle emitters"""
    
    def __init__(self):
        self.emitters: List[ParticleEmitter] = []
        
    def create_explosion(self, x: float, y: float, intensity: int = 50) -> ParticleEmitter:
        """Create explosion effect"""
        emitter = ParticleEmitter(x, y)
        emitter.emission_rate = intensity * 10
        emitter.lifetime_range = (0.5, 2.0)
        emitter.velocity_range = ((-200, -200), (200, 200))
        emitter.color_range = ((255, 100, 0), (255, 255, 0))
        emitter.gravity = 50.0
        emitter.active = False  # Burst only
        
        # Emit burst
        for _ in range(intensity):
            emitter.emit_particle()
            
        self.emitters.append(emitter)
        return emitter
    
    def create_fire(self, x: float, y: float) -> ParticleEmitter:
        """Create fire effect"""
        emitter = ParticleEmitter(x, y)
        emitter.emission_rate = 30
        emitter.lifetime_range = (1.0, 2.5)
        emitter.velocity_range = ((-20, -100), (20, -50))
        emitter.color_range = ((255, 0, 0), (255, 200, 0))
        emitter.gravity = -20.0  # Upward
        
        self.emitters.append(emitter)
        return emitter
    
    def create_smoke(self, x: float, y: float) -> ParticleEmitter:
        """Create smoke effect"""
        emitter = ParticleEmitter(x, y)
        emitter.emission_rate = 20
        emitter.lifetime_range = (2.0, 4.0)
        emitter.velocity_range = ((-30, -80), (30, -40))
        emitter.color_range = ((100, 100, 100), (200, 200, 200))
        emitter.gravity = -10.0
        
        self.emitters.append(emitter)
        return emitter
    
    def update(self, delta_time: float):
        """Update all emitters"""
        active_emitters = []
        for emitter in self.emitters:
            emitter.update(delta_time)
            # Keep emitter if it's active or has particles
            if emitter.active or emitter.particles:
                active_emitters.append(emitter)
        
        self.emitters = active_emitters
    
    def render(self, renderer):
        """Render all emitters"""
        for emitter in self.emitters:
            emitter.render(renderer)
    
    def clear(self):
        """Clear all emitters"""
        self.emitters.clear()
        
    def create_menu_particle(self, x: float, y: float) -> ParticleEmitter:
        """Create floating menu particles - Axarion style"""
        emitter = ParticleEmitter(x, y)
        emitter.emission_rate = 2
        emitter.lifetime_range = (3.0, 5.0)
        emitter.velocity_range = ((-20, -80), (20, -40))
        emitter.color_range = ((255, 165, 0), (255, 200, 100))  # Axarion orange
        emitter.gravity = -5.0  # Slow upward drift
        emitter.active = False  # Single burst
        
        # Emit just one particle
        emitter.emit_particle()
        
        self.emitters.append(emitter)
        return emitter
    
    def create_dust_particle(self, x: float, y: float) -> ParticleEmitter:
        """Create floating dust particles for menu background"""
        emitter = ParticleEmitter(x, y)
        emitter.emission_rate = 1
        emitter.lifetime_range = (8.0, 15.0)  # Long floating
        emitter.velocity_range = ((-15, -30), (15, 30))  # Slow drift
        emitter.color_range = ((120, 120, 140), (160, 160, 180))  # Dusty grey-blue
        emitter.gravity = -2.0  # Very slow upward drift
        emitter.size_range = (0.5, 2.0)  # Small particles
        emitter.active = False
        
        # Single dust particle
        emitter.emit_particle()
        
        self.emitters.append(emitter)
        return emitter
    
    def create_atmospheric_particle(self, x: float, y: float) -> ParticleEmitter:
        """Create atmospheric particles for game"""
        emitter = ParticleEmitter(x, y)
        emitter.emission_rate = 1
        emitter.lifetime_range = (6.0, 12.0)
        emitter.velocity_range = ((-10, -20), (10, 20))
        emitter.color_range = ((60, 60, 80), (100, 100, 120))  # Dark atmospheric
        emitter.gravity = -1.0
        emitter.size_range = (0.3, 1.5)
        emitter.active = False
        
        emitter.emit_particle()
        
        self.emitters.append(emitter)
        return emitter
    
    def create_smoke_cloud(self, x: float, y: float) -> ParticleEmitter:
        """Create smoke cloud to hide software limits"""
        emitter = ParticleEmitter(x, y)
        emitter.emission_rate = 8
        emitter.lifetime_range = (10.0, 20.0)  # Long lasting smoke
        emitter.velocity_range = ((-5, -15), (5, 15))  # Slow spread
        emitter.color_range = ((70, 70, 80), (90, 90, 100))  # Dark grey smoke
        emitter.gravity = -0.5  # Very slow rise
        emitter.size_range = (3.0, 8.0)  # Large smoke particles
        emitter.active = True  # Continuous emission
        
        # Emit initial burst
        for _ in range(15):
            emitter.emit_particle()
        
        self.emitters.append(emitter)
        return emitter

# Global particle system
particle_system = ParticleSystem()
