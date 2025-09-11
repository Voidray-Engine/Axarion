
"""
Mod System for Axarion Engine
Loads and manages user-created modifications and content
"""

import os
import json
import importlib.util
import sys
from typing import Dict, List, Any, Optional
from pathlib import Path

class ModSystem:
    """Advanced mod system for loading user-created content"""
    
    def __init__(self):
        self.mods_directory = "mods/"
        self.loaded_mods = {}
        self.mod_metadata = {}
        self.mod_assets = {}
        self.mod_scripts = {}
        self.mod_configs = {}
        
        # Mod loading order and dependencies
        self.load_order = []
        self.mod_dependencies = {}
        
        # Security settings
        self.allow_script_execution = True
        self.sandbox_mode = False
        
        # Create mods directory
        os.makedirs(self.mods_directory, exist_ok=True)
        
        # Create example mod structure
        self._create_example_mod_structure()
    
    def _create_example_mod_structure(self):
        """Create example mod structure for users"""
        example_mod_path = os.path.join(self.mods_directory, "example_mod")
        
        if not os.path.exists(example_mod_path):
            os.makedirs(example_mod_path, exist_ok=True)
            os.makedirs(os.path.join(example_mod_path, "assets", "images"), exist_ok=True)
            os.makedirs(os.path.join(example_mod_path, "assets", "sounds"), exist_ok=True)
            os.makedirs(os.path.join(example_mod_path, "scripts"), exist_ok=True)
            
            # Create example mod.json
            example_config = {
                "name": "Example Mod",
                "version": "1.0.0",
                "author": "Your Name",
                "description": "Example mod showing structure",
                "engine_version": "0.5",
                "dependencies": [],
                "assets": {},
                "scripts": ["main.py"],
                "enabled": False
            }
            
            with open(os.path.join(example_mod_path, "mod.json"), 'w') as f:
                json.dump(example_config, f, indent=2)

            
            with open(os.path.join(example_mod_path, "scripts", "main.py"), 'w') as f:
                f.write()
            
            with open(os.path.join(example_mod_path, "assets",), 'w') as f:
                json.dump(f, indent=2)
    
    def scan_mods(self) -> List[str]:
        """Scan mods directory for available mods"""
        available_mods = []
        
        if not os.path.exists(self.mods_directory):
            return available_mods
        
        for item in os.listdir(self.mods_directory):
            mod_path = os.path.join(self.mods_directory, item)
            
            if os.path.isdir(mod_path):
                config_file = os.path.join(mod_path, "mod.json")
                if os.path.exists(config_file):
                    available_mods.append(item)
        
        return available_mods
    
    def load_mod_config(self, mod_name: str) -> Optional[Dict]:
        """Load mod configuration"""
        config_path = os.path.join(self.mods_directory, mod_name, "mod.json")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Failed to load mod config for {mod_name}: {e}")
            return None
    
    def load_mod(self, mod_name: str) -> bool:
        """Load a specific mod"""
        if mod_name in self.loaded_mods:
            print(f"Mod {mod_name} already loaded")
            return True
        
        try:
            # Load mod configuration
            config = self.load_mod_config(mod_name)
            if not config:
                return False
            
            # Check if mod is enabled
            if not config.get("enabled", True):
                print(f"Mod {mod_name} is disabled")
                return False
            
            mod_path = os.path.join(self.mods_directory, mod_name)
            
            # Load mod assets
            self._load_mod_assets(mod_name, mod_path, config)
            
            # Load mod scripts
            if self.allow_script_execution:
                self._load_mod_scripts(mod_name, mod_path, config)
            
            # Store mod data
            self.loaded_mods[mod_name] = {
                "path": mod_path,
                "config": config,
                "loaded": True
            }
            
            self.mod_metadata[mod_name] = config
            
            print(f"✅ Mod loaded: {config.get('name', mod_name)} v{config.get('version', '1.0')}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load mod {mod_name}: {e}")
            return False
    
    def _load_mod_assets(self, mod_name: str, mod_path: str, config: Dict):
        """Load mod assets"""
        assets_path = os.path.join(mod_path, "assets")
        
        if not os.path.exists(assets_path):
            return
        
        self.mod_assets[mod_name] = {}
        assets_config = config.get("assets", {})
        
        # Load images
        images_path = os.path.join(assets_path, "images")
        if os.path.exists(images_path):
            self.mod_assets[mod_name]["images"] = {}
            
            for image_file in assets_config.get("images", []):
                image_path = os.path.join(images_path, image_file)
                if os.path.exists(image_path):
                    asset_name = f"{mod_name}_{os.path.splitext(image_file)[0]}"
                    
                    # Load through asset manager
                    from .asset_manager import asset_manager
                    if asset_manager.load_image(asset_name, image_path):
                        self.mod_assets[mod_name]["images"][image_file] = asset_name
        
        # Load sounds
        sounds_path = os.path.join(assets_path, "sounds")
        if os.path.exists(sounds_path):
            self.mod_assets[mod_name]["sounds"] = {}
            
            for sound_file in assets_config.get("sounds", []):
                sound_path = os.path.join(sounds_path, sound_file)
                if os.path.exists(sound_path):
                    asset_name = f"{mod_name}_{os.path.splitext(sound_file)[0]}"
                    
                    # Load through asset manager
                    from .asset_manager import asset_manager
                    if asset_manager.load_sound(asset_name, sound_path):
                        self.mod_assets[mod_name]["sounds"][sound_file] = asset_name
        
        # Load data files
        self.mod_assets[mod_name]["data"] = {}
        for data_file in assets_config.get("data", []):
            data_path = os.path.join(assets_path, data_file)
            if os.path.exists(data_path):
                try:
                    with open(data_path, 'r', encoding='utf-8') as f:
                        if data_file.endswith('.json'):
                            data = json.load(f)
                        else:
                            data = f.read()
                    
                    self.mod_assets[mod_name]["data"][data_file] = data
                except Exception as e:
                    print(f"Failed to load data file {data_file}: {e}")
    
    def _load_mod_scripts(self, mod_name: str, mod_path: str, config: Dict):
        """Load mod scripts"""
        scripts_path = os.path.join(mod_path, "scripts")
        
        if not os.path.exists(scripts_path):
            return
        
        self.mod_scripts[mod_name] = {}
        
        for script_file in config.get("scripts", []):
            script_path = os.path.join(scripts_path, script_file)
            
            if os.path.exists(script_path):
                try:
                    # Load Python module
                    module_name = f"mod_{mod_name}_{os.path.splitext(script_file)[0]}"
                    spec = importlib.util.spec_from_file_location(module_name, script_path)
                    module = importlib.util.module_from_spec(spec)
                    
                    # Add to sys.modules to make it importable
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    
                    self.mod_scripts[mod_name][script_file] = module
                    
                except Exception as e:
                    print(f"Failed to load script {script_file}: {e}")
    
    def load_all_mods(self) -> int:
        """Load all available mods"""
        available_mods = self.scan_mods()
        loaded_count = 0
        
        # Load mods in dependency order
        for mod_name in available_mods:
            if self.load_mod(mod_name):
                loaded_count += 1
        
        print(f"📦 Loaded {loaded_count}/{len(available_mods)} mods")
        return loaded_count
    
    def unload_mod(self, mod_name: str) -> bool:
        """Unload a specific mod"""
        if mod_name not in self.loaded_mods:
            return False
        
        try:
            # Call cleanup functions in scripts
            if mod_name in self.mod_scripts:
                for script_name, module in self.mod_scripts[mod_name].items():
                    if hasattr(module, 'cleanup'):
                        try:
                            module.cleanup(None)  # Pass engine reference if needed
                        except Exception as e:
                            print(f"Error in cleanup for {script_name}: {e}")
            
            # Remove mod assets from asset manager
            if mod_name in self.mod_assets:
                from .asset_manager import asset_manager
                
                for asset_type, assets in self.mod_assets[mod_name].items():
                    if asset_type in ["images", "sounds"]:
                        for asset_name in assets.values():
                            asset_manager.unload_asset(asset_name)
            
            # Remove from loaded mods
            del self.loaded_mods[mod_name]
            
            if mod_name in self.mod_assets:
                del self.mod_assets[mod_name]
            
            if mod_name in self.mod_scripts:
                del self.mod_scripts[mod_name]
            
            if mod_name in self.mod_metadata:
                del self.mod_metadata[mod_name]
            
            print(f"🗑️ Mod unloaded: {mod_name}")
            return True
            
        except Exception as e:
            print(f"Failed to unload mod {mod_name}: {e}")
            return False
    
    def initialize_mods(self, engine):
        """Initialize all loaded mods"""
        for mod_name, scripts in self.mod_scripts.items():
            for script_name, module in scripts.items():
                if hasattr(module, 'initialize'):
                    try:
                        module.initialize(engine)
                    except Exception as e:
                        print(f"Error initializing {mod_name}/{script_name}: {e}")
    
    def update_mods(self, engine, delta_time: float):
        """Update all loaded mods"""
        for mod_name, scripts in self.mod_scripts.items():
            for script_name, module in scripts.items():
                if hasattr(module, 'update'):
                    try:
                        module.update(engine, delta_time)
                    except Exception as e:
                        print(f"Error updating {mod_name}/{script_name}: {e}")
    
    def call_mod_function(self, function_name: str, *args, **kwargs):
        """Call a function in all loaded mods"""
        results = {}
        
        for mod_name, scripts in self.mod_scripts.items():
            for script_name, module in scripts.items():
                if hasattr(module, function_name):
                    try:
                        result = getattr(module, function_name)(*args, **kwargs)
                        results[f"{mod_name}/{script_name}"] = result
                    except Exception as e:
                        print(f"Error calling {function_name} in {mod_name}/{script_name}: {e}")
        
        return results
    
    def get_mod_asset(self, mod_name: str, asset_type: str, asset_name: str):
        """Get mod asset"""
        if mod_name not in self.mod_assets:
            return None
        
        assets = self.mod_assets[mod_name].get(asset_type, {})
        return assets.get(asset_name)
    
    def get_mod_data(self, mod_name: str, data_file: str):
        """Get mod data"""
        if mod_name not in self.mod_assets:
            return None
        
        data_assets = self.mod_assets[mod_name].get("data", {})
        return data_assets.get(data_file)
    
    def list_loaded_mods(self) -> List[Dict]:
        """List all loaded mods with metadata"""
        loaded_list = []
        
        for mod_name, mod_data in self.loaded_mods.items():
            config = mod_data["config"]
            loaded_list.append({
                "name": mod_name,
                "display_name": config.get("name", mod_name),
                "version": config.get("version", "1.0"),
                "author": config.get("author", "Unknown"),
                "description": config.get("description", "No description"),
                "enabled": config.get("enabled", True)
            })
        
        return loaded_list
    
    def enable_mod(self, mod_name: str) -> bool:
        """Enable a mod"""
        config_path = os.path.join(self.mods_directory, mod_name, "mod.json")
        
        try:
            config = self.load_mod_config(mod_name)
            if config:
                config["enabled"] = True
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                return True
        except Exception as e:
            print(f"Failed to enable mod {mod_name}: {e}")
        
        return False
    
    def disable_mod(self, mod_name: str) -> bool:
        """Disable a mod"""
        config_path = os.path.join(self.mods_directory, mod_name, "mod.json")
        
        try:
            # Unload if currently loaded
            if mod_name in self.loaded_mods:
                self.unload_mod(mod_name)
            
            config = self.load_mod_config(mod_name)
            if config:
                config["enabled"] = False
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                return True
        except Exception as e:
            print(f"Failed to disable mod {mod_name}: {e}")
        
        return False
    
    def reload_mod(self, mod_name: str) -> bool:
        """Reload a specific mod"""
        if mod_name in self.loaded_mods:
            if self.unload_mod(mod_name):
                return self.load_mod(mod_name)
        else:
            return self.load_mod(mod_name)
        
        return False
    
    def get_mod_info(self, mod_name: str) -> Optional[Dict]:
        """Get detailed mod information"""
        config = self.load_mod_config(mod_name)
        if not config:
            return None
        
        mod_path = os.path.join(self.mods_directory, mod_name)
        
        info = {
            "name": mod_name,
            "config": config,
            "loaded": mod_name in self.loaded_mods,
            "path": mod_path,
            "size": self._get_directory_size(mod_path),
            "assets_count": 0,
            "scripts_count": len(config.get("scripts", []))
        }
        
        # Count assets
        assets_config = config.get("assets", {})
        for asset_type, asset_list in assets_config.items():
            info["assets_count"] += len(asset_list)
        
        return info
    
    def _get_directory_size(self, directory: str) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
        except Exception:
            pass
        
        return total_size
    
    def cleanup(self):
        """Clean up mod system"""
        # Unload all mods
        mod_names = list(self.loaded_mods.keys())
        for mod_name in mod_names:
            self.unload_mod(mod_name)
        
        # Clear all data
        self.loaded_mods.clear()
        self.mod_metadata.clear()
        self.mod_assets.clear()
        self.mod_scripts.clear()
        self.mod_configs.clear()

# Global mod system instance
mod_system = ModSystem()
