"""
Axarion Studio 
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
import os
import json
import subprocess
import threading
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

# Import Axarion Engine components for error checking
try:
    from engine.core import AxarionEngine
    from utils.file_manager import FileManager
except ImportError as e:
    print(f"Warning: Some Axarion components not available: {e}")

# Import PyInstaller for game packaging
try:
    import PyInstaller
    PYINSTALLER_AVAILABLE = True
except ImportError:
    PYINSTALLER_AVAILABLE = False

class SyntaxHighlighter:
    """Advanced syntax highlighter for Python"""

    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.setup_tags()

    def setup_tags(self):
        """Configure syntax highlighting tags"""
        # Python keywords
        self.text_widget.tag_configure("python_keyword", foreground="#FF7F50")
        self.text_widget.tag_configure("python_string", foreground="#98FB98")
        self.text_widget.tag_configure("python_comment", foreground="#87CEEB")
        self.text_widget.tag_configure("python_function", foreground="#FFD700")
        self.text_widget.tag_configure("python_number", foreground="#DDA0DD")

        # AXScript support removed

        # Error highlighting
        self.text_widget.tag_configure("error", background="#FF4444", foreground="white")
        self.text_widget.tag_configure("warning", background="#FFAA00", foreground="black")

    def highlight_python(self, content: str):
        """Highlight Python syntax"""
        python_keywords = [
            "def", "class", "if", "else", "elif", "for", "while", "try", "except",
            "finally", "with", "as", "import", "from", "return", "yield", "break",
            "continue", "pass", "lambda", "and", "or", "not", "in", "is", "True",
            "False", "None", "global", "nonlocal", "async", "await"
        ]

        lines = content.split('\n')
        for line_num, line in enumerate(lines):
            # Keywords
            for keyword in python_keywords:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                for match in re.finditer(pattern, line):
                    start = f"{line_num + 1}.{match.start()}"
                    end = f"{line_num + 1}.{match.end()}"
                    self.text_widget.tag_add("python_keyword", start, end)

            # Strings
            string_patterns = [r'"[^"]*"', r"'[^']*'", r'""".*?"""', r"'''.*?'''"]
            for pattern in string_patterns:
                for match in re.finditer(pattern, line):
                    start = f"{line_num + 1}.{match.start()}"
                    end = f"{line_num + 1}.{match.end()}"
                    self.text_widget.tag_add("python_string", start, end)

            # Comments
            comment_match = re.search(r'#.*$', line)
            if comment_match:
                start = f"{line_num + 1}.{comment_match.start()}"
                end = f"{line_num + 1}.{comment_match.end()}"
                self.text_widget.tag_add("python_comment", start, end)

            # Functions
            func_match = re.search(r'def\s+(\w+)', line)
            if func_match:
                start = f"{line_num + 1}.{func_match.start(1)}"
                end = f"{line_num + 1}.{func_match.end(1)}"
                self.text_widget.tag_add("python_function", start, end)

            # Numbers
            for match in re.finditer(r'\b\d+\.?\d*\b', line):
                start = f"{line_num + 1}.{match.start()}"
                end = f"{line_num + 1}.{match.end()}"
                self.text_widget.tag_add("python_number", start, end)

    def highlight_axscript(self, content: str):
        """AXScript support has been removed"""
        pass

class ErrorChecker:
    """Real-time error checking for Python and AXScript"""

    def __init__(self):
        # AXScript support removed
        pass

    def check_python_syntax(self, code: str) -> List[Dict]:
        """Check Python syntax for errors"""
        errors = []
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            errors.append({
                'type': 'error',
                'line': e.lineno or 1,
                'message': f"Syntax Error: {e.msg}",
                'column': e.offset or 0
            })
        except Exception as e:
            errors.append({
                'type': 'error',
                'line': 1,
                'message': f"Error: {str(e)}",
                'column': 0
            })

        # Check for common game development issues
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            # Check for undefined Axarion components
            if 'GameObject' in line and 'from engine.game_object import GameObject' not in code:
                errors.append({
                    'type': 'warning',
                    'line': i,
                    'message': "GameObject used but not imported",
                    'column': line.find('GameObject')
                })

            if 'AxarionEngine' in line and 'from engine.core import AxarionEngine' not in code:
                errors.append({
                    'type': 'warning',
                    'line': i,
                    'message': "AxarionEngine used but not imported",
                    'column': line.find('AxarionEngine')
                })

        return errors

    def check_axscript_syntax(self, code: str) -> List[Dict]:
        """AXScript support has been removed"""
        return []

class FileExplorer:
    """File explorer panel for project management"""

    def __init__(self, parent, editor_callback):
        self.parent = parent
        self.editor_callback = editor_callback
        self.current_project = None

        self.frame = ttk.Frame(parent)
        self.setup_ui()

    def setup_ui(self):
        """Setup file explorer UI"""
        # Toolbar
        toolbar = ttk.Frame(self.frame)
        toolbar.pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(toolbar, text="📁 Open Project", command=self.open_project).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📄 New File", command=self.new_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🗂️ New Folder", command=self.new_folder).pack(side=tk.LEFT, padx=2)

        # Tree view
        self.tree = ttk.Treeview(self.frame)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Scrollbar
        scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Bind events
        self.tree.bind('<Double-1>', self.on_file_double_click)
        self.tree.bind('<Button-3>', self.on_right_click)

    def open_project(self):
        """Open project directory"""
        directory = filedialog.askdirectory(title="Select Project Directory")
        if directory:
            self.current_project = directory
            self.refresh_tree()

    def refresh_tree(self):
        """Refresh file tree"""
        if not self.current_project:
            return

        # Clear tree
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Add files and folders
        self.add_directory_to_tree("", self.current_project)

    def add_directory_to_tree(self, parent, path):
        """Add directory contents to tree"""
        try:
            for item in sorted(os.listdir(path)):
                if item.startswith('.'):
                    continue

                item_path = os.path.join(path, item)

                if os.path.isdir(item_path):
                    # Folder
                    folder_id = self.tree.insert(parent, 'end', text=f"📁 {item}", 
                                               values=[item_path], open=False)
                    self.add_directory_to_tree(folder_id, item_path)
                else:
                    # File
                    icon = self.get_file_icon(item)
                    self.tree.insert(parent, 'end', text=f"{icon} {item}", 
                                   values=[item_path])
        except PermissionError:
            pass

    def get_file_icon(self, filename):
        """Get icon for file type"""
        ext = os.path.splitext(filename)[1].lower()
        icons = {
            '.py': '🐍',
            '.json': '📋',
            '.png': '🖼️',
            '.jpg': '🖼️',
            '.wav': '🔊',
            '.mp3': '🎵',
            '.txt': '📄',
            '.md': '📖'
        }
        return icons.get(ext, '📄')

    def on_file_double_click(self, event):
        """Handle file double click"""
        selection = self.tree.selection()
        if selection:
            item = selection[0]
            file_path = self.tree.item(item, 'values')[0]

            if os.path.isfile(file_path):
                self.editor_callback(file_path)

    def new_file(self):
        """Create new file"""
        if not self.current_project:
            messagebox.showwarning("Warning", "Please open a project first")
            return

        filename = simpledialog.askstring("New File", "Enter filename:")
        if filename:
            file_path = os.path.join(self.current_project, filename)
            try:
                with open(file_path, 'w') as f:
                    f.write("")
                self.refresh_tree()
                self.editor_callback(file_path)
            except Exception as e:
                messagebox.showerror("Error", f"Could not create file: {e}")

    def new_folder(self):
        """Create new folder"""
        if not self.current_project:
            messagebox.showwarning("Warning", "Please open a project first")
            return

        foldername = simpledialog.askstring("New Folder", "Enter folder name:")
        if foldername:
            folder_path = os.path.join(self.current_project, foldername)
            try:
                os.makedirs(folder_path, exist_ok=True)
                self.refresh_tree()
            except Exception as e:
                messagebox.showerror("Error", f"Could not create folder: {e}")

    def on_right_click(self, event):
        """Handle right click context menu"""
        # TODO: Implement context menu
        pass

class ErrorPanel:
    """Error and warning display panel"""

    def __init__(self, parent):
        self.parent = parent
        self.frame = ttk.Frame(parent)
        self.setup_ui()
        self.errors = []

    def setup_ui(self):
        """Setup error panel UI"""
        # Header
        header = ttk.Frame(self.frame)
        header.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(header, text="🚨 Problems", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)

        self.error_count = ttk.Label(header, text="0 errors, 0 warnings")
        self.error_count.pack(side=tk.RIGHT)

        # Error list
        self.error_tree = ttk.Treeview(self.frame, columns=('type', 'line', 'message'), show='headings', height=6)
        self.error_tree.heading('type', text='Type')
        self.error_tree.heading('line', text='Line')
        self.error_tree.heading('message', text='Message')

        self.error_tree.column('type', width=80)
        self.error_tree.column('line', width=60)
        self.error_tree.column('message', width=400)

        self.error_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def update_errors(self, errors: List[Dict]):
        """Update error display"""
        self.errors = errors

        # Clear existing items
        for item in self.error_tree.get_children():
            self.error_tree.delete(item)

        # Add new errors
        error_count = 0
        warning_count = 0

        for error in errors:
            icon = "🔴" if error['type'] == 'error' else "🟡"
            self.error_tree.insert('', 'end', values=(
                f"{icon} {error['type'].title()}",
                error['line'],
                error['message']
            ))

            if error['type'] == 'error':
                error_count += 1
            else:
                warning_count += 1

        # Update count
        self.error_count.config(text=f"{error_count} errors, {warning_count} warnings")

class GameTemplateManager:
    """Manager for game templates and snippets"""

    def __init__(self):
        self.templates = {
            "Basic Game": '''from engine.core import AxarionEngine
from engine.game_object import GameObject
import pygame

# Create engine
engine = AxarionEngine(800, 600, "My Game")
engine.initialize()

# Create scene
scene = engine.create_scene("Main")
engine.current_scene = scene

# Create player
player = GameObject("Player", "rectangle")
player.position = (100, 100)
player.set_property("color", (100, 200, 255))

# Python-based movement
def update_player():
    keys = pygame.key.get_pressed()
    speed = 200
    if keys[pygame.K_LEFT]:
        player.position = (player.position[0] - speed * 0.016, player.position[1])
    if keys[pygame.K_RIGHT]:
        player.position = (player.position[0] + speed * 0.016, player.position[1])

scene.add_object(player)
engine.run()''',

            "Platformer Game": '''from engine.core import AxarionEngine
from engine.game_object import GameObject
import pygame

# Create platformer game
engine = AxarionEngine(800, 600, "Platformer")
engine.initialize()

scene = engine.create_scene("Level1")
engine.current_scene = scene

# Create player with physics
player = GameObject("Player", "rectangle")
player.position = (100, 400)
player.set_property("color", (100, 200, 255))
player.set_property("mass", 1.0)

# Python-based platformer movement
def update_platformer():
    keys = pygame.key.get_pressed()
    speed = 300
    if keys[pygame.K_LEFT]:
        player.velocity = (player.velocity[0] - speed * 0.016, player.velocity[1])
    if keys[pygame.K_RIGHT]:
        player.velocity = (player.velocity[0] + speed * 0.016, player.velocity[1])
    if keys[pygame.K_SPACE]:
        player.velocity = (player.velocity[0], -400)

# Create platform
platform = GameObject("Platform", "rectangle")
platform.position = (400, 500)
platform.set_property("width", 200)
platform.set_property("height", 20)
platform.set_property("color", (100, 100, 100))

scene.add_object(player)
scene.add_object(platform)
engine.run()'''
        }

    def get_templates(self):
        """Get all available templates"""
        return list(self.templates.keys())

    def get_template_code(self, template_name):
        """Get code for specific template"""
        return self.templates.get(template_name, "")

class AxarionStudio:
    """Main Axarion Studio IDE class"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Axarion Studio")
        self.root.geometry("1400x900")

        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Core components
        self.error_checker = ErrorChecker()
        self.template_manager = GameTemplateManager()
        self.file_manager = FileManager()

        # Current file info
        self.current_file = None
        self.current_file_type = None
        self.unsaved_changes = False

        # Engine configuration settings
        self.engine_settings = {
            "rendering": {
                "resolution_width": 800,
                "resolution_height": 600,
                "target_fps": 60,
                "vsync": True,
                "render_distance": 1000,
                "anti_aliasing": False,
                "texture_filtering": True,
                "max_sprites": 1000,
                "render_scale": 1.0,
                "sprite_batching": True,
                "pixel_perfect": False
            },
            "physics": {
                "gravity": 980.0,
                "physics_iterations": 8,
                "collision_detection": "continuous",
                "enable_physics": True,
                "max_velocity": 500.0,
                "sleep_threshold": 5.0,
                "sub_stepping": True,
                "warm_starting": True
            },
            "audio": {
                "master_volume": 1.0,
                "sfx_volume": 1.0,
                "music_volume": 1.0,
                "audio_quality": "high"
            },
            "debug": {
                "show_fps": False,
                "show_collision_boxes": False,
                "enable_profiler": False,
                "verbose_logging": False,
                "memory_tracking": False,
                "performance_overlay": False,
                "show_physics_debug": False,
                "wireframe_mode": False,
                "log_level": "INFO"
            },
            "development": {
                "hot_reload": False,
                "auto_compile": False,
                "live_editing": False,
                "script_debugging": False,
                "breakpoints": False,
                "step_debugging": False,
                "variable_inspector": False,
                "call_stack": False,
                "watch_expressions": False,
                "code_coverage": False
            },
            "optimization": {
                "object_pooling": False,
                "culling": False,
                "lod_system": False,
                "texture_streaming": False,
                "async_loading": False,
                "memory_optimization": False,
                "garbage_collection": "auto",
                "preload_assets": False,
                "compress_textures": False,
                "texture_atlas": False
            },
            "networking": {
                "enable_networking": False,
                "max_connections": 4,
                "timeout": 10,
                "compression": False,
                "encryption": False,
                "lag_compensation": False,
                "prediction": False,
                "interpolation": False
            }
        }

        # Build settings (integrated Game Packager)
        self.build_settings = {
            "output_path": "",
            "game_name": "MyGame",
            "include_assets": True,
            "include_engine": True,
            "include_scripting": True,
            "include_utils": True,
            "icon_file": "",
            "show_console": False,
            "single_file": True,
            "compress": True,
            "auto_detect_main": True,
            "last_build_path": ""
        }

        # Editor settings
        self.editor_settings = {
            "appearance": {
                "font_size": 11,
                "font_family": "Consolas",
                "theme": "dark",
                "show_line_numbers": True,
                "syntax_highlighting": True
            },
            "behavior": {
                "auto_save": True,
                "auto_indent": True,
                "tab_size": 4,
                "word_wrap": False
            }
        }

        # Setup UI
        self.setup_ui()
        self.setup_menu()

        # Start real-time error checking
        self.start_error_checking()

        # Load settings
        self.load_settings()

    def setup_ui(self):
        """Setup main UI layout"""
        # Main container
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # Left panel (File Explorer)
        left_frame = ttk.Frame(main_paned, width=250)
        main_paned.add(left_frame, weight=0)

        self.file_explorer = FileExplorer(left_frame, self.open_file)
        self.file_explorer.frame.pack(fill=tk.BOTH, expand=True)

        # Right panel (Editor + Problems)
        right_paned = ttk.PanedWindow(main_paned, orient=tk.VERTICAL)
        main_paned.add(right_paned, weight=1)

        # Editor frame
        editor_frame = ttk.Frame(right_paned)
        right_paned.add(editor_frame, weight=1)

        # Editor toolbar
        self.setup_editor_toolbar(editor_frame)

        # Code editor
        self.setup_code_editor(editor_frame)

        # Problems panel
        problems_frame = ttk.Frame(right_paned, height=200)
        right_paned.add(problems_frame, weight=0)

        self.error_panel = ErrorPanel(problems_frame)
        self.error_panel.frame.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.setup_status_bar()

    def setup_editor_toolbar(self, parent):
        """Setup editor toolbar"""
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, padx=5, pady=2)

        # File operations
        ttk.Button(toolbar, text="💾 Save", command=self.save_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🔄 Save As", command=self.save_file_as).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)

        # Templates
        ttk.Label(toolbar, text="Template:").pack(side=tk.LEFT, padx=5)
        self.template_var = tk.StringVar()
        template_combo = ttk.Combobox(toolbar, textvariable=self.template_var, 
                                    values=self.template_manager.get_templates(), 
                                    state="readonly", width=15)
        template_combo.pack(side=tk.LEFT, padx=2)
        template_combo.bind('<<ComboboxSelected>>', self.insert_template)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)

        # Game controls
        ttk.Button(toolbar, text="▶️ Run Game", command=self.run_game).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🛑 Stop Game", command=self.stop_game).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)

        # Build controls
        ttk.Button(toolbar, text="📦 Build EXE", command=self.show_build_dialog).pack(side=tk.LEFT, padx=2)

        # File info
        self.file_info = ttk.Label(toolbar, text="No file open")
        self.file_info.pack(side=tk.RIGHT, padx=5)

    def setup_code_editor(self, parent):
        """Setup code editor with syntax highlighting"""
        editor_container = ttk.Frame(parent)
        editor_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)

        # Line numbers frame
        line_frame = tk.Frame(editor_container, width=50, bg='#f0f0f0')
        line_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.line_numbers = tk.Text(line_frame, width=4, bg='#f0f0f0', 
                                  state=tk.DISABLED, wrap=tk.NONE)
        self.line_numbers.pack(fill=tk.BOTH, expand=True)

        # Code editor
        self.code_editor = scrolledtext.ScrolledText(editor_container, 
                                                   wrap=tk.NONE,
                                                   bg='#1e1e1e',
                                                   fg='#ffffff',
                                                   insertbackground='white',
                                                   font=('Consolas', 11))
        self.code_editor.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Setup syntax highlighter
        self.highlighter = SyntaxHighlighter(self.code_editor)

        # Bind events
        self.code_editor.bind('<KeyRelease>', self.on_code_change)
        self.code_editor.bind('<Button-1>', self.on_code_change)
        self.code_editor.bind('<MouseWheel>', self.sync_scroll)
        self.code_editor.bind('<Configure>', self.update_line_numbers)

        # Auto-completion
        self.code_editor.bind('<KeyPress>', self.on_key_press)

    def setup_status_bar(self):
        """Setup status bar"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_text = ttk.Label(self.status_bar, text="Ready")
        self.status_text.pack(side=tk.LEFT, padx=5)

        # Cursor position
        self.cursor_pos = ttk.Label(self.status_bar, text="Line 1, Col 1")
        self.cursor_pos.pack(side=tk.RIGHT, padx=5)

        # File type
        self.file_type_label = ttk.Label(self.status_bar, text="")
        self.file_type_label.pack(side=tk.RIGHT, padx=5)

    def setup_menu(self):
        """Setup menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New File", command=self.new_file, accelerator="Ctrl+N")
        file_menu.add_command(label="Open File", command=self.open_file_dialog, accelerator="Ctrl+O")
        file_menu.add_command(label="Save", command=self.save_file, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As", command=self.save_file_as, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=lambda: self.code_editor.edit_undo(), accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=lambda: self.code_editor.edit_redo(), accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Cut", command=lambda: self.code_editor.event_generate("<<Cut>>"), accelerator="Ctrl+X")
        edit_menu.add_command(label="Copy", command=lambda: self.code_editor.event_generate("<<Copy>>"), accelerator="Ctrl+C")
        edit_menu.add_command(label="Paste", command=lambda: self.code_editor.event_generate("<<Paste>>"), accelerator="Ctrl+V")
        edit_menu.add_separator()
        edit_menu.add_command(label="Insert Engine Config", command=self.insert_engine_config)

        # Game menu
        game_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Game", menu=game_menu)
        game_menu.add_command(label="Run Game", command=self.run_game, accelerator="F5")
        game_menu.add_command(label="Stop Game", command=self.stop_game, accelerator="Shift+F5")
        game_menu.add_separator()
        game_menu.add_command(label="Create Asset Manager", command=self.create_asset_manager)
        game_menu.add_command(label="Game Templates", command=self.show_templates)

        # Build menu (primary focus)
        build_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Build", menu=build_menu)
        build_menu.add_command(label="📦 Package Game to EXE", command=self.show_build_dialog, accelerator="Ctrl+B")
        build_menu.add_separator()
        build_menu.add_command(label="🏗️ Check Build Dependencies", command=self.check_build_dependencies)
        build_menu.add_command(label="📋 Generate PyInstaller Spec", command=self.generate_build_spec)
        build_menu.add_separator()
        build_menu.add_command(label="📁 Open Build Folder", command=self.open_build_folder)
        build_menu.add_command(label="🧹 Clean Build Files", command=self.clean_build_files)

        # Settings menu (simplified)
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Build Settings", command=self.show_build_settings)
        settings_menu.add_command(label="Editor Settings", command=self.show_editor_settings)

        # Help menu (expanded with docs)
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="📖 Getting Started Guide", command=self.show_getting_started)
        help_menu.add_command(label="⚡ Quick Reference", command=self.show_quick_reference)
        help_menu.add_command(label="🎮 Complete Game Tutorial", command=self.show_complete_tutorial)
        help_menu.add_separator()
        help_menu.add_command(label="🔧 Axarion Engine Guide", command=self.show_engine_docs)
        help_menu.add_separator()
        help_menu.add_command(label="❤️ Attribution Guide", command=self.show_attribution_guide)
        help_menu.add_separator()
        help_menu.add_command(label="ℹ️ About Axarion Studio", command=self.show_about)

        # Keyboard shortcuts
        self.root.bind('<Control-n>', lambda e: self.new_file())
        self.root.bind('<Control-o>', lambda e: self.open_file_dialog())
        self.root.bind('<Control-s>', lambda e: self.save_file())
        self.root.bind('<Control-Shift-s>', lambda e: self.save_file_as())
        self.root.bind('<F5>', lambda e: self.run_game())
        self.root.bind('<Shift-F5>', lambda e: self.stop_game())
        self.root.bind('<Control-b>', lambda e: self.show_build_dialog())
        self.root.bind('<F1>', lambda e: self.show_getting_started())

    def on_code_change(self, event=None):
        """Handle code changes"""
        self.unsaved_changes = True
        self.update_title()
        self.update_line_numbers()
        self.update_cursor_position()

        # Trigger syntax highlighting
        content = self.code_editor.get('1.0', tk.END)

        if self.current_file_type == 'python':
            self.highlighter.highlight_python(content)

    def on_key_press(self, event):
        """Handle key press for auto-completion"""
        if event.keysym == 'Tab':
            # Auto-indent
            cursor_pos = self.code_editor.index(tk.INSERT)
            line_start = cursor_pos.split('.')[0] + '.0'
            line_content = self.code_editor.get(line_start, cursor_pos)

            if line_content.strip():
                return  # Don't auto-indent if there's content

            # Calculate indentation
            prev_line_num = int(cursor_pos.split('.')[0]) - 1
            if prev_line_num > 0:
                prev_line_start = f"{prev_line_num}.0"
                prev_line_end = f"{prev_line_num}.end"
                prev_line = self.code_editor.get(prev_line_start, prev_line_end)

                # Count leading spaces
                indent = len(prev_line) - len(prev_line.lstrip())

                # Add extra indent for certain keywords
                if any(keyword in prev_line for keyword in ['if', 'def', 'class', 'for', 'while', 'function', 'try']):
                    if prev_line.strip().endswith(':') or prev_line.strip().endswith('{'):
                        indent += 4

                self.code_editor.insert(tk.INSERT, ' ' * indent)
            return 'break'

    def update_line_numbers(self, event=None):
        """Update line numbers"""
        content = self.code_editor.get('1.0', tk.END)
        lines = content.split('\n')

        line_numbers_content = '\n'.join(str(i+1) for i in range(len(lines)-1))

        self.line_numbers.config(state=tk.NORMAL)
        self.line_numbers.delete('1.0', tk.END)
        self.line_numbers.insert('1.0', line_numbers_content)
        self.line_numbers.config(state=tk.DISABLED)

    def update_cursor_position(self):
        """Update cursor position in status bar"""
        cursor_pos = self.code_editor.index(tk.INSERT)
        line, col = cursor_pos.split('.')
        self.cursor_pos.config(text=f"Line {line}, Col {int(col)+1}")

    def sync_scroll(self, event):
        """Synchronize scrolling between editor and line numbers"""
        self.line_numbers.yview_moveto(self.code_editor.yview()[0])

    def update_title(self):
        """Update window title"""
        title = "Axarion Studio"
        if self.current_file:
            filename = os.path.basename(self.current_file)
            if self.unsaved_changes:
                filename += " *"
            title += f" - {filename}"
        self.root.title(title)

    def new_file(self):
        """Create new file"""
        if self.unsaved_changes:
            if not self.confirm_unsaved_changes():
                return

        self.code_editor.delete('1.0', tk.END)
        self.current_file = None
        self.current_file_type = None
        self.unsaved_changes = False
        self.update_title()
        self.file_info.config(text="New file")
        self.file_type_label.config(text="")

    def open_file_dialog(self):
        """Open file dialog"""
        file_path = filedialog.askopenfilename(
            title="Open File",
            filetypes=[
                ("Python files", "*.py"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.open_file(file_path)

    def open_file(self, file_path):
        """Open file in editor"""
        if self.unsaved_changes:
            if not self.confirm_unsaved_changes():
                return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            self.code_editor.delete('1.0', tk.END)
            self.code_editor.insert('1.0', content)

            self.current_file = file_path
            self.current_file_type = self.detect_file_type(file_path)
            self.unsaved_changes = False

            self.update_title()
            self.file_info.config(text=os.path.basename(file_path))
            self.file_type_label.config(text=self.current_file_type.upper())

            # Trigger syntax highlighting
            self.on_code_change()

            self.status_text.config(text=f"Opened: {file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Could not open file: {e}")

    def save_file(self):
        """Save current file"""
        if not self.current_file:
            self.save_file_as()
            return

        try:
            content = self.code_editor.get('1.0', tk.END)
            with open(self.current_file, 'w', encoding='utf-8') as f:
                f.write(content)

            self.unsaved_changes = False
            self.update_title()
            self.status_text.config(text=f"Saved: {self.current_file}")

        except Exception as e:
            messagebox.showerror("Error", f"Could not save file: {e}")

    def save_file_as(self):
        """Save file as new name"""
        file_path = filedialog.asksaveasfilename(
            title="Save File As",
            defaultextension=".py",
            filetypes=[
                ("Python files", "*.py"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.current_file = file_path
            self.current_file_type = self.detect_file_type(file_path)
            self.save_file()
            self.file_info.config(text=os.path.basename(file_path))
            self.file_type_label.config(text=self.current_file_type.upper())

    def detect_file_type(self, file_path):
        """Detect file type from extension"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.py':
            return 'python'
        else:
            return 'text'

    def confirm_unsaved_changes(self):
        """Confirm unsaved changes dialog"""
        result = messagebox.askyesnocancel(
            "Unsaved Changes",
            "You have unsaved changes. Do you want to save them?"
        )

        if result is True:  # Yes - save
            self.save_file()
            return True
        elif result is False:  # No - don't save
            return True
        else:  # Cancel
            return False

    def start_error_checking(self):
        """Start real-time error checking thread"""
        def check_errors():
            while True:
                if self.current_file_type == 'python':
                    content = self.code_editor.get('1.0', tk.END)
                    errors = self.error_checker.check_python_syntax(content)

                    # Update UI in main thread
                    self.root.after(0, lambda: self.error_panel.update_errors(errors))

                time.sleep(2)  # Check every 2 seconds

        error_thread = threading.Thread(target=check_errors, daemon=True)
        error_thread.start()

    def insert_template(self, event=None):
        """Insert selected template"""
        template_name = self.template_var.get()
        if template_name:
            template_code = self.template_manager.get_template_code(template_name)
            if template_code:
                # Insert at cursor position
                self.code_editor.insert(tk.INSERT, template_code)
                self.on_code_change()

    def insert_engine_config(self):
        """Insert engine configuration code"""
        config_code = self.get_engine_config_code()
        self.code_editor.insert(tk.INSERT, config_code)
        self.on_code_change()

    def run_game(self):
        """Run the current game"""
        if not self.current_file or self.current_file_type != 'python':
            messagebox.showwarning("Warning", "Please open a Python game file first")
            return

        # Prevent running Axarion Studio itself
        current_script = os.path.abspath(__file__)
        if os.path.samefile(self.current_file, current_script):
            messagebox.showwarning("Warning", "Cannot run Axarion Studio from within itself. Please open a game file first.")
            return

        if self.unsaved_changes:
            self.save_file()

        def run_in_thread():
            try:
                # Change to the file's directory before running
                file_dir = os.path.dirname(os.path.abspath(self.current_file))
                subprocess.run([sys.executable, os.path.basename(self.current_file)], 
                             cwd=file_dir)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Could not run game: {e}"))

        threading.Thread(target=run_in_thread, daemon=True).start()
        self.status_text.config(text="Running game...")

    def stop_game(self):
        """Stop the running game"""
        # TODO: Implement game stopping
        self.status_text.config(text="Stopped game")

    def create_asset_manager(self):
        """Create asset manager code"""
        asset_code = '''# Asset Manager Setup
from assets.create_sample_assets import create_sample_assets
from engine.asset_manager import asset_manager

# Create sample assets
create_sample_assets()

# Load all assets
asset_manager.load_all_assets()

print("Assets loaded successfully!")
'''
        self.code_editor.insert(tk.INSERT, asset_code)
        self.on_code_change()

    def show_templates(self):
        """Show templates dialog"""
        template_window = tk.Toplevel(self.root)
        template_window.title("Game Templates")
        template_window.geometry("600x400")

        # Template list
        listbox = tk.Listbox(template_window)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        for template in self.template_manager.get_templates():
            listbox.insert(tk.END, template)

        def insert_selected():
            selection = listbox.curselection()
            if selection:
                template_name = listbox.get(selection[0])
                template_code = self.template_manager.get_template_code(template_name)
                self.code_editor.insert(tk.INSERT, template_code)
                self.on_code_change()
                template_window.destroy()

        ttk.Button(template_window, text="Insert Template", command=insert_selected).pack(pady=5)

    def show_engine_docs(self):
        """Show engine documentation"""
        docs = """Axarion Engine Guide

Basic Setup:
1. Create AxarionEngine instance
2. Initialize the engine
3. Create scenes and objects
4. Add scripts to objects
5. Run the game

Object Types:
- rectangle: Basic rectangle shape
- circle: Circular shape
- sprite: Image-based object
- animated_sprite: Animated frames

Common Patterns:
- Use scenes for different levels
- Add tags to group objects
- Use scripts for behavior
- Load assets with asset_manager
"""

        messagebox.showinfo("Axarion Engine Guide", docs)

    def show_engine_settings(self):
        """Show engine settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Engine Settings")
        settings_window.geometry("500x600")
        settings_window.resizable(False, False)

        # Create notebook for tabs
        notebook = ttk.Notebook(settings_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Rendering tab
        rendering_frame = ttk.Frame(notebook)
        notebook.add(rendering_frame, text="Rendering")

        # Resolution settings
        ttk.Label(rendering_frame, text="Resolution:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))

        res_frame = ttk.Frame(rendering_frame)
        res_frame.pack(fill=tk.X, pady=5)

        ttk.Label(res_frame, text="Width:").pack(side=tk.LEFT)
        width_var = tk.StringVar(value=str(self.engine_settings["rendering"]["resolution_width"]))
        width_entry = ttk.Entry(res_frame, textvariable=width_var, width=10)
        width_entry.pack(side=tk.LEFT, padx=(5, 10))

        ttk.Label(res_frame, text="Height:").pack(side=tk.LEFT)
        height_var = tk.StringVar(value=str(self.engine_settings["rendering"]["resolution_height"]))
        height_entry = ttk.Entry(res_frame, textvariable=height_var, width=10)
        height_entry.pack(side=tk.LEFT, padx=5)

        # FPS settings
        ttk.Label(rendering_frame, text="Performance:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(15, 5))

        fps_frame = ttk.Frame(rendering_frame)
        fps_frame.pack(fill=tk.X, pady=5)

        ttk.Label(fps_frame, text="Target FPS:").pack(side=tk.LEFT)
        fps_var = tk.StringVar(value=str(self.engine_settings["rendering"]["target_fps"]))
        fps_entry = ttk.Entry(fps_frame, textvariable=fps_var, width=10)
        fps_entry.pack(side=tk.LEFT, padx=5)

        vsync_var = tk.BooleanVar(value=self.engine_settings["rendering"]["vsync"])
        ttk.Checkbutton(fps_frame, text="VSync", variable=vsync_var).pack(side=tk.LEFT, padx=15)

        # Render distance
        ttk.Label(rendering_frame, text="Render Distance:").pack(anchor=tk.W, pady=(15, 5))
        distance_var = tk.StringVar(value=str(self.engine_settings["rendering"]["render_distance"]))
        distance_entry = ttk.Entry(rendering_frame, textvariable=distance_var, width=15)
        distance_entry.pack(anchor=tk.W, pady=5)

        # Quality settings
        ttk.Label(rendering_frame, text="Quality:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(15, 5))

        aa_var = tk.BooleanVar(value=self.engine_settings["rendering"]["anti_aliasing"])
        ttk.Checkbutton(rendering_frame, text="Anti-aliasing", variable=aa_var).pack(anchor=tk.W, pady=2)

        filter_var = tk.BooleanVar(value=self.engine_settings["rendering"]["texture_filtering"])
        ttk.Checkbutton(rendering_frame, text="Texture Filtering", variable=filter_var).pack(anchor=tk.W, pady=2)

        # Physics tab
        physics_frame = ttk.Frame(notebook)
        notebook.add(physics_frame, text="Physics")

        enable_physics_var = tk.BooleanVar(value=self.engine_settings["physics"]["enable_physics"])
        ttk.Checkbutton(physics_frame, text="Enable Physics", variable=enable_physics_var).pack(anchor=tk.W, pady=10)

        ttk.Label(physics_frame, text="Gravity:").pack(anchor=tk.W, pady=(10, 5))
        gravity_var = tk.StringVar(value=str(self.engine_settings["physics"]["gravity"]))
        gravity_entry = ttk.Entry(physics_frame, textvariable=gravity_var, width=15)
        gravity_entry.pack(anchor=tk.W, pady=5)

        ttk.Label(physics_frame, text="Physics Iterations:").pack(anchor=tk.W, pady=(10, 5))
        iterations_var = tk.StringVar(value=str(self.engine_settings["physics"]["physics_iterations"]))
        iterations_entry = ttk.Entry(physics_frame, textvariable=iterations_var, width=15)
        iterations_entry.pack(anchor=tk.W, pady=5)

        ttk.Label(physics_frame, text="Collision Detection:").pack(anchor=tk.W, pady=(10, 5))
        collision_var = tk.StringVar(value=self.engine_settings["physics"]["collision_detection"])
        collision_combo = ttk.Combobox(physics_frame, textvariable=collision_var, 
                                     values=["discrete", "continuous", "continuous_dynamic"], state="readonly")
        collision_combo.pack(anchor=tk.W, pady=5)

        # Audio tab
        audio_frame = ttk.Frame(notebook)
        notebook.add(audio_frame, text="Audio")

        ttk.Label(audio_frame, text="Volume Settings:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=10)

        # Master volume
        ttk.Label(audio_frame, text="Master Volume:").pack(anchor=tk.W, pady=(5, 2))
        master_vol_var = tk.DoubleVar(value=self.engine_settings["audio"]["master_volume"])
        master_vol_scale = ttk.Scale(audio_frame, from_=0.0, to=1.0, variable=master_vol_var, orient=tk.HORIZONTAL)
        master_vol_scale.pack(fill=tk.X, pady=2)

        # SFX volume
        ttk.Label(audio_frame, text="SFX Volume:").pack(anchor=tk.W, pady=(10, 2))
        sfx_vol_var = tk.DoubleVar(value=self.engine_settings["audio"]["sfx_volume"])
        sfx_vol_scale = ttk.Scale(audio_frame, from_=0.0, to=1.0, variable=sfx_vol_var, orient=tk.HORIZONTAL)
        sfx_vol_scale.pack(fill=tk.X, pady=2)

        # Music volume
        ttk.Label(audio_frame, text="Music Volume:").pack(anchor=tk.W, pady=(10, 2))
        music_vol_var = tk.DoubleVar(value=self.engine_settings["audio"]["music_volume"])
        music_vol_scale = ttk.Scale(audio_frame, from_=0.0, to=1.0, variable=music_vol_var, orient=tk.HORIZONTAL)
        music_vol_scale.pack(fill=tk.X, pady=2)

        # Audio quality
        ttk.Label(audio_frame, text="Audio Quality:").pack(anchor=tk.W, pady=(15, 5))
        quality_var = tk.StringVar(value=self.engine_settings["audio"]["audio_quality"])
        quality_combo = ttk.Combobox(audio_frame, textvariable=quality_var, 
                                   values=["low", "medium", "high"], state="readonly")
        quality_combo.pack(anchor=tk.W, pady=5)

        # Debug tab
        debug_frame = ttk.Frame(notebook)
        notebook.add(debug_frame, text="Debug")

        ttk.Label(debug_frame, text="Debug Options:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=10)

        show_fps_var = tk.BooleanVar(value=self.engine_settings["debug"]["show_fps"])
        ttk.Checkbutton(debug_frame, text="Show FPS Counter", variable=show_fps_var).pack(anchor=tk.W, pady=5)

        show_collision_var = tk.BooleanVar(value=self.engine_settings["debug"]["show_collision_boxes"])
        ttk.Checkbutton(debug_frame, text="Show Collision Boxes", variable=show_collision_var).pack(anchor=tk.W, pady=5)

        enable_profiler_var = tk.BooleanVar(value=self.engine_settings["debug"]["enable_profiler"])
        ttk.Checkbutton(debug_frame, text="Enable Profiler", variable=enable_profiler_var).pack(anchor=tk.W, pady=5)

        verbose_log_var = tk.BooleanVar(value=self.engine_settings["debug"]["verbose_logging"])
        ttk.Checkbutton(debug_frame, text="Verbose Logging", variable=verbose_log_var).pack(anchor=tk.W, pady=5)

        # Buttons
        button_frame = ttk.Frame(settings_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        def apply_settings():
            try:
                # Update rendering settings
                self.engine_settings["rendering"]["resolution_width"] = int(width_var.get())
                self.engine_settings["rendering"]["resolution_height"] = int(height_var.get())
                self.engine_settings["rendering"]["target_fps"] = int(fps_var.get())
                self.engine_settings["rendering"]["vsync"] = vsync_var.get()
                self.engine_settings["rendering"]["render_distance"] = float(distance_var.get())
                self.engine_settings["rendering"]["anti_aliasing"] = aa_var.get()
                self.engine_settings["rendering"]["texture_filtering"] = filter_var.get()

                # Update physics settings
                self.engine_settings["physics"]["enable_physics"] = enable_physics_var.get()
                self.engine_settings["physics"]["gravity"] = float(gravity_var.get())
                self.engine_settings["physics"]["physics_iterations"] = int(iterations_var.get())
                self.engine_settings["physics"]["collision_detection"] = collision_var.get()

                # Update audio settings
                self.engine_settings["audio"]["master_volume"] = master_vol_var.get()
                self.engine_settings["audio"]["sfx_volume"] = sfx_vol_var.get()
                self.engine_settings["audio"]["music_volume"] = music_vol_var.get()
                self.engine_settings["audio"]["audio_quality"] = quality_var.get()

                # Update debug settings
                self.engine_settings["debug"]["show_fps"] = show_fps_var.get()
                self.engine_settings["debug"]["show_collision_boxes"] = show_collision_var.get()
                self.engine_settings["debug"]["enable_profiler"] = enable_profiler_var.get()
                self.engine_settings["debug"]["verbose_logging"] = verbose_log_var.get()

                self.save_settings()
                self.status_text.config(text="Engine settings updated")
                settings_window.destroy()

            except ValueError as e:
                messagebox.showerror("Error", f"Invalid value: {e}")

        ttk.Button(button_frame, text="Apply", command=apply_settings).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=settings_window.destroy).pack(side=tk.RIGHT, padx=5)

    def show_editor_settings(self):
        """Show editor settings dialog"""
        editor_window = tk.Toplevel(self.root)
        editor_window.title("Editor Settings")
        editor_window.geometry("400x400")
        editor_window.resizable(False, False)

        # Appearance settings
        ttk.Label(editor_window, text="Appearance:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, padx=10, pady=(10, 5))

        # Font settings
        font_frame = ttk.Frame(editor_window)
        font_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(font_frame, text="Font:").pack(side=tk.LEFT)
        font_var = tk.StringVar(value=self.editor_settings["appearance"]["font_family"])
        font_combo = ttk.Combobox(font_frame, textvariable=font_var, 
                                values=["Consolas", "Courier New", "Monaco", "Source Code Pro"], 
                                state="readonly", width=15)
        font_combo.pack(side=tk.LEFT, padx=5)

        ttk.Label(font_frame, text="Size:").pack(side=tk.LEFT, padx=(10, 0))
        size_var = tk.StringVar(value=str(self.editor_settings["appearance"]["font_size"]))
        size_entry = ttk.Entry(font_frame, textvariable=size_var, width=5)
        size_entry.pack(side=tk.LEFT, padx=5)

        # Theme
        theme_frame = ttk.Frame(editor_window)
        theme_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(theme_frame, text="Theme:").pack(side=tk.LEFT)
        theme_var = tk.StringVar(value=self.editor_settings["appearance"]["theme"])
        theme_combo = ttk.Combobox(theme_frame, textvariable=theme_var, 
                                 values=["dark", "light"], state="readonly")
        theme_combo.pack(side=tk.LEFT, padx=5)

        # Display options
        line_nums_var = tk.BooleanVar(value=self.editor_settings["appearance"]["show_line_numbers"])
        ttk.Checkbutton(editor_window, text="Show Line Numbers", variable=line_nums_var).pack(anchor=tk.W, padx=10, pady=2)

        syntax_var = tk.BooleanVar(value=self.editor_settings["appearance"]["syntax_highlighting"])
        ttk.Checkbutton(editor_window, text="Syntax Highlighting", variable=syntax_var).pack(anchor=tk.W, padx=10, pady=2)

        # Behavior settings
        ttk.Label(editor_window, text="Behavior:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, padx=10, pady=(20, 5))

        auto_save_var = tk.BooleanVar(value=self.editor_settings["behavior"]["auto_save"])
        ttk.Checkbutton(editor_window, text="Auto Save", variable=auto_save_var).pack(anchor=tk.W, padx=10, pady=2)

        auto_indent_var = tk.BooleanVar(value=self.editor_settings["behavior"]["auto_indent"])
        ttk.Checkbutton(editor_window, text="Auto Indent", variable=auto_indent_var).pack(anchor=tk.W, padx=10, pady=2)

        word_wrap_var = tk.BooleanVar(value=self.editor_settings["behavior"]["word_wrap"])
        ttk.Checkbutton(editor_window, text="Word Wrap", variable=word_wrap_var).pack(anchor=tk.W, padx=10, pady=2)

        # Tab size
        tab_frame = ttk.Frame(editor_window)
        tab_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(tab_frame, text="Tab Size:").pack(side=tk.LEFT)
        tab_var = tk.StringVar(value=str(self.editor_settings["behavior"]["tab_size"]))
        tab_entry = ttk.Entry(tab_frame, textvariable=tab_var, width=5)
        tab_entry.pack(side=tk.LEFT, padx=5)

        # Buttons
        button_frame = ttk.Frame(editor_window)
        button_frame.pack(fill=tk.X, padx=10, pady=20)

        def apply_editor_settings():
            try:
                # Update appearance settings
                self.editor_settings["appearance"]["font_family"] = font_var.get()
                self.editor_settings["appearance"]["font_size"] = int(size_var.get())
                self.editor_settings["appearance"]["theme"] = theme_var.get()
                self.editor_settings["appearance"]["show_line_numbers"] = line_nums_var.get()
                self.editor_settings["appearance"]["syntax_highlighting"] = syntax_var.get()

                # Update behavior settings
                self.editor_settings["behavior"]["auto_save"] = auto_save_var.get()
                self.editor_settings["behavior"]["auto_indent"] = auto_indent_var.get()
                self.editor_settings["behavior"]["word_wrap"] = word_wrap_var.get()
                self.editor_settings["behavior"]["tab_size"] = int(tab_var.get())

                self.apply_editor_appearance()
                self.save_settings()
                self.status_text.config(text="Editor settings updated")
                editor_window.destroy()

            except ValueError as e:
                messagebox.showerror("Error", f"Invalid value: {e}")

        ttk.Button(button_frame, text="Apply", command=apply_editor_settings).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=editor_window.destroy).pack(side=tk.RIGHT, padx=5)

    def apply_editor_appearance(self):
        """Apply editor appearance settings"""
        # Update code editor font and theme
        font_family = self.editor_settings["appearance"]["font_family"]
        font_size = self.editor_settings["appearance"]["font_size"]
        theme = self.editor_settings["appearance"]["theme"]

        self.code_editor.config(font=(font_family, font_size))

        if theme == "dark":
            self.code_editor.config(bg='#1e1e1e', fg='#ffffff', insertbackground='white')
        else:
            self.code_editor.config(bg='#ffffff', fg='#000000', insertbackground='black')

        # Update line numbers visibility
        if self.editor_settings["appearance"]["show_line_numbers"]:
            self.line_numbers.pack(fill=tk.BOTH, expand=True)
        else:
            self.line_numbers.pack_forget()

        # Update word wrap
        if self.editor_settings["behavior"]["word_wrap"]:
            self.code_editor.config(wrap=tk.WORD)
        else:
            self.code_editor.config(wrap=tk.NONE)

    def reset_settings(self):
        """Reset all settings to defaults"""
        if messagebox.askyesno("Reset Settings", "Are you sure you want to reset all settings to defaults?"):
            # Reset engine settings
            self.engine_settings = {
                "rendering": {
                    "resolution_width": 800,
                    "resolution_height": 600,
                    "target_fps": 60,
                    "vsync": True,
                    "render_distance": 1000,
                    "anti_aliasing": False,
                    "texture_filtering": True,
                    "max_sprites": 1000,
                    "render_scale": 1.0,
                    "sprite_batching": True,
                    "pixel_perfect": False
                },
                "physics": {
                    "gravity": 980.0,
                    "physics_iterations": 8,
                    "collision_detection": "continuous",
                    "enable_physics": True,
                    "max_velocity": 500.0,
                    "sleep_threshold": 5.0,
                    "sub_stepping": True,
                    "warm_starting": True
                },
                "audio": {
                    "master_volume": 1.0,
                    "sfx_volume": 1.0,
                    "music_volume": 1.0,
                    "audio_quality": "high"
                },
                "debug": {
                    "show_fps": False,
                    "show_collision_boxes": False,
                    "enable_profiler": False,
                    "verbose_logging": False,
                    "memory_tracking": False,
                    "performance_overlay": False,
                    "show_physics_debug": False,
                    "wireframe_mode": False,
                    "log_level": "INFO"
                },
                "development": {
                    "hot_reload": False,
                    "auto_compile": False,
                    "live_editing": False,
                    "script_debugging": False,
                    "breakpoints": False,
                    "step_debugging": False,
                    "variable_inspector": False,
                    "call_stack": False,
                    "watch_expressions": False,
                    "code_coverage": False
                },
                "optimization": {
                    "object_pooling": False,
                    "culling": False,
                    "lod_system": False,
                    "texture_streaming":False,
                    "async_loading": False,
                    "memory_optimization": False,
                    "garbage_collection": "auto",
                    "preload_assets": False,
                    "compress_textures": False,
                    "texture_atlas": False
                },
                "networking": {
                    "enable_networking": False,
                    "max_connections": 4,
                    "timeout": 10,
                    "compression": False,
                    "encryption": False,
                    "lag_compensation": False,
                    "prediction": False,
                    "interpolation": False
                }
            }

            # Reset editor settings
            self.editor_settings = {
                "appearance": {
                    "font_size": 11,
                    "font_family": "Consolas",
                    "theme": "dark",
                    "show_line_numbers": True,
                    "syntax_highlighting": True
                },
                "behavior": {
                    "auto_save": True,
                    "auto_indent": True,
                    "tab_size": 4,
                    "word_wrap": False
                }
            }

            self.apply_editor_appearance()
            self.save_settings()
            self.status_text.config(text="Settings reset to defaults")

    def save_settings(self):
        """Save settings to file"""
        try:
            settings_data = {
                "engine": self.engine_settings,
                "editor": self.editor_settings,
                "build": self.build_settings
            }

            settings_file = "axarion_studio_settings.json"
            with open(settings_file, 'w') as f:
                json.dump(settings_data, f, indent=2)

        except Exception as e:
            print(f"Failed to save settings: {e}")

    def load_settings(self):
        """Load settings from file"""
        try:
            settings_file = "axarion_studio_settings.json"
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    settings_data = json.load(f)

                if "engine" in settings_data:
                    self.engine_settings.update(settings_data["engine"])

                if "editor" in settings_data:
                    self.editor_settings.update(settings_data["editor"])

                if "build" in settings_data:
                    self.build_settings.update(settings_data["build"])

                self.apply_editor_appearance()

        except Exception as e:
            print(f"Failed to load settings: {e}")

    def get_engine_config_code(self):
        """Generate engine configuration code based on settings"""
        config_code = f"""# Engine Configuration (Generated by Axarion Studio)
engine_config = {{
    "width": {self.engine_settings["rendering"]["resolution_width"]},
    "height": {self.engine_settings["rendering"]["resolution_height"]},
    "fps": {self.engine_settings["rendering"]["target_fps"]},
    "vsync": {self.engine_settings["rendering"]["vsync"]},
    "debug": {self.engine_settings["debug"]["show_fps"]},
    "verbose": {self.engine_settings["debug"]["verbose_logging"]}
}}

# Initialize engine with configuration
engine = AxarionEngine(
    engine_config["width"], 
    engine_config["height"], 
    "My Game"
)
engine.initialize(**engine_config)
"""
        return config_code

    def show_advanced_settings(self):
        """Show advanced developer settings"""
        adv_window = tk.Toplevel(self.root)
        adv_window.title("Advanced Developer Settings")
        adv_window.geometry("600x700")
        adv_window.resizable(True, True)

        # Create notebook for tabs
        notebook = ttk.Notebook(adv_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Development tab
        dev_frame = ttk.Frame(notebook)
        notebook.add(dev_frame, text="Development")

        ttk.Label(dev_frame, text="Code Development:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=10)

        hot_reload_var = tk.BooleanVar(value=self.engine_settings["development"]["hot_reload"])
        ttk.Checkbutton(dev_frame, text="Hot Reload (auto-refresh on save)", variable=hot_reload_var).pack(anchor=tk.W, pady=2)

        auto_compile_var = tk.BooleanVar(value=self.engine_settings["development"]["auto_compile"])
        ttk.Checkbutton(dev_frame, text="Auto Compile Scripts", variable=auto_compile_var).pack(anchor=tk.W, pady=2)

        live_edit_var = tk.BooleanVar(value=self.engine_settings["development"]["live_editing"])
        ttk.Checkbutton(dev_frame, text="Live Editing (edit while running)", variable=live_edit_var).pack(anchor=tk.W, pady=2)

        ttk.Label(dev_frame, text="Debugging Tools:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(15, 5))

        script_debug_var = tk.BooleanVar(value=self.engine_settings["development"]["script_debugging"])
        ttk.Checkbutton(dev_frame, text="Advanced Script Debugging", variable=script_debug_var).pack(anchor=tk.W, pady=2)

        breakpoints_var = tk.BooleanVar(value=self.engine_settings["development"]["breakpoints"])
        ttk.Checkbutton(dev_frame, text="Breakpoint Support", variable=breakpoints_var).pack(anchor=tk.W, pady=2)

        step_debug_var = tk.BooleanVar(value=self.engine_settings["development"]["step_debugging"])
        ttk.Checkbutton(dev_frame, text="Step-by-Step Debugging", variable=step_debug_var).pack(anchor=tk.W, pady=2)

        var_inspector_var = tk.BooleanVar(value=self.engine_settings["development"]["variable_inspector"])
        ttk.Checkbutton(dev_frame, text="Variable Inspector", variable=var_inspector_var).pack(anchor=tk.W, pady=2)

        call_stack_var = tk.BooleanVar(value=self.engine_settings["development"]["call_stack"])
        ttk.Checkbutton(dev_frame, text="Call Stack Viewer", variable=call_stack_var).pack(anchor=tk.W, pady=2)

        watch_expr_var = tk.BooleanVar(value=self.engine_settings["development"]["watch_expressions"])
        ttk.Checkbutton(dev_frame, text="Watch Expressions", variable=watch_expr_var).pack(anchor=tk.W, pady=2)

        code_coverage_var = tk.BooleanVar(value=self.engine_settings["development"]["code_coverage"])
        ttk.Checkbutton(dev_frame, text="Code Coverage Analysis", variable=code_coverage_var).pack(anchor=tk.W, pady=2)

        # Optimization tab
        opt_frame = ttk.Frame(notebook)
        notebook.add(opt_frame, text="Optimization")

        ttk.Label(opt_frame, text="Performance Optimizations:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=10)

        obj_pooling_var = tk.BooleanVar(value=self.engine_settings["optimization"]["object_pooling"])
        ttk.Checkbutton(opt_frame, text="Object Pooling (reuse objects)", variable=obj_pooling_var).pack(anchor=tk.W, pady=2)

        culling_var = tk.BooleanVar(value=self.engine_settings["optimization"]["culling"])
        ttk.Checkbutton(opt_frame, text="Frustum Culling", variable=culling_var).pack(anchor=tk.W, pady=2)

        lod_var = tk.BooleanVar(value=self.engine_settings["optimization"]["lod_system"])
        ttk.Checkbutton(opt_frame, text="Level of Detail (LOD) System", variable=lod_var).pack(anchor=tk.W, pady=2)

        tex_stream_var = tk.BooleanVar(value=self.engine_settings["optimization"]["texture_streaming"])
        ttk.Checkbutton(opt_frame, text="Texture Streaming", variable=tex_stream_var).pack(anchor=tk.W, pady=2)

        async_load_var = tk.BooleanVar(value=self.engine_settings["optimization"]["async_loading"])
        ttk.Checkbutton(opt_frame, text="Asynchronous Asset Loading", variable=async_load_var).pack(anchor=tk.W, pady=2)

        mem_opt_var = tk.BooleanVar(value=self.engine_settings["optimization"]["memory_optimization"])
        ttk.Checkbutton(opt_frame, text="Memory Optimization", variable=mem_opt_var).pack(anchor=tk.W, pady=2)

        ttk.Label(opt_frame, text="Garbage Collection:").pack(anchor=tk.W, pady=(15, 5))
        gc_var = tk.StringVar(value=self.engine_settings["optimization"]["garbage_collection"])
        gc_combo = ttk.Combobox(opt_frame, textvariable=gc_var, 
                               values=["auto", "manual", "aggressive", "conservative"], state="readonly")
        gc_combo.pack(anchor=tk.W, pady=5)

        preload_var = tk.BooleanVar(value=self.engine_settings["optimization"]["preload_assets"])
        ttk.Checkbutton(opt_frame, text="Preload All Assets", variable=preload_var).pack(anchor=tk.W, pady=2)

        compress_var = tk.BooleanVar(value=self.engine_settings["optimization"]["compress_textures"])
        ttk.Checkbutton(opt_frame, text="Compress Textures", variable=compress_var).pack(anchor=tk.W, pady=2)

        atlas_var = tk.BooleanVar(value=self.engine_settings["optimization"]["texture_atlas"])
        ttk.Checkbutton(opt_frame, text="Texture Atlas Generation", variable=atlas_var).pack(anchor=tk.W, pady=2)

        # Networking tab
        net_frame = ttk.Frame(notebook)
        notebook.add(net_frame, text="Networking")

        enable_net_var = tk.BooleanVar(value=self.engine_settings["networking"]["enable_networking"])
        ttk.Checkbutton(net_frame, text="Enable Networking", variable=enable_net_var).pack(anchor=tk.W, pady=10)

        ttk.Label(net_frame, text="Max Connections:").pack(anchor=tk.W, pady=(10, 5))
        max_conn_var = tk.StringVar(value=str(self.engine_settings["networking"]["max_connections"]))
        max_conn_entry = ttk.Entry(net_frame, textvariable=max_conn_var, width=10)
        max_conn_entry.pack(anchor=tk.W, pady=5)

        ttk.Label(net_frame, text="Timeout (seconds):").pack(anchor=tk.W, pady=(10, 5))
        timeout_var = tk.StringVar(value=str(self.engine_settings["networking"]["timeout"]))
        timeout_entry = ttk.Entry(net_frame, textvariable=timeout_var, width=10)
        timeout_entry.pack(anchor=tk.W, pady=5)

        net_compress_var = tk.BooleanVar(value=self.engine_settings["networking"]["compression"])
        ttk.Checkbutton(net_frame, text="Network Compression", variable=net_compress_var).pack(anchor=tk.W, pady=5)

        encrypt_var = tk.BooleanVar(value=self.engine_settings["networking"]["encryption"])
        ttk.Checkbutton(net_frame, text="Encryption", variable=encrypt_var).pack(anchor=tk.W, pady=2)

        lag_comp_var = tk.BooleanVar(value=self.engine_settings["networking"]["lag_compensation"])
        ttk.Checkbutton(net_frame, text="Lag Compensation", variable=lag_comp_var).pack(anchor=tk.W, pady=2)

        prediction_var = tk.BooleanVar(value=self.engine_settings["networking"]["prediction"])
        ttk.Checkbutton(net_frame, text="Client-side Prediction", variable=prediction_var).pack(anchor=tk.W, pady=2)

        interp_var = tk.BooleanVar(value=self.engine_settings["networking"]["interpolation"])
        ttk.Checkbutton(net_frame, text="Entity Interpolation", variable=interp_var).pack(anchor=tk.W, pady=2)

        # Buttons
        button_frame = ttk.Frame(adv_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        def apply_advanced_settings():
            try:
                # Update development settings
                self.engine_settings["development"]["hot_reload"] = hot_reload_var.get()
                self.engine_settings["development"]["auto_compile"] = auto_compile_var.get()
                self.engine_settings["development"]["live_editing"] = live_edit_var.get()
                self.engine_settings["development"]["script_debugging"] = script_debug_var.get()
                self.engine_settings["development"]["breakpoints"] = breakpoints_var.get()
                self.engine_settings["development"]["step_debugging"] = step_debug_var.get()
                self.engine_settings["development"]["variable_inspector"] = var_inspector_var.get()
                self.engine_settings["development"]["call_stack"] = call_stack_var.get()
                self.engine_settings["development"]["watch_expressions"] = watch_expr_var.get()
                self.engine_settings["development"]["code_coverage"] = code_coverage_var.get()

                # Update optimization settings
                self.engine_settings["optimization"]["object_pooling"] = obj_pooling_var.get()
                self.engine_settings["optimization"]["culling"] = culling_var.get()
                self.engine_settings["optimization"]["lod_system"] = lod_var.get()
                self.engine_settings["optimization"]["texture_streaming"] = tex_stream_var.get()
                self.engine_settings["optimization"]["async_loading"] = async_load_var.get()
                self.engine_settings["optimization"]["memory_optimization"] = mem_opt_var.get()
                self.engine_settings["optimization"]["garbage_collection"] = gc_var.get()
                self.engine_settings["optimization"]["preload_assets"] = preload_var.get()
                self.engine_settings["optimization"]["compress_textures"] = compress_var.get()
                self.engine_settings["optimization"]["texture_atlas"] = atlas_var.get()

                # Update networking settings
                self.engine_settings["networking"]["enable_networking"] = enable_net_var.get()
                self.engine_settings["networking"]["max_connections"] = int(max_conn_var.get())
                self.engine_settings["networking"]["timeout"] = int(timeout_var.get())
                self.engine_settings["networking"]["compression"] = net_compress_var.get()
                self.engine_settings["networking"]["encryption"] = encrypt_var.get()
                self.engine_settings["networking"]["lag_compensation"] = lag_comp_var.get()
                self.engine_settings["networking"]["prediction"] = prediction_var.get()
                self.engine_settings["networking"]["interpolation"] = interp_var.get()

                self.save_settings()
                self.status_text.config(text="Advanced settings updated")
                adv_window.destroy()

            except ValueError as e:
                messagebox.showerror("Error", f"Invalid value: {e}")

        ttk.Button(button_frame, text="Apply", command=apply_advanced_settings).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=adv_window.destroy).pack(side=tk.RIGHT, padx=5)

    def show_performance_settings(self):
        """Show performance tuning dialog"""
        perf_window = tk.Toplevel(self.root)
        perf_window.title("Performance Tuning")
        perf_window.geometry("500x600")

        ttk.Label(perf_window, text="Performance & Memory Settings", font=('Arial', 12, 'bold')).pack(pady=10)

        # Rendering performance
        render_frame = ttk.LabelFrame(perf_window, text="Rendering Performance")
        render_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(render_frame, text="Max Sprites:").pack(anchor=tk.W, padx=5)
        max_sprites_var = tk.StringVar(value=str(self.engine_settings["rendering"]["max_sprites"]))
        max_sprites_entry = ttk.Entry(render_frame, textvariable=max_sprites_var, width=10)
        max_sprites_entry.pack(anchor=tk.W, padx=5, pady=2)

        ttk.Label(render_frame, text="Render Scale:").pack(anchor=tk.W, padx=5)
        render_scale_var = tk.StringVar(value=str(self.engine_settings["rendering"]["render_scale"]))
        render_scale_entry = ttk.Entry(render_frame, textvariable=render_scale_var, width=10)
        render_scale_entry.pack(anchor=tk.W, padx=5, pady=2)

        sprite_batch_var = tk.BooleanVar(value=self.engine_settings["rendering"]["sprite_batching"])
        ttk.Checkbutton(render_frame, text="Sprite Batching", variable=sprite_batch_var).pack(anchor=tk.W, padx=5, pady=2)

        pixel_perfect_var = tk.BooleanVar(value=self.engine_settings["rendering"]["pixel_perfect"])
        ttk.Checkbutton(render_frame, text="Pixel Perfect", variable=pixel_perfect_var).pack(anchor=tk.W, padx=5, pady=2)

        # Physics performance
        physics_frame = ttk.LabelFrame(perf_window, text="Physics Performance")
        physics_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(physics_frame, text="Max Velocity:").pack(anchor=tk.W, padx=5)
        max_vel_var = tk.StringVar(value=str(self.engine_settings["physics"]["max_velocity"]))
        max_vel_entry = ttk.Entry(physics_frame, textvariable=max_vel_var, width=10)
        max_vel_entry.pack(anchor=tk.W, padx=5, pady=2)

        ttk.Label(physics_frame, text="Sleep Threshold:").pack(anchor=tk.W, padx=5)
        sleep_thresh_var = tk.StringVar(value=str(self.engine_settings["physics"]["sleep_threshold"]))
        sleep_thresh_entry = ttk.Entry(physics_frame, textvariable=sleep_thresh_var, width=10)
        sleep_thresh_entry.pack(anchor=tk.W, padx=5, pady=2)

        sub_step_var = tk.BooleanVar(value=self.engine_settings["physics"]["sub_stepping"])
        ttk.Checkbutton(physics_frame, text="Sub-stepping", variable=sub_step_var).pack(anchor=tk.W, padx=5, pady=2)

        warm_start_var = tk.BooleanVar(value=self.engine_settings["physics"]["warm_starting"])
        ttk.Checkbutton(physics_frame, text="Warm Starting", variable=warm_start_var).pack(anchor=tk.W, padx=5, pady=2)

        # Debug performance
        debug_frame = ttk.LabelFrame(perf_window, text="Debug & Profiling")
        debug_frame.pack(fill=tk.X, padx=10, pady=10)

        mem_track_var = tk.BooleanVar(value=self.engine_settings["debug"]["memory_tracking"])
        ttk.Checkbutton(debug_frame, text="Memory Tracking", variable=mem_track_var).pack(anchor=tk.W, padx=5, pady=2)

        perf_overlay_var = tk.BooleanVar(value=self.engine_settings["debug"]["performance_overlay"])
        ttk.Checkbutton(debug_frame, text="Performance Overlay", variable=perf_overlay_var).pack(anchor=tk.W, padx=5, pady=2)

        physics_debug_var = tk.BooleanVar(value=self.engine_settings["debug"]["show_physics_debug"])
        ttk.Checkbutton(debug_frame, text="Physics Debug Visuals", variable=physics_debug_var).pack(anchor=tk.W, padx=5, pady=2)

        wireframe_var = tk.BooleanVar(value=self.engine_settings["debug"]["wireframe_mode"])
        ttk.Checkbutton(debug_frame, text="Wireframe Mode", variable=wireframe_var).pack(anchor=tk.W, padx=5, pady=2)

        ttk.Label(debug_frame, text="Log Level:").pack(anchor=tk.W, padx=5)
        log_level_var = tk.StringVar(value=self.engine_settings["debug"]["log_level"])
        log_level_combo = ttk.Combobox(debug_frame, textvariable=log_level_var, 
                                     values=["DEBUG", "INFO", "WARNING", "ERROR"], state="readonly")
        log_level_combo.pack(anchor=tk.W, padx=5, pady=2)

        def apply_performance_settings():
            try:
                self.engine_settings["rendering"]["max_sprites"] = int(max_sprites_var.get())
                self.engine_settings["rendering"]["render_scale"] = float(render_scale_var.get())
                self.engine_settings["rendering"]["sprite_batching"] = sprite_batch_var.get()
                self.engine_settings["rendering"]["pixel_perfect"] = pixel_perfect_var.get()

                self.engine_settings["physics"]["max_velocity"] = float(max_vel_var.get())
                self.engine_settings["physics"]["sleep_threshold"] = float(sleep_thresh_var.get())
                self.engine_settings["physics"]["sub_stepping"] = sub_step_var.get()
                self.engine_settings["physics"]["warm_starting"] = warm_start_var.get()

                self.engine_settings["debug"]["memory_tracking"] = mem_track_var.get()
                self.engine_settings["debug"]["performance_overlay"] = perf_overlay_var.get()
                self.engine_settings["debug"]["show_physics_debug"] = physics_debug_var.get()
                self.engine_settings["debug"]["wireframe_mode"] = wireframe_var.get()
                self.engine_settings["debug"]["log_level"] = log_level_var.get()

                self.save_settings()
                self.status_text.config(text="Performance settings updated")
                perf_window.destroy()

            except ValueError as e:
                messagebox.showerror("Error", f"Invalid value: {e}")

        button_frame = ttk.Frame(perf_window)
        button_frame.pack(fill=tk.X, padx=10, pady=20)

        ttk.Button(button_frame, text="Apply", command=apply_performance_settings).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=perf_window.destroy).pack(side=tk.RIGHT, padx=5)

    def show_code_analysis_settings(self):
        """Show code analysis and quality settings"""
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title("Code Analysis & Quality")
        analysis_window.geometry("450x500")

        ttk.Label(analysis_window, text="Code Quality & Analysis Tools", font=('Arial', 12, 'bold')).pack(pady=10)

        # Static analysis
        static_frame = ttk.LabelFrame(analysis_window, text="Static Analysis")
        static_frame.pack(fill=tk.X, padx=10, pady=10)

        syntax_check_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(static_frame, text="Real-time Syntax Checking", variable=syntax_check_var).pack(anchor=tk.W, padx=5, pady=2)

        style_check_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(static_frame, text="Style Guidelines (PEP8)", variable=style_check_var).pack(anchor=tk.W, padx=5, pady=2)

        complexity_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(static_frame, text="Complexity Analysis", variable=complexity_var).pack(anchor=tk.W, padx=5, pady=2)

        unused_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(static_frame, text="Unused Variables Detection", variable=unused_var).pack(anchor=tk.W, padx=5, pady=2)

        # Code metrics
        metrics_frame = ttk.LabelFrame(analysis_window, text="Code Metrics")
        metrics_frame.pack(fill=tk.X, padx=10, pady=10)

        lines_count_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(metrics_frame, text="Lines of Code Counter", variable=lines_count_var).pack(anchor=tk.W, padx=5, pady=2)

        function_count_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(metrics_frame, text="Function Complexity", variable=function_count_var).pack(anchor=tk.W, padx=5, pady=2)

        duplicate_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(metrics_frame, text="Duplicate Code Detection", variable=duplicate_var).pack(anchor=tk.W, padx=5, pady=2)

        # Code formatting
        format_frame = ttk.LabelFrame(analysis_window, text="Auto-formatting")
        format_frame.pack(fill=tk.X, padx=10, pady=10)

        auto_format_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(format_frame, text="Auto-format on Save", variable=auto_format_var).pack(anchor=tk.W, padx=5, pady=2)

        import_sort_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(format_frame, text="Sort Imports", variable=import_sort_var).pack(anchor=tk.W, padx=5, pady=2)

        trailing_spaces_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(format_frame, text="Remove Trailing Spaces", variable=trailing_spaces_var).pack(anchor=tk.W, padx=5, pady=2)

        # Documentation
        docs_frame = ttk.LabelFrame(analysis_window, text="Documentation")
        docs_frame.pack(fill=tk.X, padx=10, pady=10)

        docstring_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(docs_frame, text="Docstring Validation", variable=docstring_var).pack(anchor=tk.W, padx=5, pady=2)

        comment_ratio_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(docs_frame, text="Comment Ratio Analysis", variable=comment_ratio_var).pack(anchor=tk.W, padx=5, pady=2)

        def apply_analysis_settings():
            # These would integrate with the error checker and highlighter
            self.status_text.config(text="Code analysis settings updated")
            analysis_window.destroy()

        button_frame = ttk.Frame(analysis_window)
        button_frame.pack(fill=tk.X, padx=10, pady=20)

        ttk.Button(button_frame, text="Apply", command=apply_analysis_settings).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=analysis_window.destroy).pack(side=tk.RIGHT, padx=5)

    def export_settings(self):
        """Export settings to file"""
        file_path = filedialog.asksaveasfilename(
            title="Export Settings",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                settings_data = {
                    "engine": self.engine_settings,
                    "editor": self.editor_settings,
                    "version": "1.0",
                    "export_date": time.strftime("%Y-%m-%d %H:%M:%S")
                }

                with open(file_path, 'w') as f:
                    json.dump(settings_data, f, indent=2)

                self.status_text.config(text=f"Settings exported to {file_path}")
                messagebox.showinfo("Export Complete", "Settings exported successfully!")

            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export settings: {e}")

    def import_settings(self):
        """Import settings from file"""
        file_path = filedialog.askopenfilename(
            title="Import Settings",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'r') as f:
                    settings_data = json.load(f)

                if "engine" in settings_data:
                    self.engine_settings.update(settings_data["engine"])

                if "editor" in settings_data:
                    self.editor_settings.update(settings_data["editor"])

                self.apply_editor_appearance()
                self.save_settings()
                self.status_text.config(text=f"Settings imported from {file_path}")
                messagebox.showinfo("Import Complete", "Settings imported successfully!")

            except Exception as e:
                messagebox.showerror("Import Error", f"Failed to import settings: {e}")

    # Version information
    STUDIO_VERSION = "1.3"
    ENGINE_VERSION = "0.8.9"

    def show_build_settings(self):
        """Show simplified build settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Build Settings")
        settings_window.geometry("400x300")
        settings_window.resizable(False, False)

        # Build defaults
        build_frame = ttk.LabelFrame(settings_window, text="Default Build Options")
        build_frame.pack(fill=tk.X, padx=10, pady=10)

        include_assets_var = tk.BooleanVar(value=self.build_settings["include_assets"])
        ttk.Checkbutton(build_frame, text="Include assets folder by default", variable=include_assets_var).pack(anchor=tk.W, padx=5, pady=2)

        include_engine_var = tk.BooleanVar(value=self.build_settings["include_engine"])
        ttk.Checkbutton(build_frame, text="Include Axarion Engine", variable=include_engine_var).pack(anchor=tk.W, padx=5, pady=2)

        single_file_var = tk.BooleanVar(value=self.build_settings["single_file"])
        ttk.Checkbutton(build_frame, text="Build as single file", variable=single_file_var).pack(anchor=tk.W, padx=5, pady=2)

        compress_var = tk.BooleanVar(value=self.build_settings["compress"])
        ttk.Checkbutton(build_frame, text="Compress executable", variable=compress_var).pack(anchor=tk.W, padx=5, pady=2)

        # Buttons
        button_frame = ttk.Frame(settings_window)
        button_frame.pack(fill=tk.X, padx=10, pady=20)

        def apply_build_settings():
            self.build_settings["include_assets"] = include_assets_var.get()
            self.build_settings["include_engine"] = include_engine_var.get()
            self.build_settings["single_file"] = single_file_var.get()
            self.build_settings["compress"] = compress_var.get()
            self.save_settings()
            settings_window.destroy()

        ttk.Button(button_frame, text="Apply", command=apply_build_settings).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=settings_window.destroy).pack(side=tk.RIGHT, padx=5)

    def show_getting_started(self):
        """Show Getting Started documentation"""
        self.show_documentation_file("DOCS/GETTING_STARTED.md", "📖 Getting Started Guide")

    def show_quick_reference(self):
        """Show Quick Reference documentation"""
        self.show_documentation_file("DOCS/QUICK_REFERENCE.md", "⚡ Quick Reference")

    def show_complete_tutorial(self):
        """Show Complete Game Tutorial"""
        self.show_documentation_file("DOCS/TUTORIAL_COMPLETE_GAME.md", "🎮 Complete Game Tutorial")

    def show_attribution_guide(self):
        """Show Attribution Guide"""
        self.show_documentation_file("DOCS/Axarion_Attribution_Guide.md", "❤️ Attribution Guide")

    def show_documentation_file(self, filepath, title):
        """Show documentation file in a window"""
        doc_window = tk.Toplevel(self.root)
        doc_window.title(title)
        doc_window.geometry("800x600")
        doc_window.resizable(True, True)

        # Text widget with scrollbar
        text_frame = ttk.Frame(doc_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        text_widget = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, font=('Consolas', 10))
        text_widget.pack(fill=tk.BOTH, expand=True)

        # Load documentation content
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            text_widget.insert(tk.END, content)
            text_widget.config(state=tk.DISABLED)
        except FileNotFoundError:
            text_widget.insert(tk.END, f"Documentation file not found: {filepath}")
            text_widget.config(state=tk.DISABLED)
        except Exception as e:
            text_widget.insert(tk.END, f"Error loading documentation: {e}")
            text_widget.config(state=tk.DISABLED)

        # Close button
        button_frame = ttk.Frame(doc_window)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(button_frame, text="Close", command=doc_window.destroy).pack(side=tk.RIGHT)

    def show_about(self):
        """Show about dialog with version information"""
        about_text = f"""Axarion Studio v{self.STUDIO_VERSION} - Game Build System

🎮 Professional Game Development IDE for Axarion Engine

Version Information:
• Studio Version: {self.STUDIO_VERSION}
• Engine Version: {self.ENGINE_VERSION}

Core Features:
• 📦 One-click game packaging to EXE
• ⚡ Real-time error checking & debugging
• 🎨 Advanced syntax highlighting
• 📁 Comprehensive file explorer
• ▶️ Integrated game runner
• 🐍 Full Python support
• 🔧 Build system with dependency management

Build & Distribution:
• 🚀 Integrated EXE packager (no external tools needed!)
• 📋 Automatic dependency detection
• 📁 Asset bundling system
• 🖼️ Custom icon support
• 🗜️ Compression options
• 📄 Single-file or directory builds

Engine Integration:
• Complete Axarion Engine v{self.ENGINE_VERSION}
• Advanced physics system
• Particle effects
• Audio system with 3D positioning
• Animation system
• Asset management

Perfect for indie game developers who want a code-first approach! 

Created with ❤️ for the game development community"""

        messagebox.showinfo("About Axarion Studio", about_text)

    # ===============================
    # INTEGRATED GAME PACKAGER METHODS
    # ===============================

    def show_studio_build_dialog(self):
        """Show build dialog for Axarion Studio itself"""
        messagebox.showinfo("Build Axarion Studio", 
                           "To build Axarion Studio as executable:\n\n"
                           "1. Install PyInstaller: pip install pyinstaller\n"
                           "2. Run: pyinstaller --onefile --windowed axarion_studio.py\n"
                           "3. Find executable in dist/ folder")

    def show_build_dialog(self):
        """Show integrated build/package dialog"""
        # Special case: Building Axarion Studio itself
        current_script = os.path.abspath(__file__)
        if self.current_file and os.path.samefile(self.current_file, current_script):
            self.show_studio_build_dialog()
            return
            
        if not self.file_explorer.current_project:
            messagebox.showwarning("No Project", "Please open a project first using the file explorer.")
            return

        build_window = tk.Toplevel(self.root)
        build_window.title("📦 Build Game to EXE")
        build_window.geometry("700x800")
        build_window.resizable(True, True)

        # Main frame
        main_frame = ttk.Frame(build_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="🎮 Build Axarion Game to EXE", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 15))

        # Auto-detect project info
        project_path = self.file_explorer.current_project
        main_file = self.detect_main_file(project_path)

        # Project info (read-only)
        info_frame = ttk.LabelFrame(main_frame, text="📁 Project Information")
        info_frame.pack(fill=tk.X, pady=10)

        ttk.Label(info_frame, text="Project Folder:").pack(anchor=tk.W, padx=5)
        project_label = ttk.Label(info_frame, text=project_path, foreground="blue")
        project_label.pack(anchor=tk.W, padx=15, pady=2)

        ttk.Label(info_frame, text="Main File:").pack(anchor=tk.W, padx=5, pady=(10, 0))
        main_var = tk.StringVar(value=main_file if main_file else "main.py")
        main_combo = ttk.Combobox(info_frame, textvariable=main_var, width=50)
        main_combo['values'] = self.get_python_files(project_path)
        main_combo.pack(anchor=tk.W, padx=15, pady=2)

        # Build settings
        settings_frame = ttk.LabelFrame(main_frame, text="⚙️ Build Settings")
        settings_frame.pack(fill=tk.X, pady=10)

        # Game name
        ttk.Label(settings_frame, text="Game Name:").pack(anchor=tk.W, padx=5)
        name_var = tk.StringVar(value=self.build_settings["game_name"])
        name_entry = ttk.Entry(settings_frame, textvariable=name_var, width=50)
        name_entry.pack(anchor=tk.W, padx=15, pady=2)

        # Output folder
        ttk.Label(settings_frame, text="Output Folder:").pack(anchor=tk.W, padx=5, pady=(10, 0))
        output_frame = ttk.Frame(settings_frame)
        output_frame.pack(fill=tk.X, padx=15, pady=2)

        output_var = tk.StringVar(value=self.build_settings["output_path"] or os.path.join(project_path, "dist"))
        output_entry = ttk.Entry(output_frame, textvariable=output_var, width=45)
        output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="Browse", 
                  command=lambda: self.browse_output_folder(output_var)).pack(side=tk.RIGHT, padx=(5, 0))

        # Icon (optional)
        ttk.Label(settings_frame, text="Icon (optional):").pack(anchor=tk.W, padx=5, pady=(10, 0))
        icon_frame = ttk.Frame(settings_frame)
        icon_frame.pack(fill=tk.X, padx=15, pady=2)

        icon_var = tk.StringVar(value=self.build_settings["icon_file"])
        icon_entry = ttk.Entry(icon_frame, textvariable=icon_var, width=45)
        icon_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(icon_frame, text="Browse", 
                  command=lambda: self.browse_icon_file(icon_var)).pack(side=tk.RIGHT, padx=(5, 0))

        # Build options
        options_frame = ttk.LabelFrame(main_frame, text="📋 Build Options")
        options_frame.pack(fill=tk.X, pady=10)

        assets_var = tk.BooleanVar(value=self.build_settings["include_assets"])
        ttk.Checkbutton(options_frame, text="📁 Include assets folder", 
                       variable=assets_var).pack(anchor=tk.W, padx=5, pady=2)

        engine_var = tk.BooleanVar(value=self.build_settings["include_engine"])
        ttk.Checkbutton(options_frame, text="⚡ Include Axarion Engine (recommended)", 
                       variable=engine_var).pack(anchor=tk.W, padx=5, pady=2)

        scripting_var = tk.BooleanVar(value=self.build_settings["include_scripting"])
        ttk.Checkbutton(options_frame, text="Include utilities", 
                       variable=scripting_var).pack(anchor=tk.W, padx=5, pady=2)

        utils_var = tk.BooleanVar(value=self.build_settings["include_utils"])
        ttk.Checkbutton(options_frame, text="🔧 Include utilities", 
                       variable=utils_var).pack(anchor=tk.W, padx=5, pady=2)

        console_var = tk.BooleanVar(value=self.build_settings["show_console"])
        ttk.Checkbutton(options_frame, text="💻 Show console window", 
                       variable=console_var).pack(anchor=tk.W, padx=5, pady=2)

        single_var = tk.BooleanVar(value=self.build_settings["single_file"])
        ttk.Checkbutton(options_frame, text="📄 Single file executable (slower startup)", 
                       variable=single_var).pack(anchor=tk.W, padx=5, pady=2)

        compress_var = tk.BooleanVar(value=self.build_settings["compress"])
        ttk.Checkbutton(options_frame, text="🗜️ Compress executable", 
                       variable=compress_var).pack(anchor=tk.W, padx=5, pady=2)

        # Progress
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=10)

        self.build_progress_var = tk.StringVar(value="Ready to build")
        ttk.Label(progress_frame, textvariable=self.build_progress_var).pack()

        self.build_progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.build_progress_bar.pack(fill=tk.X, pady=5)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=15)

        ttk.Button(button_frame, text="🏗️ Check Dependencies", 
                  command=lambda: self.check_build_dependencies_dialog(build_window)).pack(side=tk.LEFT, padx=5)

        build_btn = ttk.Button(button_frame, text="📦 Build Game", 
                              command=lambda: self.build_game(
                                  project_path, main_var.get(), name_var.get(), 
                                  output_var.get(), icon_var.get(),
                                  assets_var.get(), engine_var.get(), scripting_var.get(), 
                                  utils_var.get(), console_var.get(), single_var.get(), 
                                  compress_var.get(), build_window
                              ))
        build_btn.pack(side=tk.RIGHT, padx=5)

        ttk.Button(button_frame, text="📋 Generate Spec", 
                  command=lambda: self.generate_build_spec_dialog(
                      project_path, main_var.get(), name_var.get(),
                      assets_var.get(), engine_var.get(), scripting_var.get(), 
                      utils_var.get(), icon_var.get(), console_var.get()
                  )).pack(side=tk.RIGHT, padx=5)

        # Build log
        log_frame = ttk.LabelFrame(main_frame, text="📝 Build Output")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.build_log = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD)
        self.build_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Initial log message
        self.build_log_append("🎮 Axarion Studio Build System ready")
        self.build_log_append(f"📁 Project: {project_path}")
        if main_file:
            self.build_log_append(f"🐍 Detected main file: {main_file}")

    def detect_main_file(self, project_path):
        """Auto-detect main file in project"""
        for filename in ["main.py", "game.py", "app.py", "run.py"]:
            if os.path.exists(os.path.join(project_path, filename)):
                return filename
        return None

    def get_python_files(self, project_path):
        """Get list of Python files in project"""
        python_files = []
        try:
            for file in os.listdir(project_path):
                if file.endswith('.py'):
                    python_files.append(file)
        except:
            pass
        return python_files

    def browse_output_folder(self, output_var):
        """Browse for output folder"""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            output_var.set(folder)

    def browse_icon_file(self, icon_var):
        """Browse for icon file"""
        file = filedialog.askopenfilename(
            title="Select Icon File",
            filetypes=[("Icon files", "*.ico"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file:
            icon_var.set(file)

    def build_log_append(self, message):
        """Append message to build log"""
        if hasattr(self, 'build_log'):
            self.build_log.insert(tk.END, f"{message}\n")
            self.build_log.see(tk.END)
            self.root.update()

    def check_build_dependencies(self):
        """Check build dependencies"""
        if not PYINSTALLER_AVAILABLE:
            response = messagebox.askyesno(
                "Missing PyInstaller",
                "PyInstaller is required for building games to EXE.\n\nWould you like to install it now?"
            )
            if response:
                self.install_pyinstaller()
        else:
            messagebox.showinfo("Dependencies OK", "All build dependencies are available!")

    def check_build_dependencies_dialog(self, parent_window):
        """Check dependencies with dialog output"""
        self.build_log_append("🔍 Checking build dependencies...")

        if not PYINSTALLER_AVAILABLE:
            self.build_log_append("❌ PyInstaller not found")
            response = messagebox.askyesno(
                "Missing PyInstaller",
                "PyInstaller is required for building games.\n\nInstall it now?",
                parent=parent_window
            )
            if response:
                self.install_pyinstaller_with_log()
        else:
            import PyInstaller
            self.build_log_append(f"✅ PyInstaller found: version {PyInstaller.__version__}")

        try:
            import pygame
            self.build_log_append(f"✅ Pygame found: version {pygame.version.ver}")
        except ImportError:
            self.build_log_append("⚠️ Pygame not found - may be required for games")

    def install_pyinstaller(self):
        """Install PyInstaller"""
        def install():
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], 
                             check=True, capture_output=True, text=True)
                messagebox.showinfo("Success", "PyInstaller installed successfully!")
            except subprocess.CalledProcessError as e:
                messagebox.showerror("Error", f"Failed to install PyInstaller: {e}")

        threading.Thread(target=install, daemon=True).start()

    def install_pyinstaller_with_log(self):
        """Install PyInstaller with build log output"""
        self.build_log_append("📦 Installing PyInstaller...")
        self.build_progress_bar.start()

        def install():
            try:
                result = subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], 
                                       check=True, capture_output=True, text=True)
                self.root.after(0, lambda: self.build_log_append("✅ PyInstaller installed successfully!"))
                global PYINSTALLER_AVAILABLE
                PYINSTALLER_AVAILABLE = True
            except subprocess.CalledProcessError as e:
                self.root.after(0, lambda: self.build_log_append(f"❌ Installation failed: {e}"))
            finally:
                self.root.after(0, lambda: self.build_progress_bar.stop())

        threading.Thread(target=install, daemon=True).start()

    def build_game(self, project_path, main_file, game_name, output_path, icon_file,
                   include_assets, include_engine, include_scripting, include_utils,
                   show_console, single_file, compress, parent_window):
        """Build game to executable"""
        
        # Validation
        if not main_file:
            messagebox.showerror("Error", "Please select a main file", parent=parent_window)
            return

        main_path = os.path.join(project_path, main_file)
        if not os.path.exists(main_path):
            messagebox.showerror("Error", f"Main file '{main_file}' not found", parent=parent_window)
            return

        if not game_name.strip():
            messagebox.showerror("Error", "Please enter a game name", parent=parent_window)
            return

        if not PYINSTALLER_AVAILABLE:
            messagebox.showerror("Error", "PyInstaller is not installed", parent=parent_window)
            return

        # Save build settings
        self.build_settings.update({
            "game_name": game_name,
            "output_path": output_path,
            "icon_file": icon_file,
            "include_assets": include_assets,
            "include_engine": include_engine,
            "include_scripting": include_scripting,
            "include_utils": include_utils,
            "show_console": show_console,
            "single_file": single_file,
            "compress": compress
        })

        self.save_settings()

        self.build_log_append("🚀 Starting build process...")
        self.build_progress_var.set("Building game...")
        self.build_progress_bar.start()

        def build_thread():
            try:
                # Create output directory
                os.makedirs(output_path, exist_ok=True)

                # Prepare PyInstaller arguments
                args = [
                    sys.executable, "-m", "PyInstaller",
                    "--clean",
                    "--distpath", output_path,
                    "--workpath", os.path.join(output_path, "build"),
                    "--specpath", project_path,
                    "--name", game_name
                ]

                # Build options
                if single_file:
                    args.append("--onefile")
                else:
                    args.append("--onedir")

                if not show_console:
                    args.append("--windowed")

                if icon_file and os.path.exists(icon_file):
                    args.extend(["--icon", icon_file])

                if compress:
                    args.append("--upx-dir=.")

                # Add data folders
                if include_assets:
                    assets_path = os.path.join(project_path, "assets")
                    if os.path.exists(assets_path):
                        args.extend(["--add-data", f"{assets_path}{os.pathsep}assets"])
                        self.root.after(0, lambda: self.build_log_append("✅ Including assets folder"))

                if include_engine:
                    engine_path = os.path.join(project_path, "engine")
                    if os.path.exists(engine_path):
                        args.extend(["--add-data", f"{engine_path}{os.pathsep}engine"])
                        self.root.after(0, lambda: self.build_log_append("✅ Including Axarion Engine"))

                if include_scripting:
                    scripting_path = os.path.join(project_path, "utils")
                    if os.path.exists(scripting_path):
                        args.extend(["--add-data", f"{scripting_path}{os.pathsep}utils"])
                        self.root.after(0, lambda: self.build_log_append("✅ Including utilities"))

                if include_utils:
                    utils_path = os.path.join(project_path, "utils")
                    if os.path.exists(utils_path):
                        args.extend(["--add-data", f"{utils_path}{os.pathsep}utils"])
                        self.root.after(0, lambda: self.build_log_append("✅ Including utilities"))

                # Hidden imports for common dependencies
                hidden_imports = ["pygame", "json", "math", "random", "typing", "pathlib"]
                for imp in hidden_imports:
                    args.extend(["--hidden-import", imp])

                # Main file
                args.append(main_path)

                self.root.after(0, lambda: self.build_log_append(f"📋 Running PyInstaller..."))

                # Run PyInstaller
                result = subprocess.run(args, cwd=project_path, capture_output=True, text=True)

                if result.returncode == 0:
                    exe_path = os.path.join(output_path, f"{game_name}.exe")
                    self.root.after(0, lambda: self.build_log_append("✅ Build completed successfully!"))
                    self.root.after(0, lambda: self.build_log_append(f"📦 Executable: {exe_path}"))
                    
                    # Update settings
                    self.build_settings["last_build_path"] = exe_path
                    self.save_settings()

                    # Show success dialog
                    def show_success():
                        response = messagebox.askyesno(
                            "Build Complete!",
                            f"Game successfully built!\n\nExecutable: {exe_path}\n\nOpen output folder?",
                            parent=parent_window
                        )
                        if response:
                            self.open_folder(output_path)

                    self.root.after(0, show_success)
                else:
                    self.root.after(0, lambda: self.build_log_append("❌ Build failed!"))
                    self.root.after(0, lambda: self.build_log_append(result.stderr))
                    self.root.after(0, lambda: messagebox.showerror(
                        "Build Failed", 
                        f"Build failed with errors:\n{result.stderr[:500]}...",
                        parent=parent_window
                    ))

            except Exception as e:
                self.root.after(0, lambda: self.build_log_append(f"❌ Build error: {e}"))
                self.root.after(0, lambda: messagebox.showerror(
                    "Build Error", f"An error occurred during build: {e}", parent=parent_window
                ))
            finally:
                self.root.after(0, lambda: self.build_progress_bar.stop())
                self.root.after(0, lambda: self.build_progress_var.set("Build complete"))

        threading.Thread(target=build_thread, daemon=True).start()

    def generate_build_spec(self):
        """Generate PyInstaller spec file for current project"""
        if not self.file_explorer.current_project:
            messagebox.showwarning("No Project", "Please open a project first.")
            return

        project_path = self.file_explorer.current_project
        main_file = self.detect_main_file(project_path)

        if not main_file:
            main_file = "main.py"

        self.generate_build_spec_dialog(project_path, main_file, "MyGame", True, True, True, True, "", False)

    def generate_build_spec_dialog(self, project_path, main_file, game_name, 
                                  include_assets, include_engine, include_scripting, 
                                  include_utils, icon_file, show_console):
        """Generate .spec file with given parameters"""
        
        spec_content = self.create_spec_content(
            project_path, main_file, game_name, 
            include_assets, include_engine, include_scripting, 
            include_utils, icon_file, show_console
        )

        spec_file = os.path.join(project_path, f"{game_name}.spec")

        try:
            with open(spec_file, 'w', encoding='utf-8') as f:
                f.write(spec_content)
            
            self.status_text.config(text=f"Spec file generated: {spec_file}")
            messagebox.showinfo("Spec Generated", f"PyInstaller specification created:\n{spec_file}")
            
            # Ask if user wants to open the spec file
            if messagebox.askyesno("Open Spec", "Would you like to open the spec file in the editor?"):
                self.open_file(spec_file)

        except Exception as e:
            messagebox.showerror("Error", f"Could not create spec file: {e}")

    def create_spec_content(self, project_path, main_file, game_name, 
                           include_assets, include_engine, include_scripting, 
                           include_utils, icon_file, show_console):
        """Create PyInstaller .spec file content"""
        
        # Data files
        datas = []
        if include_assets:
            assets_path = os.path.join(project_path, "assets")
            if os.path.exists(assets_path):
                datas.append("('assets', 'assets')")

        if include_engine:
            engine_path = os.path.join(project_path, "engine")
            if os.path.exists(engine_path):
                datas.append("('engine', 'engine')")

        if include_scripting:
            scripting_path = os.path.join(project_path, "utils")
            if os.path.exists(scripting_path):
                datas.append("('utils', 'utils')")

        if include_utils:
            utils_path = os.path.join(project_path, "utils")
            if os.path.exists(utils_path):
                datas.append("('utils', 'utils')")

        datas_str = "[" + ", ".join(datas) + "]" if datas else "[]"

        icon_line = f"icon='{icon_file}'," if icon_file and os.path.exists(icon_file) else ""
        console_line = "console=True," if show_console else "console=False,"

        spec_content = f"""# -*- mode: python ; coding: utf-8 -*-
# Axarion Engine Game Package Specification
# Generated by Axarion Studio Build System

a = Analysis(
    ['{main_file}'],
    pathex=[],
    binaries=[],
    datas={datas_str},
    hiddenimports=['pygame', 'json', 'math', 'random', 'typing', 'pathlib', 'threading'],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='{game_name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    {console_line}
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    {icon_line}
    version_info=None,
)
"""
        return spec_content

    def open_build_folder(self):
        """Open build output folder"""
        if self.build_settings["last_build_path"]:
            folder = os.path.dirname(self.build_settings["last_build_path"])
            self.open_folder(folder)
        elif self.build_settings["output_path"]:
            self.open_folder(self.build_settings["output_path"])
        elif self.file_explorer.current_project:
            build_folder = os.path.join(self.file_explorer.current_project, "dist")
            if os.path.exists(build_folder):
                self.open_folder(build_folder)
            else:
                messagebox.showinfo("No Build Folder", "No build output folder found.")
        else:
            messagebox.showwarning("No Project", "Please open a project first.")

    def clean_build_files(self):
        """Clean build temporary files"""
        if not self.file_explorer.current_project:
            messagebox.showwarning("No Project", "Please open a project first.")
            return

        project_path = self.file_explorer.current_project
        
        # Files and folders to clean
        to_clean = [
            os.path.join(project_path, "build"),
            os.path.join(project_path, "__pycache__"),
            os.path.join(project_path, "*.spec")
        ]

        cleaned_items = []
        for item in to_clean:
            if "*" in item:
                # Handle wildcards
                import glob
                for file in glob.glob(item):
                    try:
                        os.remove(file)
                        cleaned_items.append(file)
                    except:
                        pass
            elif os.path.exists(item):
                try:
                    if os.path.isdir(item):
                        import shutil
                        shutil.rmtree(item)
                    else:
                        os.remove(item)
                    cleaned_items.append(item)
                except:
                    pass

        if cleaned_items:
            messagebox.showinfo("Clean Complete", f"Cleaned {len(cleaned_items)} items:\n" + "\n".join(cleaned_items))
        else:
            messagebox.showinfo("Already Clean", "No build files found to clean.")

    def open_folder(self, folder_path):
        """Open folder in system file explorer"""
        try:
            if sys.platform == "win32":
                os.startfile(folder_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", folder_path])
            else:
                subprocess.run(["xdg-open", folder_path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder: {e}")

    

    def run(self):
        """Start the IDE"""
        self.status_text.config(text="Axarion Studio ready!")
        self.root.mainloop()

class cleanup:
    def __init__(self, properties):
        """
        Initializes the cleanup class with the given lists.

        :param properties: A list to store properties.
        """
        # Initialize lists
        self.properties = properties

    def add_property(self, property):
        """Adds a property to the list of properties."""
        self.properties.append(property)

    def cleanup(self):
        """Clears all the lists and sets the variables to None."""
        # Clear references
        self.properties.clear()

if __name__ == "__main__":
    # Create and run Axarion Studio
    studio = AxarionStudio()
    studio.run()
