"""
MuJoCo Viewer Wrapper for WSL
Uses pygame for visualization since native viewer doesn't work on WSL
"""
import os
import sys
import pygame
import numpy as np

# Force OSMesa for headless rendering
os.environ['MUJOCO_GL'] = 'osmesa'

# Import MuJoCo after setting environment
import mujoco

print("MuJoCo viewer wrapper loaded - using pygame backend")

class MockViewer:
    """Pygame-based viewer that mimics the native viewer API"""
    
    def __init__(self, model, data, **kwargs):
        self.model = model
        self.data = data
        self.paused = False
        self.exit = False
        
        # Camera attributes
        self.cam = None
        
        # Mock additional attributes that real viewer might have
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Initialize pygame viewer
        self._setup_pygame()
    
    def _setup_pygame(self):
        pygame.init()
        # Use smaller default size that works with default framebuffer
        self.width = 640
        self.height = 480
        
        # Create a window for visualization
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("MuJoCo Viewer (Pygame)")
        
        # Create MuJoCo renderer with matching size
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        
        # Set up camera
        self.cam = mujoco.MjvCamera()
        self.cam.distance = 4.0
        self.cam.elevation = -20
        self.cam.azimuth = 90
    
    def sync(self):
        """Update the visualization."""
        if self.model and self.data:
            try:
                # Update scene with camera
                self.renderer.update_scene(self.data, camera=self.cam)
                
                # Render to pixels
                pixels = self.renderer.render()
                
                # Convert to pygame surface and display
                surf = pygame.surfarray.make_surface(pixels.swapaxes(0, 1))
                self.screen.blit(surf, (0, 0))
            except Exception as e:
                print(f"Rendering error: {e}")
                # Fill screen with background color on error
                self.screen.fill((50, 50, 50))
    
    def render_loop(self):
        """Run the rendering loop until the window is closed."""
        import mujoco
        
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
            
            # Step physics if not paused
            if not self.paused and self.model and self.data:
                mujoco.mj_step(self.model, self.data)
            
            # Render the scene
            self.sync()
            
            # Update display
            pygame.display.flip()
            
            # Control frame rate (30 FPS)
            clock.tick(30)
        
        self.close()
    
    def close(self):
        """Close the viewer."""
        if hasattr(self, 'renderer'):
            self.renderer.close()
        if pygame.get_init():
            pygame.quit()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()

# Intercept mujoco.viewer module
class MockViewerModule:
    """Mock module that provides viewer functionality"""
    
    @staticmethod
    def launch(model=None, data=None):
        """Launch the viewer with pygame backend"""
        return launch(model, data)
    
    @staticmethod
    def launch_passive(model, data):
        """Launch passive viewer (no stepping)"""
        viewer = MockViewer(model, data)
        viewer.paused = True
        viewer.render_loop()
        return viewer

# Replace the viewer module
if 'mujoco.viewer' in sys.modules:
    sys.modules['mujoco.viewer'] = MockViewerModule()
else:
    # Create a mock module
    import types
    viewer_module = types.ModuleType('viewer')
    viewer_module.launch = MockViewerModule.launch
    viewer_module.launch_passive = MockViewerModule.launch_passive
    sys.modules['mujoco.viewer'] = viewer_module
    
    # Also add it to mujoco module
    mujoco.viewer = viewer_module

def launch(model=None, data=None):
    """
    Launch the MuJoCo viewer (pygame implementation).
    
    Args:
        model: MuJoCo model object (optional)
        data: MuJoCo data object (optional)
    
    Returns:
        MockViewer instance
    """
    if model is None:
        # Create a simple test model if none provided
        import mujoco
        import numpy as np
        
        # Simple XML for a basic scene
        xml = """
        <mujoco>
            <worldbody>
                <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
                <body pos="0 0 1">
                    <joint type="free"/>
                    <geom type="box" size=".1 .1 .1" rgba="0 .9 0 1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
    elif data is None:
        # Create data if only model provided
        import mujoco
        data = mujoco.MjData(model)
    
    viewer = MockViewer(model, data)
    viewer.render_loop()  # Run the render loop
    return viewer

# Make the wrapper more compatible
mujoco.viewer.handle_viewer_lazy = lambda m, d: None

# Update the MockViewerModule class to support launch_passive
MockViewerModule.launch_passive = lambda model, data: MockViewer(model, data)
