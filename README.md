# SO101_Sim

SO101_Sim is a MuJoCo simulation environment for the SO-101 robot arm.

## Tested Setup

- WSL2 Ubuntu 24.04
- Python 3.11
- Conda environment

## Installation

```bash
# Clone repository
git clone https://github.com/ramkumarkoppu/so101_sim.git
cd so101_sim

# Create conda environment
conda create -n so101-env python=3.11
conda activate so101-env

# Install dependencies
pip install -e .
pip install pygame
```

## WSL Viewer

Since MuJoCo's native viewer doesn't work on WSL, use the included pygame viewer:

```bash
python so101_front_view_safe.py
```

### Files
- `mujoco_viewer_wrapper.py` - Intercepts MuJoCo viewer calls
- `so101_front_view_safe.py` - Pygame-based viewer (640x480 rendered, 1280x960 displayed)

### Controls
- **Arrow Keys**: Base rotation (←/→), Shoulder (↑/↓)
- **Q/A**: Elbow
- **W/S**: Wrist pitch
- **E/D**: Wrist roll
- **R/F**: Gripper
- **1-5**: Camera views
- **Space**: Reset joints
- **ESC**: Exit

### Usage in Scripts

```python
import mujoco_viewer_wrapper  # Import first!
import mujoco
from so101_sim import task_suite

env = task_suite.create_task_env(
    task_name='SO100HandOverBanana',
    time_limit=120.0,
    cameras=(),
    image_observation_enabled=False,
)

timestep = env.reset()
physics = env._physics
mujoco.viewer.launch(physics.model.ptr, physics.data.ptr)
```

## Notes

- Framebuffer limited to 640x480 on WSL
- The viewer wrapper automatically sets osmesa rendering
- Viewer runs at 30 FPS