#!/usr/bin/env python3
"""
SO100 Front View with Safe Framebuffer Size
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from so101_sim import task_suite
import numpy as np
import pygame
import mujoco

def main():
    print("SO100 Front View Controller")
    print("\nControls:")
    print("  Arrow Keys: Base rotation & shoulder")
    print("  Q/A: Elbow")
    print("  W/S: Wrist pitch") 
    print("  E/D: Wrist roll")
    print("  R/F: Gripper")
    print("  1-5: Camera views")
    print("  +/-: Zoom in/out")
    print("  Space: Reset joints")
    print("  H: Toggle help")
    print("  ESC: Exit")
    
    # Create environment
    env = task_suite.create_task_env(
        task_name='SO100HandOverBanana',
        time_limit=120.0,
        cameras=(),
        image_observation_enabled=False,
    )
    
    timestep = env.reset()
    physics = env._physics
    
    # Use maximum safe framebuffer size
    width, height = 640, 480
    
    # Initialize pygame with larger display window
    pygame.init()
    display_scale = 2  # Scale up for better visibility
    display_width = width * display_scale
    display_height = height * display_scale
    screen = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("SO100 Robot Arm - Interactive Control")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    # Create renderer with safe size
    renderer = mujoco.Renderer(physics.model.ptr, height=height, width=width)
    
    # Camera setup for better front view
    cam = mujoco.MjvCamera()
    cam.azimuth = 90    # Front view
    cam.elevation = -10  # Slightly above
    cam.distance = 2.0   # Good distance to see whole robot
    cam.lookat = np.array([0.0, -0.2, 0.6])  # Look at robot center
    
    joint_positions = physics.data.qpos[:6].copy()
    running = True
    show_help = True
    
    # Joint info
    joint_names = ["Base", "Shoulder", "Elbow", "Wrist Pitch", "Wrist Roll", "Gripper"]
    joint_limits = [
        (-3.14, 3.14),   # Base
        (-1.57, 1.57),   # Shoulder
        (-2.0, 2.0),     # Elbow
        (-2.0, 2.0),     # Wrist pitch
        (-3.14, 3.14),   # Wrist roll
        (-0.1, 0.1)      # Gripper
    ]
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    joint_positions[:] = 0
                    print("Reset all joints to zero")
                elif event.key == pygame.K_h:
                    show_help = not show_help
                # Camera views
                elif event.key == pygame.K_1:  # Front
                    cam.azimuth = 90
                    cam.elevation = -10
                    cam.distance = 2.0
                    cam.lookat = np.array([0.0, -0.2, 0.6])
                    print("Front view")
                elif event.key == pygame.K_2:  # Side
                    cam.azimuth = 0
                    cam.elevation = -10
                    cam.distance = 2.0
                    print("Side view")
                elif event.key == pygame.K_3:  # 3/4
                    cam.azimuth = 45
                    cam.elevation = -20
                    cam.distance = 2.5
                    print("3/4 view")
                elif event.key == pygame.K_4:  # Top-down
                    cam.azimuth = 90
                    cam.elevation = -80
                    cam.distance = 2.5
                    print("Top-down view")
                elif event.key == pygame.K_5:  # Wrist close-up
                    cam.azimuth = 90
                    cam.elevation = -5
                    cam.distance = 1.2
                    cam.lookat = np.array([0.0, -0.4, 0.5])
                    print("Wrist close-up")
                # Zoom
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    cam.distance = max(0.5, cam.distance - 0.2)
                elif event.key == pygame.K_MINUS:
                    cam.distance = min(5.0, cam.distance + 0.2)
        
        keys = pygame.key.get_pressed()
        
        # Joint control
        speeds = [0.03, 0.02, 0.02, 0.02, 0.03, 0.01]  # Different speeds for each joint
        
        # Base rotation
        if keys[pygame.K_LEFT]: joint_positions[0] -= speeds[0]
        if keys[pygame.K_RIGHT]: joint_positions[0] += speeds[0]
        
        # Shoulder
        if keys[pygame.K_UP]: joint_positions[1] -= speeds[1]
        if keys[pygame.K_DOWN]: joint_positions[1] += speeds[1]
        
        # Elbow
        if keys[pygame.K_q]: joint_positions[2] += speeds[2]
        if keys[pygame.K_a]: joint_positions[2] -= speeds[2]
        
        # Wrist pitch
        if keys[pygame.K_w]: joint_positions[3] += speeds[3]
        if keys[pygame.K_s]: joint_positions[3] -= speeds[3]
        
        # Wrist roll
        if keys[pygame.K_e]: joint_positions[4] += speeds[4]
        if keys[pygame.K_d]: joint_positions[4] -= speeds[4]
        
        # Gripper
        if keys[pygame.K_r]: joint_positions[5] += speeds[5]
        if keys[pygame.K_f]: joint_positions[5] -= speeds[5]
        
        # Apply joint limits
        for i in range(6):
            joint_positions[i] = np.clip(joint_positions[i], joint_limits[i][0], joint_limits[i][1])
        
        # Update physics
        physics.data.qpos[:6] = joint_positions
        physics.forward()
        env.step(np.zeros(6))
        
        # Render scene
        renderer.update_scene(physics.data.ptr, camera=cam)
        pixels = renderer.render()
        
        # Create surface and scale up for display
        surf = pygame.surfarray.make_surface(pixels.swapaxes(0, 1))
        scaled_surf = pygame.transform.scale(surf, (display_width, display_height))
        screen.blit(scaled_surf, (0, 0))
        
        # Draw UI overlay
        if show_help:
            # Joint status
            y = 20
            for i in range(6):
                # Check if joint is active
                active = False
                if i == 0: active = keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]
                elif i == 1: active = keys[pygame.K_UP] or keys[pygame.K_DOWN]
                elif i == 2: active = keys[pygame.K_q] or keys[pygame.K_a]
                elif i == 3: active = keys[pygame.K_w] or keys[pygame.K_s]
                elif i == 4: active = keys[pygame.K_e] or keys[pygame.K_d]
                elif i == 5: active = keys[pygame.K_r] or keys[pygame.K_f]
                
                color = (0, 255, 0) if active else (255, 255, 100)
                
                # Show joint name, value, and percentage of range
                value = joint_positions[i]
                min_val, max_val = joint_limits[i]
                percent = (value - min_val) / (max_val - min_val) * 100 if max_val != min_val else 50
                
                text = font.render(f"{joint_names[i]}: {value:6.2f} ({percent:3.0f}%)", True, color)
                screen.blit(text, (20, y))
                
                # Draw progress bar
                bar_x = 250
                bar_width = 150
                bar_height = 15
                pygame.draw.rect(screen, (50, 50, 50), (bar_x, y, bar_width, bar_height))
                fill_width = int(bar_width * percent / 100)
                pygame.draw.rect(screen, color, (bar_x, y, fill_width, bar_height))
                
                y += 30
            
            # Camera info
            cam_text = font.render(f"View: Az={cam.azimuth:.0f}° El={cam.elevation:.0f}° Dist={cam.distance:.1f}m", True, (100, 200, 255))
            screen.blit(cam_text, (20, y + 20))
            
            # Controls reminder
            controls = [
                "1-5: Camera views | +/-: Zoom",
                "H: Hide overlay | Space: Reset"
            ]
            y = display_height - 60
            for ctrl in controls:
                text = font.render(ctrl, True, (180, 180, 180))
                screen.blit(text, (20, y))
                y += 25
        
        pygame.display.flip()
        clock.tick(30)
    
    renderer.close()
    pygame.quit()
    print("Viewer closed")

if __name__ == "__main__":
    main()
