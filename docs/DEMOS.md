# Demo Videos

This page showcases trained policies evaluated across different robots and terrains.

## Overview

The videos below demonstrate the performance of policies trained using Quadrrl's benchmark suite. Each video shows a trained policy executing velocity tracking tasks in simulation. All videos show policies evaluated at the latest training checkpoint.

## Video Gallery

### ANYmal-C Robot

ANYmal-C demonstrates robust locomotion capabilities across both flat and rough terrains using both manager-based and direct control approaches.

#### Direct Control

- **Flat Terrain**: Direct joint control on flat terrain
  - [View Video](../assets/anymal_c_flat_direct.mp4)
  
#### Manager-Based Control

- **Flat Terrain**: Demonstrates smooth velocity tracking on flat ground
  - [View Video](../assets/anymal_c_flat.mp4)
  
- **Rough Terrain**: Shows robust locomotion on challenging terrain
  - [View Video](../assets/anymal_c_rough.mp4)

### ANYmal-D Robot

ANYmal-D showcases improved robustness and performance compared to ANYmal-C, particularly on rough terrain.

- **Flat Terrain**: High-performance velocity tracking on flat ground
  - [View Video](../assets/anymal_d_flat.mp4)
  
- **Rough Terrain**: Enhanced terrain adaptation capabilities
  - [View Video](../assets/anymal_d_rough.mp4)

### Unitree Go2 Robot

Unitree Go2 demonstrates locomotion with a different robot morphology and actuation system.

- **Flat Terrain**: Velocity tracking on flat terrain
  - [View Video](../assets/go2_flat.mp4)
  
- **Rough Terrain**: Locomotion performance on rough terrain
  - [View Video](../assets/go2_rough.mp4)

## Performance Notes

- All videos show policies evaluated at the latest training checkpoint
- Videos are recorded from the Isaac Lab simulation environment
- Performance metrics for these runs can be found in the analysis notebooks (`notebooks/rsl_rl_performance.ipynb`)
- Training configurations and hyperparameters are available in the respective task configuration files

## Related Documentation

- [Training Guide](TRAINING.md) - Learn how to train your own policies
- [Getting Started](GETTING_STARTED.md) - Quick start guide
- [Project Structure](STRUCTURE.md) - Understanding the codebase
