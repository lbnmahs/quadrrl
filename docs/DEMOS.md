# Demo Videos

This page showcases trained policies evaluated across different robots and terrains.

## Overview

The videos below demonstrate the performance of policies trained using Quadrrl's benchmark suite. Each video shows a trained policy executing velocity tracking tasks in simulation. All videos show policies evaluated at the latest training checkpoint.

## Video Gallery

### ANYmal-C Robot

ANYmal-C demonstrates robust locomotion capabilities across both flat and rough terrains using both manager-based and direct control approaches.


#### Manager-Based Control

**Flat Terrain**: Demonstrates smooth velocity tracking on flat ground

<video src="../assets/anymal_c_flat.mp4" controls width="100%"></video>

**Rough Terrain**: Shows robust locomotion on challenging terrain

<video src="../assets/anymal_c_rough.mp4" controls width="100%"></video>

### ANYmal-D Robot

ANYmal-D showcases improved robustness and performance compared to ANYmal-C, particularly on rough terrain.

**Flat Terrain**: High-performance velocity tracking on flat ground

<video src="../assets/anymal_d_flat.mp4" controls width="100%"></video>

**Rough Terrain**: Enhanced terrain adaptation capabilities

<video src="../assets/anymal_d_rough.mp4" controls width="100%"></video>

### Unitree Go2 Robot

Unitree Go2 demonstrates locomotion with a different robot morphology and actuation system.

**Flat Terrain**: Velocity tracking on flat terrain

<video src="../assets/go2_flat.mp4" controls width="100%"></video>

**Rough Terrain**: Locomotion performance on rough terrain

<video src="../assets/go2_rough.mp4" controls width="100%"></video>

## Performance Notes

- All videos show policies evaluated at the latest training checkpoint
- Videos are recorded from the Isaac Lab simulation environment
- Performance metrics for these runs can be found in the analysis notebooks (`notebooks/rsl_rl_performance.ipynb`)
- Training configurations and hyperparameters are available in the respective task configuration files

## Related Documentation

- [Training Guide](TRAINING.md) - Learn how to train your own policies
- [Getting Started](GETTING_STARTED.md) - Quick start guide
- [Project Structure](STRUCTURE.md) - Understanding the codebase
