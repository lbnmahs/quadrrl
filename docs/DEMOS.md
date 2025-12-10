# Simulation Videos

This page showcases trained policies evaluated across different robots and terrains.

## Overview

The videos below demonstrate the performance of policies trained using Quadrrl's benchmark suite. Each video shows a trained policy executing velocity tracking tasks in simulation. All videos show policies evaluated at the latest training checkpoint.

> **Note**: Videos are embedded for local viewing. On GitHub, click the "Video Link" links below each video to watch them.

## Environment Gallery

### ANYmal-C Robot

ANYmal-C demonstrates robust locomotion capabilities across both flat and rough terrains using both manager-based and direct control approaches.

#### Direct Control

**Flat Terrain**: Direct joint control on flat terrain

<video src="../assets/anymal_c_flat_direct.mp4" controls width="100%"></video>

[Video Link](https://raw.githubusercontent.com/lbnmahs/quadrrl/devel/assets/anymal_c_flat_direct.mp4)

#### Manager-Based Control

**Flat Terrain**: Demonstrates smooth velocity tracking on flat ground

<video src="../assets/anymal_c_flat.mp4" controls width="100%"></video>

[Video Link](https://raw.githubusercontent.com/lbnmahs/quadrrl/devel/assets/anymal_c_flat.mp4)

**Rough Terrain**: Shows robust locomotion on challenging terrain

<video src="../assets/anymal_c_rough.mp4" controls width="100%"></video>

[Video Link](https://raw.githubusercontent.com/lbnmahs/quadrrl/devel/assets/anymal_c_rough.mp4)

### ANYmal-D Robot

ANYmal-D showcases improved robustness and performance compared to ANYmal-C, particularly on rough terrain.

**Flat Terrain**: High-performance velocity tracking on flat ground

<video src="../assets/anymal_d_flat.mp4" controls width="100%"></video>

[Video Link](https://raw.githubusercontent.com/lbnmahs/quadrrl/devel/assets/anymal_d_flat.mp4)

**Rough Terrain**: Enhanced terrain adaptation capabilities

<video src="../assets/anymal_d_rough.mp4" controls width="100%"></video>

[Video Link](https://raw.githubusercontent.com/lbnmahs/quadrrl/devel/assets/anymal_d_rough.mp4)

### Unitree Go2 Robot

Unitree Go2 demonstrates locomotion with a different robot morphology and actuation system.

**Flat Terrain**: Velocity tracking on flat terrain

<video src="../assets/go2_flat.mp4" controls width="100%"></video>

[Video Link](https://raw.githubusercontent.com/lbnmahs/quadrrl/devel/assets/go2_flat.mp4)

**Rough Terrain**: Locomotion performance on rough terrain

<video src="../assets/go2_rough.mp4" controls width="100%"></video>

[Video Link](https://raw.githubusercontent.com/lbnmahs/quadrrl/devel/assets/go2_rough.mp4)

### Spot Robot

Spot demonstrates velocity tracking on both flat and rough terrains using the manager-based controller.

**Flat Terrain**: Smooth and accurate velocity tracking on flat ground  
<video src="../assets/spot_flat.mp4" controls width="100%"></video>  
[Video Link](https://raw.githubusercontent.com/lbnmahs/quadrrl/devel/assets/spot_flat.mp4)

**Rough Terrain**: High-speed locomotion over diverse terrain  
<video src="../assets/spot_rough.mp4" controls width="100%"></video>  
[Video Link](https://raw.githubusercontent.com/lbnmahs/quadrrl/devel/assets/spot_rough.mp4)

> **Reward note:** Spot uses a more gait-centric reward structure (gait phase shaping, foot clearance, air-time balance) than the generic locomotion rewards for ANYmal/Go2. This enables richer gait coordination studies and is also leveraged in the Spot MARL leg-agent setup.

## Performance Notes

- All videos show policies evaluated at the latest training checkpoint
- Videos are recorded from the Isaac Lab simulation environment
- Performance metrics for these runs can be found in the analysis notebooks (`notebooks/rsl_rl_performance.ipynb`)
- Training configurations and hyperparameters are available in the respective task configuration files

## Related Documentation

- [Training Guide](TRAINING.md) - Learn how to train your own policies
- [Getting Started](GETTING_STARTED.md) - Quick start guide
- [Project Structure](STRUCTURE.md) - Understanding the codebase
