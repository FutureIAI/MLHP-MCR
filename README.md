# MLHP-MCR
Multi-Light Hyperbolic Prior-guided Multi-Channel Repair (MLHP-MCR) for light-and-shadow restoration of deformed Flexible Printed Circuit (FPC) images. The framework combines complementary observations under coaxial, bar, and ring illumination with hyperbolic deformation priors for shadow, specular-reflection, and local luminance restoration.

## Introduction

Flexible Printed Circuits (FPCs) are prone to bending, warping, and local creases
during production and visual inspection. These deformations change the surface
geometry and illumination response, producing shadows, specular reflections,
and local luminance variations that may obscure circuit traces, textures, and
surface defects.

MLHP-MCR exploits complementary information from three illumination conditions:
coaxial, bar, and ring illumination. A Blender-based simulation pipeline is first
used to generate pixel-aligned deformed-normal image pairs. Hyperbolic feature
representations are then extracted to provide deformation-related spatial priors.
Finally, the RGB images and hyperbolic features from the three illumination
conditions are jointly processed by a modified Restormer for image restoration.

## Components

The proposed framework mainly contains three components:

1. Blender-Based Data Generation
   - Real FPC texture mapping
   - Random deformation simulation
   - Multi-illumination imaging degradation simulation
   - Paired deformed-normal image generation

2. Hyperbolic Feature Enhancement
   - DeepLabV3+ feature extraction
   - Poincaré-ball mapping
   - Local hyperbolic feature aggregation
   - PCA-based dimensionality reduction
   - Three-channel hyperbolic prior generation

3. Multi-Channel Image Restoration
   - Coaxial, bar, and ring RGB image input
   - Hyperbolic-prior input
   - 18-channel joint input
   - Modified Restormer
   - Shadow-specific restoration constraints
   - Specular-reflection restoration constraints
