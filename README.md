# MGR VISUAL FORGE
## Model. Rig. Render. Animate. Simulate. Everything Visual.
### By Timebeunus Boyd | Money Grind Religion Inc.

---

## What Is This?

A single 1,500-line skill file that gives ANY AI coding assistant the ability to create **anything visual** — from raw geometry to final 4K output. No other AI skill covers this depth.

This isn't surface-level "use Three.js" advice. This is actual shader code, actual physics algorithms, actual rendering pipelines, actual rigging math — the same knowledge a senior graphics engineer carries after 10+ years.

## What It Covers

### Modeling
- **Signed Distance Fields (SDFs)** — 15+ primitives, smooth booleans, domain operations, ray marching
- **Marching Cubes** — SDF-to-mesh conversion with lookup tables
- **Wave Function Collapse** — procedural structure generation (buildings, dungeons, cities)
- **AI Text-to-3D** — Tripo 3.0, Meshy, Rodin, Luma, Shap-E with quality routing

### Rigging
- **Full humanoid skeleton** (65 bones, Mixamo-compatible)
- **Inverse Kinematics** — CCD and FABRIK solvers with joint constraints
- **Auto-rigging** — UniRig (SIGGRAPH 2025), Mixamo (2,500+ free animations)
- **Skinning** — Linear Blend, Dual Quaternion, blend shapes/morph targets
- **52 ARKit facial blendshapes** with real-time MediaPipe mocap

### Rendering
- **PBR materials** — 20+ parameters (metalness, roughness, clearcoat, transmission, subsurface, sheen, iridescence, anisotropy)
- **WebGPU path tracing** — full compute shader implementation
- **Deferred rendering** — G-buffer, tile-based light culling (400+ lights at 60fps)
- **Screen-space effects** — SSAO, SSR, subsurface scattering
- **4K pipeline** — TAA with Halton jitter, HDR, ACES tone mapping, adaptive resolution

### Animation
- **State machines** with blend trees (1D, 2D Simple, 2D Freeform)
- **Procedural animation** — IK foot placement, look-at, ragdoll blending
- **Motion capture retargeting** — MediaPipe body (33 landmarks) + face (468) + hands (21)

### Physics Simulation
- **Rigid body** — Rapier WASM integration
- **Cloth** — Verlet integration mass-spring system on GPU
- **Fluid** — SPH (Smoothed Particle Hydrodynamics) compute shader
- **Hair/Fur** — HairFormer neural simulation (SIGGRAPH 2025)
- **Destruction** — Voronoi fracture with fragment physics

### Mesh Operations
- **Subdivision** — Loop (triangles), Catmull-Clark (quads)
- **Decimation** — Quadric Error Metrics (Garland-Heckbert)
- **LOD generation** — 5 levels + billboard impostors
- **UV unwrapping** — LSCM, ABF++, PartUV (2025)
- **Normal map baking** — high-poly to low-poly transfer

### 2D Rendering
- **Canvas 2D** — OffscreenCanvas + Web Workers
- **PixiJS** — GPU-accelerated sprites, Spine skeletal animation
- **SVG animation** — GSAP morph, draw, motion path

### Particles
- **1M+ GPU particles** via WebGPU compute shaders
- **Curl noise** turbulence (divergence-free flow fields)
- **Force fields**, age-based interpolation, emitter systems

### Gaussian Splatting
- Photorealistic scenes from photos
- Speedy-Splat (CVPR 2025) — real-time on mobile

## Quick Install

### Claude Code
```bash
mkdir -p .claude/skills/mgr-visual-forge
cp SKILL.md .claude/skills/mgr-visual-forge/SKILL.md
```

### Cursor / Windsurf
```bash
# Append to your rules file
cat SKILL.md >> .cursorrules
```

### Any AI
Tell the AI: "Read this file" and paste SKILL.md contents.

## Tech Stack
- **Rendering:** Three.js r180+ / React Three Fiber
- **Physics:** Rapier WASM
- **Shading:** TSL (Three.js Shading Language) / GLSL / WGSL
- **Compute:** WebGPU compute shaders
- **Animation:** Three.js AnimationMixer + React Spring
- **Compression:** Draco (geometry), KTX2 (textures)
- **AI 3D:** Tripo, Meshy, Rodin, Luma, Shap-E, UniRig
- **Motion Capture:** MediaPipe (body + face + hands)
- **2D:** PixiJS, Canvas 2D, GSAP + SVG
- **Export:** glTF/GLB, FBX, MP4 (via Remotion), PNG (4K screenshots)

## Research Sources
Built from SIGGRAPH 2025, CVPR 2025, and 60+ cutting-edge research papers including:
- Speedy-Splat (fast Gaussian Splatting)
- UniRig (universal auto-rigging)
- PhysRig (physics-based rigging)
- HairFormer (neural hair simulation)
- PartUV (neural UV unwrapping)
- SoftMAC (differentiable soft body)
- OpenPBR Surface Model

## License
© 2025-2026 Money Grind Religion Inc. All Rights Reserved.
Created by Timebeunus Boyd.
