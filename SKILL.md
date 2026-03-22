---
name: mgr-visual-forge
description: The ultimate 3D/2D/4K visual creation engine — model, rig, render, mesh, animate, simulate, compose, and export anything visual. Combines procedural modeling (SDFs, marching cubes, WFC), AI generation (Gaussian Splatting, NeRF, text-to-3D), rigging (IK/FK, UniRig, auto-skinning), rendering (PBR, path tracing, WebGPU compute, deferred pipelines), animation (state machines, blend trees, procedural, mocap retargeting), physics (rigid, soft, cloth, fluid, hair/fur), 2D (Canvas, SVG, PixiJS, skeletal), 4K (TAA, HDR, tone mapping, upscaling), VFX (particles, volumetrics, destruction, crowd sim), and mesh ops (subdivision, decimation, booleans, UV unwrapping, LOD, normal baking). Every technique from SIGGRAPH 2025, CVPR 2025, and cutting-edge web graphics research.
user-invocable: true
allowed-tools: Read, Write, Edit, Glob, Grep, Bash, Agent, WebSearch, WebFetch
---

# MGR VISUAL FORGE
## Model. Rig. Render. Animate. Simulate. Everything Visual.
### Created by Timebeunus Boyd | Money Grind Religion Inc.
### © 2025-2026 Money Grind Religion Inc. All Rights Reserved.

---

> **What this is:** The most comprehensive visual creation skill ever built for an AI coding assistant. It doesn't just know 3D — it knows every vertex, every shader instruction, every physics timestep, every pixel pipeline from raw geometry to final 4K output. It combines 60+ research papers, SIGGRAPH/CVPR 2025 breakthroughs, and production-proven patterns into one unified system.

> **When to use:** Any time the user asks to create, model, rig, animate, render, simulate, compose, or export anything visual — 3D scenes, 2D graphics, 4K renders, games, films, VFX, diagrams, or procedural art.

---

## CHAPTER 1: CORE STACK & ARCHITECTURE

### Primary Stack
```
Rendering Engine:    Three.js r180+ (WebGL2 → WebGPU auto-detection)
React Binding:       React Three Fiber (R3F) — declarative scene graph
Utilities:           drei — 200+ helper components
Physics:             Rapier WASM — rigid/soft body, character controller
Post-Processing:     @react-three/postprocessing — 15+ GPU effects
Particles:           three.quarks / wawa-vfx / custom compute shaders
Animation:           Three.js AnimationMixer + React Spring + custom state machines
Shading Language:    TSL (Three.js Shading Language) — compiles to GLSL/WGSL
Compute:             WebGPU compute shaders (particle sim, physics, SDF eval)
Compression:         Draco (geometry), KTX2 (textures — 70% size reduction)
Model Format:        glTF 2.0 / GLB (universal standard)
```

### Architecture Layers
```
┌─────────────────────────────────────────────────┐
│  APPLICATION LAYER                               │
│  Scene Composer → Asset Manager → Export Pipeline │
├─────────────────────────────────────────────────┤
│  SIMULATION LAYER                                │
│  Physics → Particles → Cloth → Fluid → Hair     │
├─────────────────────────────────────────────────┤
│  ANIMATION LAYER                                 │
│  State Machine → Blend Trees → IK → Procedural  │
├─────────────────────────────────────────────────┤
│  RENDERING LAYER                                 │
│  PBR → Post-FX → TAA → HDR → Tone Mapping      │
├─────────────────────────────────────────────────┤
│  GEOMETRY LAYER                                  │
│  Mesh Ops → SDFs → Procedural → LOD → Instancing│
├─────────────────────────────────────────────────┤
│  GPU LAYER                                       │
│  WebGPU Compute → Shader Programs → Buffer Mgmt │
└─────────────────────────────────────────────────┘
```

### Scene Setup (Production Template)
```typescript
import { Canvas } from '@react-three/fiber';
import { Environment, ContactShadows, AdaptiveDpr, AdaptiveEvents, Preload } from '@react-three/drei';
import { EffectComposer, Bloom, Vignette, SMAA, ToneMapping } from '@react-three/postprocessing';
import { Physics } from '@react-three/rapier';
import { Suspense } from 'react';

export function Scene() {
  return (
    <Canvas
      shadows
      dpr={[1, 2]}
      gl={{ antialias: false, powerPreference: 'high-performance' }}
      camera={{ position: [0, 5, 10], fov: 55, near: 0.1, far: 1000 }}
    >
      <AdaptiveDpr pixelated />
      <AdaptiveEvents />
      <color attach="background" args={['#0a0a0a']} />

      <Suspense fallback={null}>
        <Environment preset="city" background blur={0.5} />
        <Physics gravity={[0, -9.81, 0]} timeStep="vary">
          {/* Scene content */}
        </Physics>
        <ContactShadows position={[0, -0.01, 0]} opacity={0.5} blur={2} />
      </Suspense>

      <EffectComposer multisampling={0}>
        <SMAA />
        <Bloom intensity={0.3} luminanceThreshold={0.9} luminanceSmoothing={0.3} />
        <Vignette eskil={false} offset={0.1} darkness={0.5} />
        <ToneMapping mode={4} /> {/* ACES Filmic */}
      </EffectComposer>

      <Preload all />
    </Canvas>
  );
}
```

---

## CHAPTER 2: MODELING

### 2.1 Procedural Mesh Generation

#### Signed Distance Fields (SDFs)
The mathematical backbone of procedural modeling. Every point in 3D space stores its signed distance to the nearest surface. Negative = inside, positive = outside, zero = surface.

```glsl
// Primitive SDFs
float sdSphere(vec3 p, float r) { return length(p) - r; }
float sdBox(vec3 p, vec3 b) { vec3 q = abs(p) - b; return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0); }
float sdCylinder(vec3 p, float h, float r) { vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h); return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)); }
float sdTorus(vec3 p, vec2 t) { vec2 q = vec2(length(p.xz) - t.x, p.y); return length(q) - t.y; }
float sdCapsule(vec3 p, vec3 a, vec3 b, float r) { vec3 pa = p - a, ba = b - a; float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0); return length(pa - ba * h) - r; }

// Boolean Operations (CSG)
float opUnion(float d1, float d2) { return min(d1, d2); }
float opSubtract(float d1, float d2) { return max(-d1, d2); }
float opIntersect(float d1, float d2) { return max(d1, d2); }

// Smooth Boolean (blending)
float opSmoothUnion(float d1, float d2, float k) {
  float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
  return mix(d2, d1, h) - k * h * (1.0 - h);
}
float opSmoothSubtract(float d1, float d2, float k) {
  float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
  return mix(d2, -d1, h) + k * h * (1.0 - h);
}

// Domain Operations
vec3 opRepeat(vec3 p, vec3 c) { return mod(p + 0.5 * c, c) - 0.5 * c; } // Infinite repetition
vec3 opTwist(vec3 p, float k) { float c = cos(k * p.y); float s = sin(k * p.y); mat2 m = mat2(c, -s, s, c); return vec3(m * p.xz, p.y); }
vec3 opBend(vec3 p, float k) { float c = cos(k * p.x); float s = sin(k * p.x); mat2 m = mat2(c, -s, s, c); return vec3(p.x, m * p.yz); }

// Ray Marching (rendering SDFs)
float rayMarch(vec3 ro, vec3 rd) {
  float t = 0.0;
  for (int i = 0; i < 256; i++) {
    vec3 p = ro + rd * t;
    float d = sceneSDF(p);
    if (d < 0.001) return t;  // Hit
    if (t > 100.0) break;      // Miss
    t += d;
  }
  return -1.0;
}

// Normal from SDF (central differences)
vec3 calcNormal(vec3 p) {
  vec2 e = vec2(0.0001, 0.0);
  return normalize(vec3(
    sceneSDF(p + e.xyy) - sceneSDF(p - e.xyy),
    sceneSDF(p + e.yxy) - sceneSDF(p - e.yxy),
    sceneSDF(p + e.yyx) - sceneSDF(p - e.yyx)
  ));
}
```

#### SDF → Mesh Conversion (Marching Cubes)
```typescript
// Marching Cubes: convert SDF volume to triangle mesh
// 256 possible cube configurations → precomputed lookup tables

function marchingCubes(sdf: (p: Vec3) => number, bounds: Box3, resolution: number): Mesh {
  const step = bounds.size / resolution;
  const vertices: number[] = [];
  const indices: number[] = [];

  for (let x = 0; x < resolution; x++) {
    for (let y = 0; y < resolution; y++) {
      for (let z = 0; z < resolution; z++) {
        // Sample 8 corners of cube
        const corners = getCubeCorners(x, y, z, step, bounds.min);
        const values = corners.map(c => sdf(c));

        // Determine cube configuration (8 bits → 0-255)
        let cubeIndex = 0;
        for (let i = 0; i < 8; i++) {
          if (values[i] < 0) cubeIndex |= (1 << i);
        }

        // Skip empty/full cubes
        if (EDGE_TABLE[cubeIndex] === 0) continue;

        // Interpolate edge intersections
        const edgeVertices = interpolateEdges(corners, values, cubeIndex);

        // Generate triangles from lookup table
        for (let i = 0; TRI_TABLE[cubeIndex][i] !== -1; i += 3) {
          const idx = vertices.length / 3;
          vertices.push(...edgeVertices[TRI_TABLE[cubeIndex][i]]);
          vertices.push(...edgeVertices[TRI_TABLE[cubeIndex][i + 1]]);
          vertices.push(...edgeVertices[TRI_TABLE[cubeIndex][i + 2]]);
          indices.push(idx, idx + 1, idx + 2);
        }
      }
    }
  }

  return createMesh(vertices, indices);
}
```

#### Wave Function Collapse (Procedural Structures)
```typescript
// 3D WFC for generating structures (buildings, dungeons, cities)
interface WFCTile {
  id: string;
  model: GLTF;
  sockets: { px: string; nx: string; py: string; ny: string; pz: string; nz: string; };
  weight: number;
  rotations: number[]; // 0, 90, 180, 270
}

class WFC3D {
  grid: Set<string>[][][][]; // [x][y][z] = set of possible tile IDs
  size: [number, number, number];

  constructor(tiles: WFCTile[], size: [number, number, number]) {
    // Initialize: every cell can be any tile
    this.grid = Array.from({ length: size[0] }, () =>
      Array.from({ length: size[1] }, () =>
        Array.from({ length: size[2] }, () =>
          new Set(tiles.map(t => t.id))
        )
      )
    );
  }

  // Core algorithm: Observe → Propagate → Repeat
  solve(): boolean {
    while (!this.isCollapsed()) {
      const cell = this.findLowestEntropy(); // Most constrained cell
      if (!cell) return false; // Contradiction

      this.observe(cell);    // Collapse to one tile (weighted random)
      this.propagate(cell);  // Remove incompatible neighbors (constraint propagation)
    }
    return true;
  }

  findLowestEntropy(): [number, number, number] | null {
    let minEntropy = Infinity;
    let candidates: [number, number, number][] = [];
    // Find uncollapsed cell with fewest possibilities
    for (let x = 0; x < this.size[0]; x++)
      for (let y = 0; y < this.size[1]; y++)
        for (let z = 0; z < this.size[2]; z++) {
          const entropy = this.grid[x][y][z].size;
          if (entropy <= 1) continue;
          if (entropy < minEntropy) { minEntropy = entropy; candidates = [[x, y, z]]; }
          else if (entropy === minEntropy) candidates.push([x, y, z]);
        }
    // Random among ties (adds variety)
    return candidates.length > 0 ? candidates[Math.floor(Math.random() * candidates.length)] : null;
  }

  propagate(start: [number, number, number]): void {
    const stack = [start];
    while (stack.length > 0) {
      const [x, y, z] = stack.pop()!;
      const possible = this.grid[x][y][z];
      // Check all 6 neighbors
      for (const [dx, dy, dz, face, oppFace] of DIRECTIONS) {
        const nx = x + dx, ny = y + dy, nz = z + dz;
        if (!this.inBounds(nx, ny, nz)) continue;
        const neighbor = this.grid[nx][ny][nz];
        const validSockets = new Set([...possible].flatMap(id => this.tiles[id].sockets[face]));
        const before = neighbor.size;
        for (const nId of neighbor) {
          if (!validSockets.has(this.tiles[nId].sockets[oppFace])) neighbor.delete(nId);
        }
        if (neighbor.size < before) stack.push([nx, ny, nz]); // Changed → re-propagate
      }
    }
  }
}
```

### 2.2 AI-Assisted 3D Generation

#### Text-to-3D Pipeline
```typescript
// Multi-provider text-to-3D with quality routing
interface Text3DProvider {
  name: string;
  generate(prompt: string, options?: GenerateOptions): Promise<GLTFAsset>;
  quality: 'draft' | 'standard' | 'high';
  speed: 'fast' | 'medium' | 'slow';
  hasPBR: boolean;
  hasRig: boolean;
}

const providers: Text3DProvider[] = [
  { name: 'Tripo AI 3.0', quality: 'high', speed: 'medium', hasPBR: true, hasRig: true },
  { name: 'Meshy', quality: 'high', speed: 'medium', hasPBR: true, hasRig: true },
  { name: 'Rodin', quality: 'high', speed: 'slow', hasPBR: true, hasRig: false },
  { name: 'Luma Genie', quality: 'standard', speed: 'fast', hasPBR: false, hasRig: false },
  { name: 'Shap-E (local)', quality: 'draft', speed: 'fast', hasPBR: false, hasRig: false },
];

// Route to best provider based on need
function selectProvider(need: { quality: string; needsRig: boolean; needsPBR: boolean }): Text3DProvider {
  return providers.find(p =>
    p.quality >= need.quality &&
    (!need.needsRig || p.hasRig) &&
    (!need.needsPBR || p.hasPBR)
  ) || providers[0];
}
```

#### Gaussian Splatting (Photorealistic Scenes)
```typescript
// 3D Gaussian Splatting: scene as millions of learnable 3D Gaussians
// Each Gaussian: position (mean), shape (covariance), color (spherical harmonics), opacity

// Loading a .splat file in R3F
import { Splat } from '@react-three/drei';

function PhotorealisticScene() {
  return (
    <Canvas>
      <Splat
        src="/scene.splat"
        position={[0, 0, 0]}
        scale={1}
        toneMapped={false}
      />
    </Canvas>
  );
}

// Converting photos to splat (pipeline):
// 1. COLMAP: Structure from Motion → sparse point cloud
// 2. Initialize Gaussians at SfM points
// 3. Differentiable rasterization: tile-based, front-to-back blending
// 4. Optimize via photometric loss (L1 + D-SSIM)
// 5. Adaptive density control: split large Gaussians, prune transparent ones
// 6. Export .splat or .ply

// Speedy-Splat (CVPR 2025): 2x speedup through accurate primitive localization,
// 6x through 90%+ scene pruning — enables real-time mobile rendering
```

---

## CHAPTER 3: RIGGING

### 3.1 Skeleton Systems

#### Bone Hierarchy
```typescript
interface Bone {
  name: string;
  parent: string | null;
  position: [number, number, number];     // Local position relative to parent
  rotation: [number, number, number, number]; // Quaternion (x, y, z, w)
  scale: [number, number, number];
  children: string[];
}

// Standard humanoid skeleton (65 bones — Mixamo compatible)
const HUMANOID_SKELETON: Record<string, string | null> = {
  'Hips': null,                    // ROOT
  'Spine': 'Hips',
  'Spine1': 'Spine',
  'Spine2': 'Spine1',
  'Neck': 'Spine2',
  'Head': 'Neck',
  'HeadTop_End': 'Head',
  // Left arm chain
  'LeftShoulder': 'Spine2',
  'LeftArm': 'LeftShoulder',
  'LeftForeArm': 'LeftArm',
  'LeftHand': 'LeftForeArm',
  'LeftHandThumb1': 'LeftHand', /*...thumb2, thumb3...*/
  'LeftHandIndex1': 'LeftHand', /*...index2, index3...*/
  'LeftHandMiddle1': 'LeftHand', /*...*/
  'LeftHandRing1': 'LeftHand', /*...*/
  'LeftHandPinky1': 'LeftHand', /*...*/
  // Right arm (mirror)
  'RightShoulder': 'Spine2', /*...*/
  // Left leg chain
  'LeftUpLeg': 'Hips',
  'LeftLeg': 'LeftUpLeg',
  'LeftFoot': 'LeftLeg',
  'LeftToeBase': 'LeftFoot',
  // Right leg (mirror)
  'RightUpLeg': 'Hips', /*...*/
};
```

#### Inverse Kinematics (IK)
```typescript
// CCD (Cyclic Coordinate Descent) IK — fast, iterative, works for any chain length
function solveCCDIK(chain: Bone[], target: Vec3, iterations: number = 10, threshold: number = 0.01): void {
  for (let iter = 0; iter < iterations; iter++) {
    // Iterate from end effector parent to root
    for (let i = chain.length - 2; i >= 0; i--) {
      const bone = chain[i];
      const effector = chain[chain.length - 1];

      const bonePos = getWorldPosition(bone);
      const effectorPos = getWorldPosition(effector);

      // Vector from bone to effector
      const toEffector = normalize(sub(effectorPos, bonePos));
      // Vector from bone to target
      const toTarget = normalize(sub(target, bonePos));

      // Rotation to align effector direction with target direction
      const axis = cross(toEffector, toTarget);
      const angle = Math.acos(clamp(dot(toEffector, toTarget), -1, 1));

      if (length(axis) > 0.0001) {
        bone.rotation = multiplyQuaternions(
          bone.rotation,
          quaternionFromAxisAngle(normalize(axis), angle)
        );
      }

      // Apply joint constraints (hinge, ball-socket, twist limits)
      applyConstraints(bone);
    }

    // Check convergence
    if (distance(getWorldPosition(chain[chain.length - 1]), target) < threshold) break;
  }
}

// FABRIK (Forward And Backward Reaching IK) — smoother, more natural results
function solveFABRIK(chain: Vec3[], target: Vec3, boneLengths: number[], iterations: number = 10): Vec3[] {
  const root = chain[0].clone();

  for (let iter = 0; iter < iterations; iter++) {
    // BACKWARD pass: move end toward target, propagate to root
    chain[chain.length - 1] = target.clone();
    for (let i = chain.length - 2; i >= 0; i--) {
      const dir = normalize(sub(chain[i], chain[i + 1]));
      chain[i] = add(chain[i + 1], scale(dir, boneLengths[i]));
    }

    // FORWARD pass: restore root, propagate to end
    chain[0] = root.clone();
    for (let i = 1; i < chain.length; i++) {
      const dir = normalize(sub(chain[i], chain[i - 1]));
      chain[i] = add(chain[i - 1], scale(dir, boneLengths[i - 1]));
    }
  }

  return chain;
}
```

### 3.2 Skinning (Mesh Deformation)

#### Linear Blend Skinning (LBS)
```glsl
// Vertex shader: deform mesh vertices based on bone transforms
// Each vertex influenced by up to 4 bones (standard)

attribute vec4 skinIndex;   // Which 4 bones affect this vertex
attribute vec4 skinWeight;  // How much each bone affects it (sum = 1.0)
uniform mat4 boneMatrices[MAX_BONES]; // Bone world transforms

vec4 skinVertex(vec4 position) {
  mat4 skinMatrix =
    boneMatrices[int(skinIndex.x)] * skinWeight.x +
    boneMatrices[int(skinIndex.y)] * skinWeight.y +
    boneMatrices[int(skinIndex.z)] * skinWeight.z +
    boneMatrices[int(skinIndex.w)] * skinWeight.w;
  return skinMatrix * position;
}

// Dual Quaternion Skinning (DQS) — eliminates volume collapse at joints
// Instead of blending matrices (which causes candy-wrapper artifacts),
// blend dual quaternions representing rigid transforms
```

#### Blend Shapes / Morph Targets
```typescript
// Per-vertex displacement vectors for facial expressions, damage states, etc.
// Final position = base + Σ(weight_i × delta_i)

interface MorphTarget {
  name: string;           // e.g., "smile", "blink_L", "jaw_open"
  deltas: Float32Array;   // Per-vertex XYZ offsets
}

// 52 ARKit-compatible facial blendshapes
const ARKIT_BLENDSHAPES = [
  'eyeBlinkLeft', 'eyeBlinkRight', 'eyeWideLeft', 'eyeWideRight',
  'eyeSquintLeft', 'eyeSquintRight', 'eyeLookUpLeft', 'eyeLookUpRight',
  'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight',
  'eyeLookOutLeft', 'eyeLookOutRight',
  'jawOpen', 'jawForward', 'jawLeft', 'jawRight',
  'mouthClose', 'mouthFunnel', 'mouthPucker', 'mouthLeft', 'mouthRight',
  'mouthSmileLeft', 'mouthSmileRight', 'mouthFrownLeft', 'mouthFrownRight',
  'mouthDimpleLeft', 'mouthDimpleRight', 'mouthStretchLeft', 'mouthStretchRight',
  'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
  'mouthPressLeft', 'mouthPressRight', 'mouthLowerDownLeft', 'mouthLowerDownRight',
  'mouthUpperUpLeft', 'mouthUpperUpRight',
  'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
  'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
  'noseSneerLeft', 'noseSneerRight', 'tongueOut',
];

// Drive blendshapes from MediaPipe face landmarks (real-time mocap)
function faceLandmarksToBlendshapes(landmarks: NormalizedLandmarkList): Record<string, number> {
  const weights: Record<string, number> = {};
  // Eye openness from landmark distances
  weights.eyeBlinkLeft = 1.0 - clamp(eyeOpenRatio(landmarks, 'left') * 2, 0, 1);
  weights.eyeBlinkRight = 1.0 - clamp(eyeOpenRatio(landmarks, 'right') * 2, 0, 1);
  // Jaw openness from chin-to-nose distance
  weights.jawOpen = clamp(jawOpenRatio(landmarks) * 1.5, 0, 1);
  // Smile from mouth corner positions
  weights.mouthSmileLeft = clamp(mouthCornerRaise(landmarks, 'left'), 0, 1);
  weights.mouthSmileRight = clamp(mouthCornerRaise(landmarks, 'right'), 0, 1);
  // ... 47 more blendshapes derived from landmark geometry
  return weights;
}
```

### 3.3 Auto-Rigging (AI-Powered)

#### UniRig Pipeline (SIGGRAPH 2025)
```
Input: Any 3D mesh (no topology requirements)

Stage 1 — Skeleton Prediction:
  GPT-like autoregressive transformer
  Input: Mesh vertices + normals (sampled)
  Output: Bone hierarchy as token sequence
  Format: (parent_idx, position_x, position_y, position_z, bone_name)
  Predicts bones one-by-one, each conditioned on all previous

Stage 2 — Skinning Weight Prediction:
  Bone-Point Cross Attention mechanism
  Input: Predicted skeleton + mesh vertices
  Output: Per-vertex skinning weights (4 bones per vertex)
  Uses attention between bone features and vertex features

Result: Fully rigged mesh ready for animation
Works on: humanoids, animals, robots, fantasy creatures, furniture, anything
```

#### Mixamo Auto-Rig (65-bone standard)
```
Upload mesh → Auto-detect body proportions → Generate skeleton → Assign weights
Access 2,500+ motion capture animations (free)
Export: FBX with skeleton + animations
Use in R3F: useGLTF + useAnimations hooks
```

---

## CHAPTER 4: RENDERING

### 4.1 PBR (Physically Based Rendering)

#### Material Model
```typescript
// OpenPBR Surface Model (industry standard 2025)
interface PBRMaterial {
  // Base layer
  baseColor: Color;           // Albedo / diffuse color
  metalness: number;          // 0 = dielectric, 1 = metal
  roughness: number;          // 0 = mirror, 1 = matte
  // Specular
  specularIntensity: number;  // F0 reflectance strength
  specularColor: Color;       // Tint specular for non-metals
  ior: number;                // Index of refraction (glass = 1.5, water = 1.33, diamond = 2.42)
  // Clear coat
  clearcoat: number;          // Second specular layer (car paint, lacquer)
  clearcoatRoughness: number;
  // Transmission
  transmission: number;       // 0 = opaque, 1 = fully transparent
  thickness: number;          // Volume thickness for refraction
  attenuationColor: Color;    // Color absorption through volume
  attenuationDistance: number;
  // Subsurface
  subsurface: number;         // Skin, wax, marble scattering
  subsurfaceColor: Color;
  // Sheen
  sheen: number;              // Fabric, velvet
  sheenRoughness: number;
  sheenColor: Color;
  // Emission
  emissive: Color;
  emissiveIntensity: number;
  // Iridescence
  iridescence: number;        // Soap bubbles, oil slicks, beetles
  iridescenceIOR: number;
  iridescenceThicknessRange: [number, number];
  // Anisotropy
  anisotropy: number;         // Brushed metal, hair
  anisotropyRotation: number;
}

// Texture Maps
interface PBRTextures {
  map: Texture;               // Base color (sRGB)
  normalMap: Texture;         // Surface detail (tangent space)
  roughnessMap: Texture;      // Per-pixel roughness (linear)
  metalnessMap: Texture;      // Per-pixel metalness (linear)
  aoMap: Texture;             // Ambient occlusion (linear)
  displacementMap: Texture;   // Height-based vertex displacement
  emissiveMap: Texture;       // Glow regions
  envMap: Texture;            // Environment reflections (HDR cubemap/equirect)
}
```

### 4.2 WebGPU Path Tracing
```wgsl
// Compute shader: physically accurate path tracing
@group(0) @binding(0) var<storage, read> scene: SceneData;
@group(0) @binding(1) var<storage, read_write> accumulator: array<vec4f>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;

struct Ray { origin: vec3f, direction: vec3f }
struct Hit { t: f32, normal: vec3f, materialId: u32 }

fn traceRay(ray: Ray) -> vec3f {
  var throughput = vec3f(1.0);
  var radiance = vec3f(0.0);
  var currentRay = ray;

  for (var bounce: u32 = 0; bounce < MAX_BOUNCES; bounce++) {
    let hit = intersectScene(currentRay);
    if (hit.t < 0.0) {
      radiance += throughput * sampleEnvironment(currentRay.direction);
      break;
    }

    let material = scene.materials[hit.materialId];
    let hitPos = currentRay.origin + currentRay.direction * hit.t;

    // Direct lighting (shadow ray to each light)
    for (var i: u32 = 0; i < scene.lightCount; i++) {
      let lightDir = normalize(scene.lights[i].position - hitPos);
      let shadowRay = Ray(hitPos + hit.normal * 0.001, lightDir);
      if (!intersectAny(shadowRay, distance(scene.lights[i].position, hitPos))) {
        let brdf = evaluateBRDF(material, hit.normal, -currentRay.direction, lightDir);
        let NdotL = max(dot(hit.normal, lightDir), 0.0);
        radiance += throughput * brdf * scene.lights[i].color * NdotL;
      }
    }

    // Indirect bounce (importance sample BRDF)
    let sample = sampleBRDF(material, hit.normal, -currentRay.direction);
    throughput *= sample.weight;
    currentRay = Ray(hitPos + hit.normal * 0.001, sample.direction);

    // Russian roulette termination
    let p = max(throughput.x, max(throughput.y, throughput.z));
    if (random() > p) break;
    throughput /= p;
  }
  return radiance;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let pixel = vec2f(f32(id.x), f32(id.y));
  let ray = generateCameraRay(pixel);
  let color = traceRay(ray);

  // Progressive accumulation
  let idx = id.y * WIDTH + id.x;
  let prevColor = accumulator[idx].xyz;
  let sampleCount = accumulator[idx].w + 1.0;
  let newColor = prevColor + (color - prevColor) / sampleCount;
  accumulator[idx] = vec4f(newColor, sampleCount);

  textureStore(output, vec2i(id.xy), vec4f(toneMap(newColor), 1.0));
}
```

### 4.3 Deferred Rendering Pipeline
```
Pass 1 — G-Buffer Generation:
  Render all geometry into multiple render targets:
  - RT0: Albedo (RGB) + Metalness (A)
  - RT1: Normal (RGB, encoded) + Roughness (A)
  - RT2: Emission (RGB) + AO (A)
  - RT3: Depth (32-bit float)
  Single geometry pass, no per-light overhead.

Pass 2 — Light Culling (Compute):
  Divide screen into 16x16 tiles.
  For each tile, test all lights against tile frustum.
  Build per-tile light list (supports 400+ lights at 60fps).

Pass 3 — Lighting (Compute/Fragment):
  For each pixel, read G-buffer data.
  Evaluate only lights in that pixel's tile.
  PBR BRDF evaluation per light.
  Add environment IBL (Image-Based Lighting).

Pass 4 — Post-Processing:
  SSAO → SSR → Bloom → DOF → Motion Blur → TAA → Tone Mapping → Color Grading
```

### 4.4 Screen-Space Effects

#### SSAO (Screen Space Ambient Occlusion)
```
Sample random hemisphere directions around each pixel's surface normal.
For each sample: compare depth against depth buffer.
If occluded: increment AO counter.
Result: soft contact shadows in crevices and corners.
Blur pass: bilateral filter preserves edges.
```

#### SSR (Screen Space Reflections)
```
For each pixel: calculate reflection vector from view direction + normal.
Ray march through depth buffer in reflection direction.
Hi-Z acceleration: use mip chain of depth buffer for large steps.
Fallback: environment map where SSR misses (off-screen reflections).
```

### 4.5 4K Rendering & Quality

#### Temporal Anti-Aliasing (TAA)
```
Per Frame:
1. Jitter camera sub-pixel using Halton(2,3) sequence
2. Render scene at full resolution
3. Reproject previous frame using motion vectors
4. Blend: current (10-20%) + reprojected history (80-90%)
5. Neighborhood clamping: clamp history to min/max of 3x3 current pixels
   (prevents ghosting on fast-moving objects)
6. Output: temporally stable, effectively supersampled image

Pipeline position: After deferred lighting, before tone mapping.
Operates on HDR values for correct blending.
```

#### HDR Pipeline
```
Render in HDR (float16 or float32 render targets)
→ Auto-exposure (luminance histogram, adaptation speed)
→ Bloom (threshold → downsample → blur → upsample chain)
→ Tone mapping (ACES Filmic: S-curve, preserves highlights/shadows)
→ Color grading (LUT-based or parametric: lift/gamma/gain)
→ Output: sRGB or HDR10 (for HDR displays, Rec.2020 color space)
```

#### Resolution Scaling
```typescript
// Adaptive DPR (Device Pixel Ratio) for consistent 60fps
// R3F: <Canvas dpr={[1, 2]}> + <AdaptiveDpr pixelated />
// Automatically reduces resolution when GPU-bound, restores when headroom available

// Manual upscaling for 4K output:
// Render at lower internal resolution (1440p)
// TAA + sharpening filter = visually indistinguishable from native 4K
// Performance: 2-3x faster than native 4K
```

---

## CHAPTER 5: ANIMATION

### 5.1 Animation State Machine
```typescript
// Hierarchical state machine for character animation
interface AnimationState {
  name: string;
  clip: AnimationClip;
  speed: number;
  loop: boolean;
  transitions: Transition[];
  blendTree?: BlendTree; // Optional blend tree instead of single clip
}

interface Transition {
  to: string;
  condition: () => boolean;
  duration: number;        // Crossfade duration in seconds
  hasExitTime: boolean;    // Wait for clip to finish?
  exitTime: number;        // Normalized time to exit (0-1)
}

interface BlendTree {
  type: '1D' | '2D_SimpleDirectional' | '2D_FreeformCartesian';
  parameter: string;       // e.g., "speed" or ["moveX", "moveY"]
  children: { clip: AnimationClip; threshold: number | [number, number]; }[];
}

class AnimationStateMachine {
  currentState: AnimationState;
  mixer: AnimationMixer;

  update(dt: number, params: Record<string, number>) {
    // Check transitions
    for (const t of this.currentState.transitions) {
      if (t.condition()) {
        this.transition(t);
        break;
      }
    }

    // Update blend tree weights if active
    if (this.currentState.blendTree) {
      this.updateBlendTree(this.currentState.blendTree, params);
    }

    this.mixer.update(dt);
  }

  transition(t: Transition) {
    const fromAction = this.mixer.existingAction(this.currentState.clip);
    const toState = this.states[t.to];
    const toAction = this.mixer.clipAction(toState.clip);

    // Crossfade
    toAction.reset().setEffectiveTimeScale(toState.speed).setEffectiveWeight(1).play();
    fromAction?.crossFadeTo(toAction, t.duration, true);

    this.currentState = toState;
  }
}

// Example: Character locomotion
const locomotionStates: AnimationState[] = [
  {
    name: 'idle',
    clip: idleClip,
    speed: 1,
    loop: true,
    transitions: [
      { to: 'walk', condition: () => speed > 0.1, duration: 0.2, hasExitTime: false, exitTime: 0 },
      { to: 'jump', condition: () => jumpPressed, duration: 0.1, hasExitTime: false, exitTime: 0 },
    ],
  },
  {
    name: 'walk',
    clip: walkClip,
    speed: 1,
    loop: true,
    blendTree: {
      type: '1D',
      parameter: 'speed',
      children: [
        { clip: walkClip, threshold: 1 },
        { clip: jogClip, threshold: 3 },
        { clip: runClip, threshold: 6 },
        { clip: sprintClip, threshold: 10 },
      ],
    },
    transitions: [
      { to: 'idle', condition: () => speed < 0.1, duration: 0.3, hasExitTime: false, exitTime: 0 },
      { to: 'jump', condition: () => jumpPressed, duration: 0.1, hasExitTime: false, exitTime: 0 },
    ],
  },
];
```

### 5.2 Procedural Animation
```typescript
// Procedural foot placement using IK
function proceduralWalk(character: Character, terrain: Terrain, dt: number) {
  const hipHeight = 1.0;
  const stepLength = 0.6;
  const stepHeight = 0.15;
  const stepSpeed = 3.0;

  // Raycast from hip to find ground
  for (const foot of ['left', 'right']) {
    const hipPos = character.getBoneWorldPosition(foot === 'left' ? 'LeftUpLeg' : 'RightUpLeg');
    const groundHit = terrain.raycast(hipPos, [0, -1, 0], hipHeight * 2);

    if (groundHit) {
      const targetPos = groundHit.point;
      targetPos.y += 0.05; // Slight offset above ground

      // Smooth step motion with sine curve for height
      const phase = (character.walkPhase + (foot === 'right' ? 0.5 : 0)) % 1;
      if (phase < 0.5) {
        // Swing phase: foot in air
        const t = phase * 2; // 0-1 during swing
        targetPos.y += Math.sin(t * Math.PI) * stepHeight;
        targetPos.x += Math.sin(t * Math.PI) * stepLength * character.moveDirection.x;
        targetPos.z += Math.sin(t * Math.PI) * stepLength * character.moveDirection.z;
      }

      // Solve IK to place foot
      const chain = character.getIKChain(foot === 'left' ? 'LeftFoot' : 'RightFoot');
      solveFABRIK(chain, targetPos, character.legBoneLengths);
    }
  }

  character.walkPhase = (character.walkPhase + dt * stepSpeed) % 1;
}

// Procedural look-at (head/eye tracking)
function proceduralLookAt(character: Character, target: Vec3) {
  const headBone = character.getBone('Head');
  const neckBone = character.getBone('Neck');

  const headPos = getWorldPosition(headBone);
  const direction = normalize(sub(target, headPos));

  // Split rotation between neck (40%) and head (60%)
  const fullRotation = quaternionLookAt(direction, [0, 1, 0]);
  neckBone.rotation = slerp(neckBone.rotation, fullRotation, 0.4);
  headBone.rotation = slerp(headBone.rotation, fullRotation, 0.6);

  // Clamp to natural range (prevent exorcist-spin)
  clampBoneRotation(neckBone, { yaw: [-45, 45], pitch: [-20, 20] });
  clampBoneRotation(headBone, { yaw: [-30, 30], pitch: [-30, 25] });
}
```

### 5.3 Motion Capture Retargeting
```typescript
// MediaPipe → Skeleton retargeting pipeline
import { PoseLandmarker, FaceLandmarker, HandLandmarker } from '@mediapipe/tasks-vision';

class MotionCaptureSystem {
  poseLandmarker: PoseLandmarker;   // 33 body landmarks
  faceLandmarker: FaceLandmarker;   // 468 face landmarks + 52 blendshapes
  handLandmarker: HandLandmarker;   // 21 landmarks per hand

  // Real-time retargeting: video frame → skeleton pose
  retarget(frame: VideoFrame, targetSkeleton: Skeleton): Pose {
    const bodyLandmarks = this.poseLandmarker.detect(frame);
    const faceLandmarks = this.faceLandmarker.detect(frame);
    const handLandmarks = this.handLandmarker.detect(frame);

    // Convert landmarks to bone rotations
    const pose: Pose = {};

    // Spine: from hip-center to shoulder-center direction
    const hipCenter = midpoint(bodyLandmarks[23], bodyLandmarks[24]);
    const shoulderCenter = midpoint(bodyLandmarks[11], bodyLandmarks[12]);
    pose['Spine'] = rotationFromDirection(sub(shoulderCenter, hipCenter));

    // Arms: shoulder → elbow → wrist chain
    pose['LeftArm'] = rotationBetween(bodyLandmarks[11], bodyLandmarks[13]);
    pose['LeftForeArm'] = rotationBetween(bodyLandmarks[13], bodyLandmarks[15]);
    pose['RightArm'] = rotationBetween(bodyLandmarks[12], bodyLandmarks[14]);
    pose['RightForeArm'] = rotationBetween(bodyLandmarks[14], bodyLandmarks[16]);

    // Legs: hip → knee → ankle chain
    pose['LeftUpLeg'] = rotationBetween(bodyLandmarks[23], bodyLandmarks[25]);
    pose['LeftLeg'] = rotationBetween(bodyLandmarks[25], bodyLandmarks[27]);
    pose['RightUpLeg'] = rotationBetween(bodyLandmarks[24], bodyLandmarks[26]);
    pose['RightLeg'] = rotationBetween(bodyLandmarks[26], bodyLandmarks[28]);

    // Face: 52 ARKit blendshapes
    pose['__blendshapes'] = faceLandmarksToBlendshapes(faceLandmarks);

    // Hands: 21 landmarks → finger joint rotations
    if (handLandmarks.left) pose['__leftHand'] = handToFingerRotations(handLandmarks.left);
    if (handLandmarks.right) pose['__rightHand'] = handToFingerRotations(handLandmarks.right);

    // Scale normalization: map source proportions to target skeleton
    return normalizeToSkeleton(pose, targetSkeleton);
  }
}
```

---

## CHAPTER 6: PHYSICS SIMULATION

### 6.1 Rigid Body
```typescript
// Rapier WASM integration with R3F
import { RigidBody, CuboidCollider, BallCollider, TrimeshCollider } from '@react-three/rapier';

// Dynamic body (affected by forces)
<RigidBody type="dynamic" mass={1} restitution={0.5} friction={0.8}>
  <mesh><boxGeometry /><meshStandardMaterial /></mesh>
  <CuboidCollider args={[0.5, 0.5, 0.5]} />
</RigidBody>

// Kinematic body (script-controlled, affects others)
<RigidBody type="kinematicPosition">
  <mesh ref={platformRef}><boxGeometry args={[4, 0.2, 4]} /></mesh>
</RigidBody>

// Trimesh collider (exact mesh shape for terrain)
<RigidBody type="fixed">
  <mesh geometry={terrainGeometry}><meshStandardMaterial /></mesh>
  <TrimeshCollider args={[terrainVertices, terrainIndices]} />
</RigidBody>
```

### 6.2 Soft Body / Cloth
```typescript
// Cloth simulation: mass-spring system on GPU
// Compute shader updates particle positions using Verlet integration

struct ClothParticle {
  position: vec3f,
  previousPosition: vec3f,
  acceleration: vec3f,
  mass: f32,
  pinned: u32,  // 0 = free, 1 = pinned to world
}

// Verlet integration step
fn updateParticle(p: ptr<storage, ClothParticle>, dt: f32, gravity: vec3f) {
  if ((*p).pinned == 1u) { return; }
  let velocity = (*p).position - (*p).previousPosition;
  (*p).previousPosition = (*p).position;
  (*p).position += velocity * DAMPING + (gravity + (*p).acceleration) * dt * dt;
  (*p).acceleration = vec3f(0.0);
}

// Distance constraint (maintain spring length)
fn constrainDistance(a: ptr<storage, ClothParticle>, b: ptr<storage, ClothParticle>, restLength: f32) {
  let delta = (*b).position - (*a).position;
  let currentLength = length(delta);
  let correction = delta * (1.0 - restLength / currentLength) * 0.5;
  if ((*a).pinned == 0u) { (*a).position += correction; }
  if ((*b).pinned == 0u) { (*b).position -= correction; }
}

// Run 10-20 constraint iterations per frame for stability
```

### 6.3 Fluid Simulation (SPH on GPU)
```wgsl
// Smoothed Particle Hydrodynamics — compute shader
// Each particle: position, velocity, density, pressure

@compute @workgroup_size(64)
fn computeDensity(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  var density: f32 = 0.0;
  let pos = particles[i].position;

  // Sum contributions from all neighbors within smoothing radius
  for (var j: u32 = 0; j < particleCount; j++) {
    let r = distance(pos, particles[j].position);
    if (r < SMOOTHING_RADIUS) {
      density += PARTICLE_MASS * poly6Kernel(r, SMOOTHING_RADIUS);
    }
  }
  particles[i].density = density;
  particles[i].pressure = GAS_CONSTANT * (density - REST_DENSITY);
}

@compute @workgroup_size(64)
fn computeForces(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  var pressureForce = vec3f(0.0);
  var viscosityForce = vec3f(0.0);

  for (var j: u32 = 0; j < particleCount; j++) {
    if (i == j) { continue; }
    let r = distance(particles[i].position, particles[j].position);
    if (r < SMOOTHING_RADIUS) {
      let dir = normalize(particles[j].position - particles[i].position);
      // Pressure force
      pressureForce -= dir * PARTICLE_MASS *
        (particles[i].pressure + particles[j].pressure) / (2.0 * particles[j].density) *
        spikyGradient(r, SMOOTHING_RADIUS);
      // Viscosity force
      viscosityForce += VISCOSITY * PARTICLE_MASS *
        (particles[j].velocity - particles[i].velocity) / particles[j].density *
        viscosityLaplacian(r, SMOOTHING_RADIUS);
    }
  }

  particles[i].acceleration = (pressureForce + viscosityForce) / particles[i].density + GRAVITY;
}
```

### 6.4 Hair/Fur Simulation
```
Modern approach (2025): Neural hair simulation

HairFormer (SIGGRAPH 2025 — Transformer-based):
  Stage 1: Static network predicts draped shape from hairstyle + body + pose
  Stage 2: Dynamic network generates motion (flying hairs, secondary bounce)
  Supports: 10,000-80,000 individual strands
  Real-time at 30fps on modern GPU

Fallback (traditional): Position-Based Dynamics (PBD)
  - Each hair strand = chain of particles with distance + bending constraints
  - Collide against simplified head/body mesh
  - Wind: apply force field with turbulence noise
  - Rendering: line primitives or tube geometry, screen-space AA
```

### 6.5 Destruction Physics
```typescript
// Pre-fracture mesh using Voronoi decomposition
// At runtime: replace intact mesh with fragments + apply impulse

function fractureOnImpact(mesh: Mesh, impactPoint: Vec3, impactForce: number) {
  // Generate Voronoi cells around impact point
  const seeds = generateVoronoiSeeds(impactPoint, impactForce, 15); // 15 fragments
  const fragments = voronoiFracture(mesh.geometry, seeds);

  // Replace mesh with rigid body fragments
  for (const fragment of fragments) {
    const body = world.createRigidBody(RigidBodyDesc.dynamic());
    const collider = world.createCollider(
      ColliderDesc.trimesh(fragment.vertices, fragment.indices),
      body
    );

    // Apply radial impulse from impact point
    const direction = normalize(sub(fragment.center, impactPoint));
    const force = scale(direction, impactForce * (1 / distance(fragment.center, impactPoint)));
    body.applyImpulse(force, true);

    // Add angular velocity for tumbling
    body.applyTorqueImpulse(randomVec3(-2, 2), true);
  }

  // Particle effects: dust, sparks, debris
  emitParticles('debris', impactPoint, impactForce);
  emitParticles('dust', impactPoint, impactForce * 0.3);
}
```

---

## CHAPTER 7: MESH OPERATIONS

### 7.1 Subdivision (Loop / Catmull-Clark)
```
Loop Subdivision (triangles):
  1. Edge midpoints: weighted average of edge vertices + adjacent face vertices
  2. Original vertices: repositioned as weighted combination of neighbors
  3. Each triangle → 4 triangles
  Result: smooth, organic surfaces from low-poly cage

Catmull-Clark Subdivision (quads):
  1. Face points: average of face vertices
  2. Edge points: average of edge midpoints + adjacent face points
  3. Vertex points: weighted combination (face_avg + 2*edge_avg + (n-3)*vertex) / n
  4. Each face → 4 quads
  Result: industry-standard smooth surfaces (used in Pixar, Disney)
```

### 7.2 Decimation (Mesh Simplification)
```typescript
// Quadric Error Metrics (Garland-Heckbert) — optimal edge collapse order
// For each vertex: compute error quadric Q = Σ(plane equations of adjacent faces)
// For each edge: collapse cost = v^T * (Q1 + Q2) * v (optimal position minimizes error)
// Priority queue: always collapse cheapest edge first

function decimateMesh(geometry: BufferGeometry, targetRatio: number): BufferGeometry {
  const targetFaces = Math.floor(geometry.index.count / 3 * targetRatio);
  // Build half-edge data structure for efficient traversal
  const halfEdge = buildHalfEdgeMesh(geometry);
  // Compute quadric for each vertex
  const quadrics = computeQuadrics(halfEdge);
  // Priority queue of edge collapses sorted by cost
  const queue = buildCollapseQueue(halfEdge, quadrics);

  while (halfEdge.faceCount > targetFaces) {
    const collapse = queue.pop(); // Lowest cost
    if (!isValidCollapse(collapse, halfEdge)) continue; // Skip if topology changed
    performCollapse(halfEdge, collapse, quadrics);
    updateNeighborCosts(halfEdge, collapse.vertex, quadrics, queue);
  }

  return halfEdgeToBufferGeometry(halfEdge);
}
```

### 7.3 LOD Generation
```typescript
// Automatic Level of Detail system
interface LODLevel {
  distance: number;       // Camera distance threshold
  geometry: BufferGeometry;
  triangleRatio: number;  // Percentage of original triangles
}

function generateLODs(geometry: BufferGeometry): LODLevel[] {
  return [
    { distance: 0,   geometry: geometry,                          triangleRatio: 1.0 },
    { distance: 20,  geometry: decimateMesh(geometry, 0.5),       triangleRatio: 0.5 },
    { distance: 50,  geometry: decimateMesh(geometry, 0.25),      triangleRatio: 0.25 },
    { distance: 100, geometry: decimateMesh(geometry, 0.1),       triangleRatio: 0.1 },
    { distance: 200, geometry: generateBillboard(geometry),       triangleRatio: 0.001 }, // Impostor
  ];
}

// R3F LOD component
import { Detailed } from '@react-three/drei';

function LODModel({ model }) {
  const lods = generateLODs(model.geometry);
  return (
    <Detailed distances={lods.map(l => l.distance)}>
      {lods.map((lod, i) => (
        <mesh key={i} geometry={lod.geometry} material={model.material} />
      ))}
    </Detailed>
  );
}
```

### 7.4 UV Unwrapping
```
Automatic UV Unwrapping Algorithms:

1. Box Projection: Project from 6 directions, blend at seams
   Best for: architectural geometry, boxes, buildings

2. Cylindrical/Spherical: Wrap UVs around axis
   Best for: characters, bottles, organic tubular shapes

3. LSCM (Least Squares Conformal Mapping): Minimize angle distortion
   Best for: complex organic meshes, characters

4. ABF++ (Angle-Based Flattening): Minimize angle deviation from 3D
   Best for: high-quality character/prop UVs

5. PartUV (2025): Part-based neural unwrapping
   Generates UV mappings with aligned charts per semantic part
   Processing: seconds on modern GPU
```

### 7.5 Normal Map Baking
```
High-poly to low-poly normal transfer:

1. Prepare: High-poly mesh (millions of tris) + Low-poly mesh (thousands)
2. For each texel on low-poly UV:
   a. Find corresponding surface point on low-poly
   b. Cast ray along low-poly normal toward high-poly
   c. Record hit normal from high-poly surface
   d. Transform to tangent space of low-poly face
   e. Encode: RGB = (normal * 0.5 + 0.5) * 255
3. Result: Texture that makes low-poly render like high-poly

GPU implementation: render high-poly from each low-poly face's tangent-space view
```

---

## CHAPTER 8: 2D RENDERING

### 8.1 Canvas 2D (High Performance)
```typescript
// OffscreenCanvas + Web Worker for 60fps 2D
const worker = new Worker('canvas-worker.js');
const offscreen = canvas.transferControlToOffscreen();
worker.postMessage({ canvas: offscreen }, [offscreen]);

// In worker:
const ctx = canvas.getContext('2d');

function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Layer system
  for (const layer of layers) {
    ctx.save();
    ctx.globalAlpha = layer.opacity;
    ctx.globalCompositeOperation = layer.blendMode;
    ctx.transform(...layer.transform);

    for (const shape of layer.shapes) {
      drawShape(ctx, shape);
    }
    ctx.restore();
  }

  requestAnimationFrame(render);
}
```

### 8.2 PixiJS (GPU-Accelerated 2D)
```typescript
import { Application, Sprite, Container, Graphics, AnimatedSprite } from 'pixi.js';

const app = new Application();
await app.init({ width: 1920, height: 1080, antialias: true, resolution: 2 });

// Sprite batching: thousands of sprites at 60fps
const container = new Container();
for (let i = 0; i < 10000; i++) {
  const sprite = Sprite.from('particle.png');
  sprite.anchor.set(0.5);
  sprite.position.set(Math.random() * 1920, Math.random() * 1080);
  container.addChild(sprite);
}
app.stage.addChild(container);

// Skeletal 2D animation (Spine runtime)
import { Spine } from '@pixi-spine/all-4.1';
const spineAnimation = new Spine(spineData);
spineAnimation.state.setAnimation(0, 'walk', true);
```

### 8.3 SVG Animation
```typescript
// GSAP + SVG for vector animation
import gsap from 'gsap';
import { DrawSVGPlugin, MorphSVGPlugin, MotionPathPlugin } from 'gsap/all';
gsap.registerPlugin(DrawSVGPlugin, MorphSVGPlugin, MotionPathPlugin);

// Line drawing animation
gsap.fromTo('#path', { drawSVG: '0%' }, { drawSVG: '100%', duration: 2, ease: 'power2.inOut' });

// Shape morphing
gsap.to('#shape1', { morphSVG: '#shape2', duration: 1.5, ease: 'elastic.out(1, 0.5)' });

// Motion path
gsap.to('#element', {
  motionPath: { path: '#curvePath', align: '#curvePath', autoRotate: true },
  duration: 3, ease: 'none', repeat: -1,
});
```

---

## CHAPTER 9: PARTICLE SYSTEMS

### 9.1 GPU Particles (WebGPU Compute)
```wgsl
// 1 million+ particles at 60fps using compute shaders
struct Particle {
  position: vec3f,
  velocity: vec3f,
  color: vec4f,
  life: f32,
  maxLife: f32,
  size: f32,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn updateParticles(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.count) { return; }

  var p = particles[i];
  p.life -= params.dt;

  if (p.life <= 0.0) {
    // Respawn
    p.position = params.emitterPosition + randomSphere(i) * params.emitterRadius;
    p.velocity = params.emitDirection + randomSphere(i + 1000u) * params.spread;
    p.life = params.minLife + random(i + 2000u) * (params.maxLife - params.minLife);
    p.maxLife = p.life;
    p.size = params.startSize;
    p.color = params.startColor;
  } else {
    // Physics
    p.velocity += params.gravity * params.dt;                    // Gravity
    p.velocity += curl3D(p.position * 0.01, params.time) * params.turbulence; // Turbulence
    p.velocity *= 1.0 - params.drag * params.dt;                // Drag

    // Force fields (attractor/repulsor)
    for (var f: u32 = 0; f < params.forceFieldCount; f++) {
      let ff = forceFields[f];
      let dir = ff.position - p.position;
      let dist = length(dir);
      if (dist < ff.radius) {
        p.velocity += normalize(dir) * ff.strength * (1.0 - dist / ff.radius) * params.dt;
      }
    }

    p.position += p.velocity * params.dt;

    // Age-based interpolation
    let t = 1.0 - p.life / p.maxLife;
    p.size = mix(params.startSize, params.endSize, t);
    p.color = mix(params.startColor, params.endColor, t);
    p.color.a *= smoothstep(0.0, 0.1, p.life / p.maxLife); // Fade out
  }

  particles[i] = p;
}
```

### 9.2 Curl Noise (Natural Turbulence)
```glsl
// Incompressible flow field — particles never cluster or diverge
vec3 curl3D(vec3 p, float time) {
  float e = 0.01;
  vec3 dx = vec3(e, 0, 0), dy = vec3(0, e, 0), dz = vec3(0, 0, e);

  float px = snoise(p + dy + time) - snoise(p - dy + time);
  float py = snoise(p + dz + time) - snoise(p - dz + time);
  float pz = snoise(p + dx + time) - snoise(p - dx + time);

  // Cross-differences give divergence-free field
  return vec3(
    (snoise(p + dy) - snoise(p - dy)) - (snoise(p + dz) - snoise(p - dz)),
    (snoise(p + dz) - snoise(p - dz)) - (snoise(p + dx) - snoise(p - dx)),
    (snoise(p + dx) - snoise(p - dx)) - (snoise(p + dy) - snoise(p - dy))
  ) / (2.0 * e);
}
```

---

## CHAPTER 10: PERFORMANCE RULES

### Hard Limits (60fps @ 1080p)
```
Draw calls:        < 100 (use instancing, merging, batching)
Triangles:         < 2M visible (use LOD, culling, occlusion)
Texture memory:    < 256MB VRAM (use KTX2, mipmaps, streaming)
Shader complexity: < 128 ALU per fragment (profile with SpectorJS)
Physics bodies:    < 1000 active (sleep inactive, use layers)
Particles:         Use GPU compute for > 10K (CPU caps at ~50K)
Lights:            < 4 shadow-casting (use deferred for 100+)
Post-processing:   Budget 4ms total (skip DOF on mobile)
```

### Optimization Checklist
```
[ ] Draco compression on all .glb models
[ ] KTX2 compression on all textures (70% size reduction)
[ ] LOD system for everything > 10K triangles
[ ] Instanced rendering for repeated objects (trees, rocks, grass)
[ ] Frustum culling (automatic in Three.js)
[ ] Occlusion culling for indoor scenes
[ ] Object pooling for spawned/destroyed objects
[ ] Adaptive DPR (<AdaptiveDpr /> in R3F)
[ ] Web Workers for heavy computation (pathfinding, AI, generation)
[ ] Dispose geometry/materials/textures when removing objects
[ ] Texture atlasing for many small textures
[ ] Geometry merging for static objects
[ ] Baked lighting for static scenes (lightmaps)
```

---

## CHAPTER 11: EXPORT & INTEGRATION

### Export Formats
```
3D Scene:    .glb (single binary, universal), .gltf (JSON + separate assets)
Animation:   .fbx (Mixamo compat), .bvh (motion capture)
Texture:     .ktx2 (GPU compressed), .exr (HDR), .png/.webp (standard)
Video:       Remotion (.mp4, .webm) — render 3D scenes to video
Image:       Canvas .toDataURL('image/png') — 4K screenshot
Diagram:     .excalidraw — architecture/flow diagrams
Splat:       .splat / .ply — Gaussian Splatting scenes
SDF:         .sdf — Signed Distance Field volumes
```

### Pipeline: Concept → Ship
```
1. CONCEPT:  User describes what they want (text/sketch/reference)
2. GENERATE: AI-assisted modeling OR procedural generation
3. REFINE:   Mesh operations (subdivide, decimate, UV unwrap)
4. RIG:      Auto-rig (UniRig/Mixamo) OR manual bone placement
5. TEXTURE:  PBR material setup, texture painting, normal baking
6. ANIMATE:  State machine + blend trees + procedural + mocap
7. SIMULATE: Physics, particles, cloth, fluid, destruction
8. LIGHT:    Environment, shadows, post-processing, tone mapping
9. OPTIMIZE: LOD, compression, instancing, performance profiling
10. EXPORT:  .glb for web, .fbx for engines, .mp4 for video
```

---

## CHAPTER 12: SKILL INTEGRATION MAP

This skill builds on and orchestrates these existing MGR skills:

```
mgr-visual-forge (THIS SKILL — master orchestrator)
├── mgr-3d-expert (R3F fundamentals, materials, lighting)
├── mgr-3d-world-builder (AI generation, world building)
├── mgr-vfx-pipeline (particles, destruction, mocap, crowd sim)
├── mgr-game-builder (ECS, physics, multiplayer, game loop)
├── mgr-video-editor-builder (FFmpeg, timeline, export)
├── mgr-remotion (programmatic video from 3D scenes)
├── mgr-excalidraw (technical diagrams of 3D pipelines)
├── mgr-frontend-pro (CSS animations, View Transitions for UI)
├── mgr-uiux-pro-max (3D editor UI/UX, controls, accessibility)
├── mgr-performance (optimization, profiling, budgeting)
└── mgr-ui-expert (component architecture for 3D tools)
```

When the user asks to build something visual, this skill auto-detects the domain and loads relevant sub-skills. It handles the FULL pipeline from empty canvas to exported, optimized, production-ready visual output.

---

**MGR VISUAL FORGE v1.0 — Money Grind Religion Inc.**
**Created by Timebeunus Boyd**
**Model. Rig. Render. Animate. Simulate. Ship.**
