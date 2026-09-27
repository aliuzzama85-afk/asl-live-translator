# Rendering Upgrade Plan (v1): Realistic 3D Hand Rendering

Scope: replace `SkeletonCanvas.jsx`'s flat 2D stick figure with a lit, 3D
solid-geometry hand and arm, rendered with Three.js (already installed and
unused), using the real `z` depth that is in the pose data but has been
thrown away until now. The main motivation is a documented limitation:
closed handshapes (A, O) are ambiguous in the flat rendering
(`PROJECT_STATUS.md` Section 14).

**This is a rendering-layer change only.** Nothing that produces the
`Timeline` changes: `usePoseSequences`, `reconstructTimeline`,
`stitchTimelines`, fingerspelling, multi-word queueing, and live speech are
untouched, and `SkeletonCanvas`'s props (`landmarkNames`, `timeline`,
`isPlaying`, `loop`, `reducedMotion`, `onEnded`, `onFrameChange`) stay
exactly the same. Nothing outside the component changes.

Baseline checked directly before writing this (not assumed): `CLAUDE.md`,
`PROJECT_STATUS.md`, `frontend/PLAN.md` (Sections 1 and "Camera fit"),
`SkeletonCanvas.jsx`, `skeletonBones.js` and its tests,
`reconstructTimeline.js`, `stitchTimelines.js`, `tokens.css`, and
`package.json`/`package-lock.json`. The depth data was checked on real
files: WLASL `about`, `angry`, `bathroom` and fingerspelling `a`, `o`, `b`,
`j` (Section 2).

---

## 1. Dependency

`three` is already a dependency: `^0.166.1` in `package.json`, resolved to
exactly `0.166.1` in `package-lock.json`, with **no dependencies of its own**,
and imported nowhere yet. Nothing new is installed. (`npm audit` is re-run at
the end anyway, as it is after every change here.)

---

## 2. The depth data: what's usable and what isn't (checked, not assumed)

Each landmark is `[x, y, z]`: `x`/`y` normalized to the image, `z` per
MediaPipe's convention (smaller = closer to the camera). Measured on real
files:

| | Hand landmarks `z` | Hand width in `x` | Pose-subset (shoulder/elbow/wrist) `z` |
|---|---|---|---|
| `about` (WLASL) | −0.071 … 0.005 | 0.06–0.35 | −1.67 … −0.14 |
| `angry` (WLASL) | −0.104 … 0.041 | 0.12–0.75 | −2.02 … −0.11 |
| `a` (dataset letter) | −0.061 … 0.000 | 0.22 | — |
| `o` (dataset letter) | −0.043 … 0.021 | 0.14 | — |
| `j` (self-recorded) | −0.224 … 0.000 | 0.13–0.25 | −2.50 … −0.31 |

- **Hand `z` is real, usable depth**: relative to that hand's own wrist
  (≈ 0 at the wrist), and on the same scale as `x`, as MediaPipe documents
  ("roughly the same scale as x"). A curled fingertip sits a few hundredths
  in front of the palm, which is exactly the information the flat view
  throws away. It's used as is (`DEPTH_SCALE = 1`), with no exaggeration, so
  the shape stays true to the data.
- **Pose-subset `z` is not usable alongside it**: it comes from MediaPipe's
  *pose* model, measured from the hips and in a different frame, and it's
  10–30× larger. Mixing the two would put an elbow up to 2.5 units behind a
  wrist whose hand spans 0.1. So the arms are drawn **in the wrist plane
  (`z = 0`)**, as context, while each hand carries its real depth. They
  meet at the wrist, where the hand's own `z` is ≈ 0.
- `x` and `y` keep the existing square-canvas simplification
  (`frontend/PLAN.md`, "Canvas sizing"); this pass doesn't change it.

**World axes** (Three.js is y-up, right-handed): `X = x`, `Y = −y` (image y
points down), `Z = −z` (so "closer to the camera" means larger `Z`, toward
the viewer).

---

## 3. Scene setup

### Camera: perspective, slightly off-axis
- **`PerspectiveCamera`, 30° vertical field of view.** That's enough
  perspective for depth to read (near fingertips look nearer) without the
  distortion of a wide lens.
- **Viewed from 20° to the side and 12° above**, not dead-on. A pure front
  view keeps the image's own viewpoint, and so it keeps the ambiguity: a
  curled finger pointing straight at the camera looks the same as a short
  straight one. Turning a little shows the curl directly while the hand
  stays recognizably "facing you". This is a fixed viewing angle, not a
  moving orbit, so signs don't swim around.
- **Framing reuses the existing soft-follow camera-fit** (Section 4).

### Lighting: form from shading, in the existing palette
Lights are attached to the camera, so the lighting stays the same as the
camera follows the hand.
- **Hemisphere light** (sky `--color-skeleton-bone` #F5F5F0, ground
  `--color-panel` #141417): soft base light, brighter from above.
- **Key light**: a white directional light from upper front-left. This is
  what shows the form: the tops of curled fingers catch light, and their
  undersides fall into shade.
- **Rim light**: a dim directional light in `--color-accent-amber` (#FFC857)
  from behind, giving the silhouette a warm edge against the near-black
  background. That's cheap polish in the design system's own accent color.
- **No shadow maps.** A hand floating in empty space has nothing nearby to
  cast a shadow onto, so shadow maps would cost GPU time for no visible gain.

### Materials: the existing token colors, now lit
- **Bones**: `MeshStandardMaterial` in `--color-skeleton-bone` (#F5F5F0),
  roughness 0.45: a soft, matte off-white.
- **Joints**: `MeshStandardMaterial` in `--color-skeleton-joint` (#FFC857,
  amber), roughness 0.35. This is the same bone/joint color split the 2D view
  uses, so it reads as the same avatar, now solid.
- **Palm**: a translucent fill (bone color, 35% opacity) spanning the wrist
  and the five knuckle bases. It gives the hand a readable surface and makes
  palm orientation (toward you / away / sideways) obvious, which a line
  drawing can't show.
- **Arms**: bone color at 35% opacity. They're context, not the sign, just as
  the 2D view draws its arm "bridge" lines dimmer.
- **Background**: `--color-bg`, read from `tokens.css` at runtime, the same
  way the 2D renderer reads its colors. Tokens stay the single source.
- `WebGLRenderer` with **anti-aliasing**, the device pixel ratio for
  sharpness, and no tone mapping, so token colors render as specified.
- **"Held" (dimmed) frames** darken the materials to 50%, the same visual cue
  as the 2D view's half-alpha.

### Geometry: solid, tapered, proportioned from the data itself
- **Bones are tapered cylinders**, one per bone in `buildSkeletonTopology`,
  narrowing toward the fingertip (tip radius 80% of base). **Joints are
  spheres**, a little wider than the bones they join, so every segment reads
  as a rounded capsule. Drawn with Three.js `InstancedMesh` (one draw for all
  bones, one for all joints) and updated in place every frame. About 100
  instances is trivial for any GPU.
- **Proportions come from the landmarks**, since segment lengths are the real
  joint positions, so finger segments naturally get shorter toward the tip.
  **Thickness scales with each hand's own palm length** (wrist → middle
  knuckle), so a hand filmed small or large looks equally solid. Finger
  radius is about 9% of palm length, tapering to about 7% at the tip. Joint
  radius is about 10–12% (a larger wrist and knuckles, smaller fingertips).
  Those are typical human hand ratios, not data from anywhere else.
- The palm's knuckle-to-knuckle lines are drawn thinner, since the palm fill
  now carries that edge.

---

## 4. Camera fit in 3D: the existing logic, extended

`skeletonBones.js`'s soft-follow camera (`computeContentBounds` →
`computeFitTransform` → `stepCamera`, including the "hold steady on a
frame with no tracked hand" rule) is **extended to 3D, not replaced**:

- `computeContentBounds3D(poses)`: the same walk over real (non-sentinel)
  points, also tracking `minZ`/`maxZ`. The existing 2D `computeContentBounds`
  now derives from it (dropping `z`), so there's one implementation of "which
  points count".
- `computeFitTransform3D(bounds)` → `{centerX, centerY, centerZ, span}`,
  where `span` covers the largest of the three extents plus the same
  `FIT_PADDING_FRACTION`.
- `stepCamera3D(camera, pose, isPoseSubsetByIndex, alpha)`: the same
  hand-only fit, the same exponential easing, and **the same hold-steady
  rule on a frame with no real hand points**, now also easing `centerZ`.
  `stepCamera` and `stepCamera3D` share one easing helper. Reduced motion
  still passes `alpha = 1`, so it snaps instantly, exactly as today.
- `cameraPlacement3D(camera, view)` → the perspective camera's world
  position and look-at target: aim at the eased center, from the fixed
  20°/12° direction, at the distance where `span` fills the view
  (`distance = (span / 2) / tan(fov / 2)`).

The existing 2D functions and their tests are unchanged. The existing tests'
`toEqual` checks on 2D shapes pin that down.

---

## 5. The phantom-hand fix (bonus)

Documented limitation (`PROJECT_STATUS.md` Section 12): a hand that isn't
tracked for a whole word (the `(0, 0, 0)` sentinel on all 21 points) was
drawn as a static cluster at the projected origin, and popped in when
tracking began. In 3D, **any joint at the sentinel is hidden, and so is every
bone and palm triangle touching one**. An untracked hand simply isn't drawn,
and the same rule removes phantom arm points. The rule is just "hide what
isn't real", with no new motion invented, so the risk is low.

The 2D fallback (Section 6) keeps its current behavior, unchanged.

---

## 6. When WebGL isn't there: fall back to the 2D renderer

If the browser can't create a WebGL context (it's disabled, unsupported, or
there's no GPU), `SkeletonCanvas` **falls back to the existing 2D Canvas
renderer**, unchanged, rather than showing a blank stage. That keeps the
2D path alive as the degraded mode (so it isn't dead code), and it's the
path the test environment takes: **happy-dom has no WebGL**, so every
existing component and App test keeps exercising the real playback loop and
the 2D drawing, unchanged.

The component checks for a context first (`canvas.getContext("webgl2")`) and
hands it to Three.js, so there's no Three.js error noise when it's missing.

---

## 7. Testing, and its limit

- **Pure logic is unit-tested** in happy-dom: `computeContentBounds3D`,
  `computeFitTransform3D`, `stepCamera3D` (easing, hold-steady, reduced
  motion), `cameraPlacement3D` (distance, direction), the world-axis mapping,
  and the per-frame geometry math (bone orientation and length, radii from
  palm length, what's hidden when a hand is untracked). That last part lives
  in a pure function, `computeHandGeometry`, precisely so it can be tested
  without a GPU.
- **WebGL rendering itself can't be tested here.** happy-dom has no GPU and
  no WebGL, the same kind of gap as the CSS layout bug found in Stage 5
  (`CLAUDE.md` gotchas). So the rendered result is checked **by eye in a real
  browser**, and that's recorded as a requirement, not an afterthought: A vs
  O, "about", two-handed "angry", and a multi-word transition, with
  screenshots.
- All existing tests must pass unchanged.

---

## 8. Explicitly out of scope

- **A photorealistic, skinned hand model** (a textured skin mesh deformed by
  the landmarks, or a Mixamo-rigged avatar). This pass is solid geometry
  (bones, joints, palm), not skin. A rigged avatar stays the separate stretch
  goal `CLAUDE.md` names.
- A face, a body, or facial expressions: there's no data for them.
- Correcting pose-subset depth or the image's aspect ratio (Section 2).
- A user-controlled camera (orbit or zoom) or changing the viewing angle.
- Shadows, reflections, and post-processing.

---

## Known gotchas / open questions

- The 20°/12° viewing angle and the thickness ratios are design judgments,
  to be checked by eye on real signs (Section 7). They're named constants,
  so they're easy to tune.
- Two hands can be far apart ("angry" spans up to 75% of the frame width), so
  the fit zooms out and each hand is small. That's the same trade-off the 2D
  view has, not a new one.
- Dataset letters are single frames held still, so they won't show
  micro-motion in 3D either.
