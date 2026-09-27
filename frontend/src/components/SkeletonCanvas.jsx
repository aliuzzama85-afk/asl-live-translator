import { useEffect, useMemo, useRef } from "react";

import { computeHandGeometry, describeHand } from "../lib/handGeometry.js";
import { createHandScene } from "../lib/handScene.js";
import {
  buildSkeletonTopology,
  computeContentBounds,
  computeContentBounds3D,
  computeFitTransform,
  computeFitTransform3D,
  filterOutPoseSubset,
  isUndetectedPoint,
  projectPoint,
  stepCamera,
  stepCamera3D,
} from "../lib/skeletonBones.js";
import styles from "./SkeletonCanvas.module.css";

/** Reads the real hex values out of tokens.css at draw time, rather than
 * duplicating them as separate JS literals -- tokens.css stays the single
 * source of truth (canvas drawing simply can't consume `var(...)` the way
 * CSS Modules can, so this is the closest equivalent). */
function readSkeletonColors() {
  const style = getComputedStyle(document.documentElement);
  return {
    bone: style.getPropertyValue("--color-skeleton-bone").trim() || "#F5F5F0",
    joint: style.getPropertyValue("--color-skeleton-joint").trim() || "#FFC857",
    bg: style.getPropertyValue("--color-bg").trim() || "#0A0A0B",
    panel: style.getPropertyValue("--color-panel").trim() || "#141417",
    accent: style.getPropertyValue("--color-accent-amber").trim() || "#FFC857",
  };
}

/** Creates the 3D scene if the browser can give this canvas a WebGL2
 * context, else returns `null` so the component keeps its 2D renderer
 * (`frontend/RENDERING_UPGRADE_PLAN.md` Section 6). Checked here, before any
 * Three.js code runs, so a missing context is a quiet fallback rather than a
 * Three.js error. happy-dom has no WebGL, so tests always take the 2D path. */
function tryCreateHandScene(canvas) {
  const context = canvas.getContext("webgl2");
  if (!context) return null;
  try {
    return createHandScene({ canvas, context, colors: readSkeletonColors() });
  } catch {
    return null;
  }
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

/** How quickly the soft-follow camera (see `cameraRef` in `SkeletonCanvas`)
 * eases toward each frame's own fit. Tuned empirically: high enough that
 * the camera settles on fast hand motion within a few hundred ms (not a
 * visible multi-second drift), low enough that it doesn't visibly snap
 * frame to frame. */
const CAMERA_SMOOTHING_ALPHA = 0.18;

function lerpPoint(pointA, pointB, t) {
  if (isUndetectedPoint(pointA) || isUndetectedPoint(pointB)) {
    return [0, 0, 0];
  }
  return [
    lerp(pointA[0], pointB[0], t),
    lerp(pointA[1], pointB[1], t),
    lerp(pointA[2], pointB[2], t),
  ];
}

/**
 * The Stage band: a lit 3D hand/arm renderer (Three.js, WebGL) driven by a
 * reconstructed timeline (see `lib/reconstructTimeline.js`), falling back to
 * the flat 2D Canvas skeleton when WebGL isn't available
 * (`frontend/RENDERING_UPGRADE_PLAN.md`).
 *
 * Playback uses a `requestAnimationFrame` loop with a delta-time
 * accumulator, advancing one source frame every `1000 / fps` ms of real
 * elapsed time (per PLAN.md Section 3) -- never one frame per rAF tick,
 * which would tie apparent sign speed to the display's refresh rate.
 *
 * When motion is not reduced, adjacent timeline frames are blended
 * (fractional lerp based on the accumulator's progress through the current
 * frame interval) for smoother-looking motion on high-refresh displays --
 * purely a rendering nicety layered on top of the true fps-accurate advance
 * logic, so it never distorts apparent sign speed. Reduced-motion users get
 * pure frame-stepped playback with no such blending, per PLAN.md's
 * accessibility section.
 *
 * @param {Object} props
 * @param {string[]} props.landmarkNames - The sequence's own `landmark_names`.
 * @param {import("../lib/reconstructTimeline.js").Timeline|null} props.timeline -
 *   The reconstructed timeline to play, or `null` while nothing is loaded.
 * @param {boolean} props.isPlaying - Whether playback is currently advancing.
 * @param {boolean} props.loop - Whether to wrap to frame 0 at the end.
 * @param {boolean} props.reducedMotion - Disables inter-frame blending.
 * @param {() => void} [props.onEnded] - Called once when a non-looping
 *   playback reaches its last frame.
 * @param {(frameIndex: number) => void} [props.onFrameChange] - Called on
 *   every tick (playing or paused) with the currently-displayed frame
 *   index. Purely a reporting hook -- this component stays word-agnostic;
 *   it only ever reports a frame index, never anything about "words". Lets
 *   a caller (e.g. `App.jsx`, for multi-word playback) map a frame index to
 *   "which word is playing now" via its own boundary table, per
 *   `frontend/MULTIWORD_PLAN.md` Section 3.
 */
export function SkeletonCanvas({
  landmarkNames,
  timeline = null,
  isPlaying,
  loop,
  reducedMotion,
  onEnded = undefined,
  onFrameChange = undefined,
}) {
  const canvasRef = useRef(null);
  const bezelRef = useRef(null);
  const dprSizeRef = useRef({ cssSize: 0, dpr: 1 });
  // The 3D scene, or `null` while on the 2D fallback (see `tryCreateHandScene`).
  const handSceneRef = useRef(null);

  const topology = useMemo(() => buildSkeletonTopology(landmarkNames), [landmarkNames]);
  const handDescription = useMemo(
    () => describeHand(landmarkNames, topology),
    [landmarkNames, topology]
  );

  // A single fit for the *whole* sequence badly dilutes the zoom benefit --
  // hands travel during a sign (a real word's combined hand span across its
  // full ~2s trajectory is ~35-45% of the frame) even though at any single
  // instant a hand only occupies ~12-17%. So the camera instead re-fits to
  // *each frame's own* real content and eases toward it (see `cameraRef`
  // below in the draw loop) -- smoothed rather than snapped, so it tracks
  // hand motion like a soft-follow camera instead of jittering frame to
  // frame. Seeded here from the whole-sequence fit purely as a stable
  // starting point before the first real frame is drawn.
  //
  // Fit to hand landmarks only, excluding the pose subset (shoulder/elbow/
  // wrist): for a sign where the hand is raised or extended (e.g. a real
  // "phone" extraction, where the shoulder-to-wrist chain alone spans
  // 60-100% of the frame), including those points would defeat the zoom
  // entirely for exactly the signs that most need it.
  const initialFitTransform = useMemo(() => {
    const poses = timeline
      ? timeline.frames.map((frame) =>
          filterOutPoseSubset(frame.pose, topology.isPoseSubsetByIndex)
        )
      : [];
    return {
      fit2D: computeFitTransform(computeContentBounds(poses)),
      fit3D: computeFitTransform3D(computeContentBounds3D(poses)),
    };
  }, [timeline, topology]);
  const cameraRef = useRef(initialFitTransform.fit2D);
  // The 3D renderer's camera: the same soft-follow fit, with depth.
  const camera3DRef = useRef(initialFitTransform.fit3D);

  const playbackRef = useRef({
    frameIndex: 0,
    accumulatorMs: 0,
    lastTimestamp: null,
    endedFired: false,
  });

  // Reset playback position and the soft-follow camera whenever a new
  // timeline is loaded, so a freshly loaded word doesn't inherit the
  // previous word's zoom/pan and visibly swim into place.
  useEffect(() => {
    playbackRef.current = {
      frameIndex: 0,
      accumulatorMs: 0,
      lastTimestamp: null,
      endedFired: false,
    };
    cameraRef.current = { ...initialFitTransform.fit2D };
    camera3DRef.current = { ...initialFitTransform.fit3D };
  }, [timeline, initialFitTransform]);

  // Created once per mounted canvas (declared before the sizing effect, so
  // the first sizing pass already reaches the scene).
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const handScene = tryCreateHandScene(canvas);
    handSceneRef.current = handScene;
    return () => {
      handSceneRef.current = null;
      handScene?.dispose();
    };
  }, []);

  // DPI-aware, responsive canvas sizing: the backing store is
  // `cssSize * devicePixelRatio`, scaled back down via ctx.scale, so the
  // skeleton stays sharp on high-DPI screens. Re-measured on resize, not
  // just on mount.
  useEffect(() => {
    const bezel = bezelRef.current;
    const canvas = canvasRef.current;
    if (!bezel || !canvas) return undefined;

    const applySize = () => {
      // The bezel box is kept square by CSS (aspect-ratio: 1 / 1); read its
      // actual rendered size rather than assuming, and take the min of
      // width/height defensively against rounding.
      const cssSize = Math.max(1, Math.floor(Math.min(bezel.clientWidth, bezel.clientHeight)));
      const dpr = window.devicePixelRatio || 1;
      dprSizeRef.current = { cssSize, dpr };
      if (handSceneRef.current) {
        handSceneRef.current.resize(cssSize, dpr);
        return;
      }
      canvas.width = Math.round(cssSize * dpr);
      canvas.height = Math.round(cssSize * dpr);
    };

    applySize();
    const observer = new ResizeObserver(applySize);
    observer.observe(bezel);
    return () => observer.disconnect();
  }, []);

  // The rAF playback + draw loop.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !timeline || timeline.frames.length === 0) return undefined;

    const handScene = handSceneRef.current;
    const ctx = handScene ? null : canvas.getContext("2d");
    let rafId;

    // The 3D renderer: the same soft-follow camera rules as the 2D one
    // below (`stepCamera3D` shares `stepCamera`'s easing and hold-steady
    // logic, and reduced motion still snaps with alpha=1). Untracked
    // landmarks are hidden, not drawn at the origin.
    const draw3D = (pose, dimmed) => {
      const camera = camera3DRef.current;
      const smoothingAlpha = reducedMotion ? 1 : CAMERA_SMOOTHING_ALPHA;
      stepCamera3D(camera, pose, topology.isPoseSubsetByIndex, smoothingAlpha);
      handScene.render(computeHandGeometry(pose, handDescription), { dimmed, camera });
    };

    const draw2D = (pose, dimmed) => {
      const { cssSize, dpr } = dprSizeRef.current;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssSize, cssSize);

      // Soft-follow camera: ease toward *this frame's own* fit rather than
      // snapping to it, so the zoom/pan tracks hand motion smoothly instead
      // of jittering frame to frame (a raw per-frame fit alone would jump
      // around with every small hand movement). Reduced-motion users get an
      // immediate snap instead (alpha=1) -- still re-fit per frame for
      // legibility, just without the continuous panning/zooming motion.
      // `stepCamera` holds the camera steady, rather than chasing
      // `computeContentBounds`' full-frame fallback, on a frame with no
      // trackable hand content at all (a cross-word transition between two
      // differently-handed signs) -- see its own docstring.
      const camera = cameraRef.current;
      const smoothingAlpha = reducedMotion ? 1 : CAMERA_SMOOTHING_ALPHA;
      stepCamera(camera, pose, topology.isPoseSubsetByIndex, smoothingAlpha);

      const colors = readSkeletonColors();
      const dimFactor = dimmed ? 0.5 : 1;

      // Bones first, joints on top.
      ctx.lineCap = "round";
      for (const bone of topology.bones) {
        const pointA = pose[bone.a];
        const pointB = pose[bone.b];
        if (bone.isPoseSubset && (isUndetectedPoint(pointA) || isUndetectedPoint(pointB))) {
          continue; // A genuinely undetected shoulder/elbow/wrist -- skip, don't draw to (0,0).
        }
        const isBridge = bone.kind === "bridge";
        ctx.globalAlpha = (isBridge ? 0.4 : 1) * dimFactor;
        ctx.strokeStyle = colors.bone;
        ctx.lineWidth = isBridge ? 2 : 3;
        const [ax, ay] = projectPoint(pointA, camera, cssSize);
        const [bx, by] = projectPoint(pointB, camera, cssSize);
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();
      }

      ctx.fillStyle = colors.joint;
      for (let i = 0; i < pose.length; i += 1) {
        const point = pose[i];
        if (topology.isPoseSubsetByIndex[i] && isUndetectedPoint(point)) {
          continue;
        }
        ctx.globalAlpha = dimFactor;
        const [px, py] = projectPoint(point, camera, cssSize);
        ctx.beginPath();
        ctx.arc(px, py, topology.jointRadiusByIndex[i], 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    };

    const draw = handScene ? draw3D : draw2D;

    const frames = timeline.frames;
    const lastIndex = frames.length - 1;

    const tick = (timestamp) => {
      const playback = playbackRef.current;

      if (!isPlaying) {
        const current = frames[playback.frameIndex];
        draw(current.pose, current.dimmed);
        onFrameChange?.(playback.frameIndex);
        playback.lastTimestamp = null;
        rafId = requestAnimationFrame(tick);
        return;
      }

      // A non-looping sign that already ran to the end leaves frameIndex
      // pinned at lastIndex and endedFired=true (see the `while` loop below).
      // Clicking Play again re-enters this branch with isPlaying=true, but
      // without this reset it would sit at the last frame forever -- true
      // yet visibly doing nothing, since nothing else ever moves frameIndex
      // back to 0. Treat "play after the end" as an explicit replay request.
      if (playback.endedFired && !loop) {
        playback.frameIndex = 0;
        playback.accumulatorMs = 0;
        playback.endedFired = false;
      }

      if (playback.lastTimestamp === null) {
        playback.lastTimestamp = timestamp;
      }
      const deltaMs = timestamp - playback.lastTimestamp;
      playback.lastTimestamp = timestamp;
      playback.accumulatorMs += deltaMs;

      while (playback.accumulatorMs >= timeline.frameDurationMs) {
        playback.accumulatorMs -= timeline.frameDurationMs;
        if (playback.frameIndex >= lastIndex) {
          if (loop) {
            playback.frameIndex = 0;
            playback.endedFired = false;
          } else {
            playback.accumulatorMs = 0;
            if (!playback.endedFired) {
              playback.endedFired = true;
              onEnded?.();
            }
          }
        } else {
          playback.frameIndex += 1;
        }
      }

      const current = frames[playback.frameIndex];
      if (reducedMotion || playback.frameIndex >= lastIndex) {
        draw(current.pose, current.dimmed);
      } else {
        const next = frames[playback.frameIndex + 1];
        const fraction = playback.accumulatorMs / timeline.frameDurationMs;
        const blended = current.pose.map((pt, i) => lerpPoint(pt, next.pose[i], fraction));
        draw(blended, current.dimmed);
      }
      onFrameChange?.(playback.frameIndex);

      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [timeline, isPlaying, loop, reducedMotion, onEnded, onFrameChange, topology, handDescription]);

  return (
    <div className={styles.stageWrapper}>
      <div ref={bezelRef} className={styles.bezel}>
        <canvas
          ref={canvasRef}
          className={styles.canvas}
          role="img"
          aria-label="ASL sign skeleton animation"
        />
      </div>
    </div>
  );
}
