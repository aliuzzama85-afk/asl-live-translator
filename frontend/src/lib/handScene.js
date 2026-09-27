/**
 * The Three.js side of the 3D hand renderer
 * (`frontend/RENDERING_UPGRADE_PLAN.md` Section 3): a lit scene of tapered
 * bones, sphere joints, and translucent palms, redrawn in place from each
 * frame's `computeHandGeometry` output.
 *
 * Deliberately thin: *what* to draw (positions, radii, visibility) is
 * decided by the pure, unit-tested `handGeometry.js`; *where the camera
 * goes* by the pure `cameraPlacement3D`. This module only turns those into
 * GPU state, which can't be tested without a real WebGL context (plan
 * Section 7), so it's verified by eye in a browser instead.
 */

import * as THREE from "three";

import { BONE_TAPER } from "./handGeometry.js";
import { VIEW_3D, cameraPlacement3D } from "./skeletonBones.js";

/** Instance capacity per mesh. A frame has 48 landmarks and 51 bones today;
 * this leaves generous room for a future larger landmark set (e.g. face
 * points) before anything would be clipped. */
const MAX_INSTANCES = 256;

const ARM_OPACITY = 0.35;
const PALM_OPACITY = 0.35;
const DIMMED_FACTOR = 0.5;

const UP = new THREE.Vector3(0, 1, 0);

/**
 * Creates the scene on a canvas whose WebGL context the caller already
 * obtained (so a missing context is detected before any Three.js code runs).
 *
 * @param {Object} options
 * @param {HTMLCanvasElement} options.canvas
 * @param {WebGL2RenderingContext} options.context
 * @param {{bone: string, joint: string, bg: string, panel: string, accent: string}} options.colors -
 *   Hex strings from tokens.css.
 * @returns {{
 *   render: (geometry: ReturnType<typeof import("./handGeometry.js").computeHandGeometry>,
 *            options: {dimmed: boolean, camera: {centerX: number, centerY: number, centerZ: number, span: number}}) => void,
 *   resize: (cssSize: number, dpr: number) => void,
 *   dispose: () => void,
 * }}
 */
export function createHandScene({ canvas, context, colors }) {
  const renderer = new THREE.WebGLRenderer({ canvas, context, antialias: true });
  renderer.toneMapping = THREE.NoToneMapping;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(colors.bg);

  const camera = new THREE.PerspectiveCamera(VIEW_3D.fovDeg, 1, 0.01, 100);
  scene.add(camera);

  // Lights ride on the camera so shading stays constant as it follows the
  // hands. A directional light shines from its position toward its target,
  // so both are placed in camera-local space (camera looks down -Z).
  const hemisphere = new THREE.HemisphereLight(colors.bone, colors.panel, 1.2);
  camera.add(hemisphere);

  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.position.set(-1.5, 2, 1); // upper front-left
  key.target.position.set(0, 0, -2);
  camera.add(key, key.target);

  const rim = new THREE.DirectionalLight(colors.accent, 1.4);
  rim.position.set(0.8, 1, -6); // behind the hands, shining back toward the viewer
  rim.target.position.set(0, 0, -2);
  camera.add(rim, rim.target);

  const baseColors = {
    bone: new THREE.Color(colors.bone),
    joint: new THREE.Color(colors.joint),
  };
  const materials = {
    handBone: new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0 }),
    handJoint: new THREE.MeshStandardMaterial({ roughness: 0.35, metalness: 0 }),
    arm: new THREE.MeshStandardMaterial({
      roughness: 0.45,
      metalness: 0,
      transparent: true,
      opacity: ARM_OPACITY,
      depthWrite: false,
    }),
    palm: new THREE.MeshStandardMaterial({
      roughness: 0.6,
      metalness: 0,
      transparent: true,
      opacity: PALM_OPACITY,
      side: THREE.DoubleSide,
      depthWrite: false,
    }),
  };
  const materialBase = {
    handBone: baseColors.bone,
    handJoint: baseColors.joint,
    arm: baseColors.bone,
    palm: baseColors.bone,
  };

  // Unit shapes, scaled per instance: a bone is a cylinder from y=0 (base)
  // to y=1 (tip) with radius 1 at the base; a joint is a unit sphere. The
  // cylinder is open-ended: both ends always sit inside a wider joint
  // sphere, and end caps there only add a second surface to z-fight with.
  const boneGeometry = new THREE.CylinderGeometry(BONE_TAPER, 1, 1, 16, 1, true);
  boneGeometry.translate(0, 0.5, 0);
  const jointGeometry = new THREE.SphereGeometry(1, 20, 14);

  const makeInstanced = (shape, material) => {
    const mesh = new THREE.InstancedMesh(shape, material, MAX_INSTANCES);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.count = 0;
    mesh.frustumCulled = false; // Instances move every frame; bounds would go stale.
    scene.add(mesh);
    return mesh;
  };
  const meshes = {
    handBones: makeInstanced(boneGeometry, materials.handBone),
    handJoints: makeInstanced(jointGeometry, materials.handBone),
    accentJoints: makeInstanced(jointGeometry, materials.handJoint),
    armBones: makeInstanced(boneGeometry, materials.arm),
    armJoints: makeInstanced(jointGeometry, materials.arm),
  };

  // One palm mesh per hand: a fan of 4 triangles (12 vertices, unindexed so
  // each triangle gets its own flat normal).
  const palmMeshes = [0, 1].map(() => {
    const shape = new THREE.BufferGeometry();
    shape.setAttribute("position", new THREE.BufferAttribute(new Float32Array(4 * 3 * 3), 3));
    const mesh = new THREE.Mesh(shape, materials.palm);
    mesh.frustumCulled = false;
    mesh.visible = false;
    scene.add(mesh);
    return mesh;
  });

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  const end = new THREE.Vector3();
  const direction = new THREE.Vector3();
  const noRotation = new THREE.Quaternion();

  const writeInstances = (mesh, items, toMatrix) => {
    let count = 0;
    for (const item of items) {
      if (!item.visible || count >= MAX_INSTANCES) continue;
      toMatrix(item);
      mesh.setMatrixAt(count, matrix);
      count += 1;
    }
    mesh.count = count;
    mesh.instanceMatrix.needsUpdate = true;
  };

  const jointMatrix = (joint) => {
    position.fromArray(joint.position);
    scale.setScalar(joint.radius);
    matrix.compose(position, noRotation, scale);
  };
  const boneMatrix = (bone) => {
    position.fromArray(bone.start);
    end.fromArray(bone.end);
    direction.subVectors(end, position);
    const length = direction.length();
    quaternion.setFromUnitVectors(UP, direction.divideScalar(length));
    scale.set(bone.radius, length, bone.radius);
    matrix.compose(position, quaternion, scale);
  };

  let lastDimmed = null;
  const applyDimming = (dimmed) => {
    if (dimmed === lastDimmed) return;
    lastDimmed = dimmed;
    const factor = dimmed ? DIMMED_FACTOR : 1;
    for (const [name, material] of Object.entries(materials)) {
      material.color.copy(materialBase[name]).multiplyScalar(factor);
    }
  };

  return {
    render(geometry, { dimmed, camera: fit }) {
      applyDimming(dimmed);

      writeInstances(
        meshes.handJoints,
        geometry.joints.filter((j) => !j.isArm && !j.isAccent),
        jointMatrix
      );
      writeInstances(
        meshes.accentJoints,
        geometry.joints.filter((j) => j.isAccent),
        jointMatrix
      );
      writeInstances(
        meshes.armJoints,
        geometry.joints.filter((j) => j.isArm),
        jointMatrix
      );
      writeInstances(
        meshes.handBones,
        geometry.bones.filter((b) => !b.isArm),
        boneMatrix
      );
      writeInstances(
        meshes.armBones,
        geometry.bones.filter((b) => b.isArm),
        boneMatrix
      );

      geometry.palms.forEach((palm, i) => {
        const mesh = palmMeshes[i];
        if (!mesh) return;
        mesh.visible = palm.visible;
        if (!palm.visible) return;
        const attribute = mesh.geometry.getAttribute("position");
        palm.triangles.flat().forEach((vertex, v) => attribute.setXYZ(v, ...vertex));
        attribute.needsUpdate = true;
        mesh.geometry.computeVertexNormals();
      });

      const placement = cameraPlacement3D(fit);
      camera.position.fromArray(placement.position);
      camera.near = placement.distance * 0.1;
      camera.far = placement.distance * 4;
      camera.updateProjectionMatrix();
      camera.lookAt(...placement.target);

      renderer.render(scene, camera);
    },

    resize(cssSize, dpr) {
      renderer.setPixelRatio(dpr);
      // `false`: leave the canvas's CSS size to the stylesheet, as the 2D
      // renderer does.
      renderer.setSize(cssSize, cssSize, false);
    },

    dispose() {
      boneGeometry.dispose();
      jointGeometry.dispose();
      palmMeshes.forEach((mesh) => mesh.geometry.dispose());
      Object.values(meshes).forEach((mesh) => mesh.dispose());
      Object.values(materials).forEach((material) => material.dispose());
      renderer.dispose();
    },
  };
}
