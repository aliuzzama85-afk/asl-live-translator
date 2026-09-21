// Vitest setup file (wired via vite.config.js's test.setupFiles).

// Registers jest-dom's matchers (toBeInTheDocument, toHaveTextContent, ...)
// onto Vitest's `expect`.
import "@testing-library/jest-dom/vitest";

import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

// happy-dom (like jsdom) has no real canvas rendering engine --
// HTMLCanvasElement.prototype.getContext("2d") returns `null`, which would
// throw as soon as SkeletonCanvas's draw loop calls a context method (e.g.
// `ctx.setTransform(...)`). Component tests only need to mount/unmount
// cleanly and assert on real DOM output (text, aria roles, canvas
// presence) -- none of them assert on actual pixel rendering -- so this
// stub is a minimal no-op 2D context, not a rendering implementation.
function createNoopCanvasContext() {
  return {
    setTransform: () => {},
    clearRect: () => {},
    beginPath: () => {},
    moveTo: () => {},
    lineTo: () => {},
    stroke: () => {},
    arc: () => {},
    fill: () => {},
    lineCap: "round",
    strokeStyle: "",
    fillStyle: "",
    lineWidth: 1,
    globalAlpha: 1,
  };
}

HTMLCanvasElement.prototype.getContext = function getContext(contextType) {
  return contextType === "2d" ? createNoopCanvasContext() : null;
};

// Vitest doesn't run with `test.globals: true` here, so Testing Library's
// own auto-cleanup (which detects a global `afterEach`) never registers --
// do it explicitly so each test unmounts its tree (and, critically, runs
// SkeletonCanvas's effect cleanups, which cancel its rAF loop and
// disconnect its ResizeObserver) before the next test starts.
afterEach(() => {
  cleanup();
});
