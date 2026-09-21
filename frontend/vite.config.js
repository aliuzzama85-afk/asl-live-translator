import { fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs/promises";

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Stopgap dev-server-only pose data serving, per frontend/PLAN.md's "Known
// gotchas" section: `pose_library/data/poses/*.json` and `manifest.json`
// live outside `frontend/` and there is no HTTP layer for them yet. Rather
// than copying/symlinking (stale on rebuild, unreliable on Windows without
// elevated permissions), this middleware reads directly from
// `pose_library/data/poses/` at request time, so it's always fresh.
//
// Serves:
//   GET /poses/manifest.json  -> pose_library/data/poses/manifest.json
//   GET /poses/<word>.json    -> pose_library/data/poses/<word>.json
//
// "word not found" and "library not built yet" (pose_library/data/ is
// gitignored and only exists once build_library.py has run) must look
// identical to the frontend -- both are a clean 404, never an uncaught
// exception or a raw 500 stack trace.
const POSES_DIR = path.resolve(__dirname, "../pose_library/data/poses");

// Only ever a bare lowercase-ish word or "manifest", no path separators or
// traversal segments -- reject anything else before it ever touches fs.
const SAFE_FILENAME = /^[a-z0-9_-]+\.json$/i;

function posesServingMiddleware() {
  return {
    name: "serve-pose-library-json",
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        if (!req.url) {
          next();
          return;
        }
        const url = new URL(req.url, "http://localhost");
        const match = url.pathname.match(/^\/poses\/([^/]+)$/);
        if (!match) {
          next();
          return;
        }

        const filename = match[1];
        if (!SAFE_FILENAME.test(filename)) {
          res.statusCode = 404;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ error: "not_found" }));
          return;
        }

        const filePath = path.join(POSES_DIR, filename);
        // Belt-and-suspenders traversal guard: resolved path must stay
        // inside POSES_DIR even though SAFE_FILENAME already forbids "/".
        if (path.dirname(filePath) !== POSES_DIR) {
          res.statusCode = 404;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ error: "not_found" }));
          return;
        }

        try {
          const data = await fs.readFile(filePath, "utf-8");
          res.statusCode = 200;
          res.setHeader("Content-Type", "application/json");
          res.end(data);
        } catch (err) {
          if (err && err.code === "ENOENT") {
            // Covers both "word not in library" and "library not built yet
            // on this machine" -- deliberately identical response, per
            // PLAN.md's gotchas section.
            res.statusCode = 404;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ error: "not_found" }));
            return;
          }
          next(err);
        }
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), posesServingMiddleware()],
  test: {
    // happy-dom over jsdom: no concrete reason to prefer jsdom here -- this
    // suite doesn't depend on any jsdom-only behavior, and happy-dom is
    // materially faster while covering everything these tests need
    // (matchMedia, ResizeObserver, requestAnimationFrame all work; only
    // canvas 2D context rendering doesn't, which src/test/setup.js stubs).
    environment: "happy-dom",
    setupFiles: ["./src/test/setup.js"],
  },
});
