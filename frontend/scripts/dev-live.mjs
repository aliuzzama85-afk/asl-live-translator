/**
 * `npm run dev:live`: starts the gloss server and the Vite dev server
 * together, for live speech (`pipeline/STAGE1_2_PLAN.md` Section 1).
 *
 * Node standard library only, no new dependency. Each process's output is
 * prefixed ([gloss] / [vite]). If either exits, or you press Ctrl+C, the
 * other is stopped too, so nothing is left running in the background.
 * `npm run dev` is unchanged and starts the app alone.
 */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const FRONTEND_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const REPO_ROOT = path.resolve(FRONTEND_DIR, "..");

/**
 * Finds the project venv's Python.
 *
 * @param {string} root - Repo root.
 * @param {string} [platform]
 * @param {(p: string) => boolean} [exists]
 * @returns {string|null} The interpreter path, or `null` if there's no venv.
 */
export function resolveVenvPython(root, platform = process.platform, exists = existsSync) {
  const candidates =
    platform === "win32"
      ? [path.join(root, ".venv", "Scripts", "python.exe")]
      : [path.join(root, ".venv", "bin", "python")];
  return candidates.find((c) => exists(c)) ?? null;
}

function prefixed(stream, label) {
  let buffer = "";
  stream.on("data", (chunk) => {
    buffer += chunk.toString();
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop();
    for (const line of lines) process.stdout.write(`${label} ${line}\n`);
  });
}

function main() {
  const python = resolveVenvPython(REPO_ROOT);
  if (!python) {
    console.error(
      "No Python venv found at .venv/. Create it first (setup_env.ps1 or setup_env.sh), " +
        "or run the app alone with: npm run dev"
    );
    process.exit(1);
  }

  const children = [];
  let stopping = false;

  function stopAll(code) {
    if (stopping) return;
    stopping = true;
    for (const child of children) {
      if (child.exitCode === null) child.kill();
    }
    process.exitCode = code;
  }

  function start(label, command, args, cwd) {
    const child = spawn(command, args, { cwd, env: process.env });
    children.push(child);
    prefixed(child.stdout, label);
    prefixed(child.stderr, label);
    child.on("exit", (code) => {
      if (!stopping) {
        console.log(`${label} exited (${code ?? "signal"}); stopping the other process.`);
        stopAll(code ?? 1);
      }
    });
    child.on("error", (err) => {
      console.error(`${label} failed to start: ${err.message}`);
      stopAll(1);
    });
  }

  start("[gloss]", python, ["-m", "pipeline.gloss_server"], REPO_ROOT);
  start(
    "[vite] ",
    process.execPath,
    [path.join(FRONTEND_DIR, "node_modules", "vite", "bin", "vite.js"), ...process.argv.slice(2)],
    FRONTEND_DIR
  );

  process.on("SIGINT", () => stopAll(0));
  process.on("SIGTERM", () => stopAll(0));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}
