/**
 * Starts the Express gateway and the Vite client together, with prefixed,
 * interleaved output and one Ctrl-C that stops both.
 *
 * Deliberately dependency-free. The obvious choice here is `concurrently`, but
 * it shells out through `cmd.exe` resolved from PATH, which fails with
 * `spawn cmd.exe ENOENT` on any Windows box whose PATH is missing System32 --
 * a surprisingly common state in trimmed VS Code / CI environments, and the
 * state of the machine this was written on. Spawning `node npm-cli.js`
 * directly needs no shell at all, so it behaves the same on Windows, macOS
 * and Linux, and `npm run dev` works straight after `git clone` with nothing
 * installed at the repository root.
 */
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

const APPS = [
  { name: "server", dir: "server", script: "dev", color: "\x1b[36m" }, // cyan
  { name: "client", dir: "client", script: "dev", color: "\x1b[32m" }, // green
];

const RESET = "\x1b[0m";
const DIM = "\x1b[2m";

/** npm ships as a JS file next to the node binary; running it with the current
 *  interpreter avoids shell resolution entirely. */
function npmCli() {
  const candidate = path.join(path.dirname(process.execPath), "node_modules", "npm", "bin", "npm-cli.js");
  return existsSync(candidate) ? candidate : null;
}

function start({ name, dir, script, color }) {
  const cwd = path.join(ROOT, dir);

  if (!existsSync(path.join(cwd, "node_modules"))) {
    console.error(`${color}[${name}]${RESET} node_modules missing -- run: npm run install:node`);
    process.exit(1);
  }

  const cli = npmCli();
  const child = cli
    ? spawn(process.execPath, [cli, "run", script], { cwd, stdio: ["ignore", "pipe", "pipe"] })
    : // Fallback for unusual installs (nvm shims, npm not beside node).
      spawn("npm", ["run", script], { cwd, stdio: ["ignore", "pipe", "pipe"], shell: true });

  const prefix = `${color}[${name}]${RESET} `;
  const forward = (stream, sink) => {
    let buffer = "";
    stream.on("data", (chunk) => {
      buffer += chunk.toString();
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) sink.write(prefix + line + "\n");
    });
  };
  forward(child.stdout, process.stdout);
  forward(child.stderr, process.stderr);

  child.on("error", (err) => console.error(`${prefix}failed to start: ${err.message}`));
  child.on("exit", (code) => {
    console.log(`${prefix}${DIM}exited with code ${code}${RESET}`);
    shutdown();
  });

  return child;
}

const children = APPS.map(start);

let shuttingDown = false;
function shutdown() {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const child of children) {
    if (!child.killed) child.kill();
  }
  // Give each child a moment to close its listener before this process goes.
  setTimeout(() => process.exit(0), 300);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

console.log(`${DIM}starting gateway on :5000 and client on :5173 -- Ctrl-C stops both${RESET}\n`);
