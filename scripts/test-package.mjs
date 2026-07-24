import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const repo = join(import.meta.dirname, "..");
const artifactRoot = mkdtempSync(join(tmpdir(), "rath-package-test-"));
const packageDir = join(artifactRoot, "package");
const consumerDir = join(artifactRoot, "consumer");
mkdirSync(packageDir);
mkdirSync(consumerDir);

try {
  const packageFile = execFileSync(
    "npm",
    ["pack", "--silent", "--pack-destination", packageDir],
    { cwd: repo, encoding: "utf8" },
  ).trim();
  writeFileSync(
    join(consumerDir, "package.json"),
    `${JSON.stringify({ private: true, type: "module" }, null, 2)}\n`,
  );
  execFileSync(
    "npm",
    ["install", "--ignore-scripts", "--no-audit", "--no-fund", join(packageDir, packageFile)],
    { cwd: consumerDir, stdio: "inherit" },
  );
  execFileSync(
    process.execPath,
    ["--input-type=module", "--eval", 'await import("@zomglings/rath")'],
    { cwd: consumerDir, stdio: "inherit" },
  );
  execFileSync(join(consumerDir, "node_modules", ".bin", "rath"), ["--help"], {
    cwd: consumerDir,
    stdio: "ignore",
  });
} catch (error) {
  console.error(`Package test artifacts: ${artifactRoot}`);
  throw error;
}

rmSync(artifactRoot, { recursive: true });
console.log("Packed package imports and its CLI runs.");
