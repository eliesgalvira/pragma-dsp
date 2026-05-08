import { execFileSync } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import packageJson from "../package.json" with { type: "json" };

const tempDir = await mkdtemp(join(tmpdir(), "pragma-dsp-pack-"));

try {
  const output = execFileSync("pnpm", ["pack", "--json", "--config.ignore-scripts=true", "--pack-destination", tempDir], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"]
  });
  const packResult = JSON.parse(output.trim());
  const packedFiles = new Set(packResult.files.map((file) => file.path));

  const targets = [
    ["main", packageJson.main],
    ["module", packageJson.module],
    ["types", packageJson.types],
    ...collectExportTargets(packageJson.exports)
  ];

  const missingTargets = targets
    .filter(([, target]) => typeof target === "string" && target.startsWith("./") && !target.includes("*"))
    .map(([field, target]) => [field, target.slice(2)])
    .filter(([, path]) => !packedFiles.has(path));

  if (missingTargets.length > 0) {
    console.error("Package manifest points at files that are not included in the packed tarball:");
    for (const [field, path] of missingTargets) {
      console.error(`- ${field}: ${path}`);
    }
    process.exitCode = 1;
  } else {
    console.log(`Validated ${targets.length} package entry targets against ${packResult.files.length} packed files.`);
  }
} finally {
  await rm(tempDir, { force: true, recursive: true });
}

function collectExportTargets(exportsMap, trail = "exports") {
  if (typeof exportsMap === "string") {
    return [[trail, exportsMap]];
  }

  if (exportsMap === null || typeof exportsMap !== "object" || Array.isArray(exportsMap)) {
    return [];
  }

  return Object.entries(exportsMap).flatMap(([key, value]) =>
    collectExportTargets(value, `${trail}.${key}`)
  );
}
