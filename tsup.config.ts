import { readdir } from "node:fs/promises";

import { defineConfig, type Options } from "tsup";

const watchMode = process.env.TSUP_WATCH_MODE === "1";

// Common options for all packages.
const opts: Options = {
  // Externalize all imports by default, except for runtime helpers generated
  // by the compiler / bundler toolchain.
  external: [/^[^./]/],
  format: ["cjs", "esm"],
  platform: "browser",
  dts: true,
  clean: true,
};

export default defineConfig([
  {
    name: "jax",
    entry: ["src/index.ts"],
    outDir: "dist",
    ...opts,
    watch: watchMode && "src",
  },

  ...(await readdir("packages")).map((pkg) => ({
    name: pkg,
    entry: [`packages/${pkg}/src/index.ts`],
    outDir: `packages/${pkg}/dist`,
    ...opts,
    cwd: `packages/${pkg}`,
    watch: watchMode && `packages/${pkg}/src`, // Unaffected by cwd.
  })),
] as Options[]);
