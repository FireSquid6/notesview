import { textLoaderPlugin } from "./plugin.ts";

await Bun.build({
  entrypoints: ["./src/index.ts"],
  outdir: "./dist",
  target: "bun",
  minify: false,
  sourcemap: "external",
  plugins: [textLoaderPlugin],
});

console.log("Build complete! Output: dist/index.js");
