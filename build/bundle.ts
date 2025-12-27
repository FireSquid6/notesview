import { textLoaderPlugin } from "./plugin.ts";
import fs from "fs";

fs.mkdirSync("./dist", { recursive: true });

await Bun.build({
  entrypoints: ["./src/index.ts"],
  outdir: "./dist",
  target: "bun",
  minify: false,
  sourcemap: "inline",
  plugins: [textLoaderPlugin],
});

console.log("Bundle complete! Output: dist/index.js");
