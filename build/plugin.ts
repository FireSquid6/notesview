import { plugin, type BunPlugin } from "bun";

export const textLoaderPlugin: BunPlugin = {
  name: "text-loader",
  setup(build) {
    build.onLoad({ filter: /\.text\.(js|css)$/ }, async (args) => {
      if (args.path.includes("loader.ts")) return;

      const text = await Bun.file(args.path).text();
      return {
        contents: `export default ${JSON.stringify(text)}`,
        loader: "js",
      };
    });
  },
};

