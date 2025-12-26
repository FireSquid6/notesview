import { plugin } from "bun";

plugin({
  name: "text-loader",
  setup(build) {
    build.onLoad({ filter: /\.text\.(js|css)$/ }, async (args) => {
      const text = await Bun.file(args.path).text();
      return {
        contents: `export default ${JSON.stringify(text)}`,
        loader: "js",
      };
    });
  },
});
