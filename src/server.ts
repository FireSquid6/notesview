import { Elysia } from "elysia";
import fs from "fs";
import path from "path";
import { renderHtml } from "./renderer";
import { getPage, jsxToHtml } from "./frontend";


export interface ServeOptions {
  port: number;
  directory: string;
}

export const MDSERVE_ROUTE = "/__mdserve"
export const PACKAGE_FILES_PREFIX = "/__packagefiles";

export const packageFiles: Record<string, string> = {
  "highlight.css": "node_modules/highlight.js/styles/tokyo-night-dark.css",
  "htmx.js": "node_modules/htmx.org/dist/htmx.min.js",
}

export function serveDirectory({ port, directory }: ServeOptions) {
  new Elysia()
    .get(`${MDSERVE_ROUTE}/*`, (ctx) => {
      const split = ctx.path.split("/");
      split.shift();
      split.shift();
      const filepath = path.resolve(
        __dirname,
        "..",
        "static",
        split.join("/"),
      );
      console.log(filepath);

      if (!fs.existsSync(filepath)) {
        return ctx.status(404);
      }

      return ctx.status(200, Bun.file(filepath));

    })
    .get(`${PACKAGE_FILES_PREFIX}/*`, async (ctx) => {
      const requestedFilename = ctx.path.split("\n").pop()!;
      const foundFilename = Object.keys(packageFiles).find(k => k === requestedFilename);

      if (foundFilename === undefined) {
        return ctx.status(404);
      }

      const filepath = path.resolve(
        __dirname,
        "..",
        packageFiles[foundFilename]!
      );

      if (!fs.existsSync(filepath)) {
        return ctx.status(404);
      }

      return ctx.status(200, Bun.file(filepath));

    })
    .get("/*", async (ctx) => {
      const filepathOptions = [
        path.join(directory, `${ctx.path}.md`),
        path.join(directory, ctx.path, "index.md"),
      ]

      let filepath = "";
      for (const fp of filepathOptions) {
        if (fs.existsSync(fp)) {
          filepath = fp;
          break;
        }
      }

      if (filepath === "") {
        return ctx.status(404);
      }

      const text = fs.readFileSync(filepath).toString();
      const content = await renderHtml(text);

      const page = getPage({
        content,
      });
      const html = jsxToHtml(page);

      ctx.set.headers["content-type"] = "text/html";
      return ctx.status(200, html);

    })
    .listen(port, () => {
      console.log(`Listening on port ${port}`);
    })
}




