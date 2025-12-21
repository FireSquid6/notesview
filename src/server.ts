import { Elysia } from "elysia";
import fs from "fs";
import path from "path";
import { renderHtml } from "./renderer";
import { getPage, jsxToHtml } from "./frontend";
import { filemapToSidebar, getFileTree, matchFilePath, printFilemap } from "./filemap";


export interface ServeOptions {
  port: number;
  directory: string;
}

export const MDSERVE_ROUTE = "/__mdserve"
export const PACKAGE_FILES_PREFIX = "/__packagefiles";

export const packageFiles: Record<string, string> = {
  "highlight.css": "node_modules/highlight.js/styles/tokyo-night-dark.css",
  "katex.css": "node_modules/katex/dist/katex.css",
  "htmx.js": "node_modules/htmx.org/dist/htmx.min.js",
  "katex.js": "node_modules/katex/dist/katex.js",
  "tailwind.css": "node_modules/tailwindcss/index.css",
}

export function serveDirectory({ port, directory }: ServeOptions) {
  const ft = getFileTree(directory);
  printFilemap(ft);

  new Elysia()
    .state("filetree", ft)
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

      if (!fs.existsSync(filepath)) {
        return ctx.status(404);
      }

      return ctx.status(200, Bun.file(filepath));

    })
    .get(`${PACKAGE_FILES_PREFIX}/fonts/*`, async (ctx)=> {
      const fontName = ctx.path.split("/").pop()!;
      const filepath = path.join("node_modules/katex/dist/fonts", fontName);

      return Bun.file(filepath);

    })
    .get(`${PACKAGE_FILES_PREFIX}/*`, async (ctx) => {
      const requestedFilename = ctx.path.split("/").pop()!;
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
      const pathParts = ctx.path.split("/");

      while (pathParts[0] === "") {
        pathParts.shift();
      }
      // TODO - ensure path parts actually starts
      const contentData = matchFilePath(pathParts, ctx.store.filetree);

      if (contentData === null) {
        // TODO - 404 page
        return ctx.status(404);
      }

      if (contentData.type === "directory-listing") {
        // TODO - directory listing
        return ctx.status(500);
      }

      const text = fs.readFileSync(contentData.filepath).toString();
      const content = await renderHtml(text);
      const filename = path.basename(contentData.filepath);
      const sidebar = filemapToSidebar(ctx.store.filetree, pathParts);


      const page = getPage({
        sidebar,
        content,
        filename,
      });
      const html = jsxToHtml(page);

      ctx.set.headers["content-type"] = "text/html";
      return ctx.status(200, html);

    })
    .listen(port, () => {
      console.log(`Listening on port ${port}`);
    })
}




