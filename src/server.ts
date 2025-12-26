import { Elysia } from "elysia";
import fs from "fs";
import path from "path";
import { renderHtml } from "./renderer";
import { getContentPage, getDirectoryPage, getSidebarForPage, jsxToHtml } from "./frontend";
import { getFileTree, matchFilePath, printFilemap as printFiletree } from "./filemap";

import jsSource from "../static/main.text.js";
import cssSource from "../static/main.text.css";


export interface ServeOptions {
  port: number;
  directory: string;
  watchForUpdates: boolean;
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

export function serveDirectory({ port, directory, watchForUpdates }: ServeOptions) {
  const ft = getFileTree(directory);
  printFiletree(ft);

  const app = new Elysia()
    .state("filetree", ft)
    .get(`${MDSERVE_ROUTE}/main.js`, (ctx) => {
      ctx.set.headers["content-type"] = "text/javascript";
      return ctx.status(200, jsSource);
    })
    .get(`${MDSERVE_ROUTE}/main.css`, (ctx) => {
      ctx.set.headers["content-type"] = "text/css";
      return ctx.status(200, cssSource);
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
    .get("/__partials/*", async (ctx) => {
      const pathParts = decodeURI(ctx.path).split("/");

      while (pathParts[0] === "" || pathParts[0] === "__partials") {
        pathParts.shift();
      }
      // TODO - ensure path parts actually starts
      const contentData = matchFilePath(pathParts, ctx.store.filetree);

      if (contentData === null) {
        return ctx.status(404);
      }

      ctx.set.headers["content-type"] = "text/plain";
      if (contentData.type === "directory-listing") {
        return ctx.status(200, "NO_UPDATE");
      } else {
        const text = fs.readFileSync(contentData.filepath).toString();
        const content = await renderHtml(text);

        return ctx.status(200, content);
      }

    })
    .get("/__only-sidebar/*", async (ctx) => {
      const pathParts = decodeURI(ctx.path).split("/");

      while (pathParts[0] === "" || pathParts[0] === "__partials") {
        pathParts.shift();
      }
      
      const sidebar = getSidebarForPage(ctx.store.filetree, pathParts);

      ctx.set.headers["content-type"] = "text/plain";
      return ctx.status(200, sidebar);

    })
    .get("/*", async (ctx) => {
      const pathParts = decodeURI(ctx.path).split("/");

      while (pathParts[0] === "") {
        pathParts.shift();
      }
      // TODO - ensure path parts actually starts
      const contentData = matchFilePath(pathParts, ctx.store.filetree);

      if (contentData === null) {
        // TODO - 404 page
        return ctx.status(404);
      }
      const filename = path.basename(contentData.filepath);

      if (contentData.type === "directory-listing") {
        const page = getDirectoryPage({
          filetree: ctx.store.filetree,
          activePath: pathParts,
          directoryName: filename,
        });

        const html = jsxToHtml(page);
        ctx.set.headers["content-type"] = "text/html";
        return ctx.status(200, html);
      } else {
        const text = fs.readFileSync(contentData.filepath).toString();
        const content = await renderHtml(text);


        const page = getContentPage({
          filetree: ctx.store.filetree,
          activePath: pathParts,
          content,
          filename,
        });
        const html = jsxToHtml(page);

        ctx.set.headers["content-type"] = "text/html";
        return ctx.status(200, html);
      }

    })
    .ws("/__update-listener", {
      open(ws) {
        ws.subscribe("updates");
      },
      close(ws) {
        ws.unsubscribe("updates");
      }
    })
    .listen(port, () => {
      console.log(`Listening on port ${port}`);
    })
  
  if (watchForUpdates) {
    const updater = timeoutRun(() => {
      const filetree = getFileTree(directory);
      console.log("\nUpdated:")
      printFiletree(filetree);

      app.store.filetree = filetree;
      app.server!.publish("updates", "UPDATE");

    }, 1000);

    fs.watch(directory, { recursive: true }, () => {
      updater();
    });
  }
}


function timeoutRun<T extends (...args: any[]) => any>(
  f: T,
  t: number
): (...args: Parameters<T>) => void {
  let lastCallTime = 0;

  return function(...args: Parameters<T>) {
    const now = Date.now();
    if (now - lastCallTime >= t) {
      lastCallTime = now;
      f(...args);
    }
  };
}

