import { Elysia } from "elysia";
import fs from "fs";
import path from "path";


export interface ServeOptions {
  port: number;
  directory: string;
}

export const MDSERVE_ROUTE = "/__mdserve"


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
    .get("/*", async (ctx) => {
      const filepathOptions = [
        path.join(directory, `${ctx.path}.md`),
        path.join(directory, ctx.path, "index.md"),
      ]

      console.log(filepathOptions);

    })
    .listen(port, () => {
      console.log(`Listening on port ${port}`);
    })
}




