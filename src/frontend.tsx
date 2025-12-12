import { PACKAGE_FILES_PREFIX } from "./server";

export interface PageOptions {
  content: string;
}


export function getPage({ content }: PageOptions): JSX.Element {
  return (
    <html>
      <head>
        <link rel="stylesheet" href={`${PACKAGE_FILES_PREFIX}/highlight.css`} />
        <link rel="stylesheet" href={`${PACKAGE_FILES_PREFIX}/katex.css`} />
        <script src={`${PACKAGE_FILES_PREFIX}/htmx.js`} />
        <script src={`${PACKAGE_FILES_PREFIX}/katex.js`} />

      </head>
      <body hx-boost>
        <div>
          <p>This is where more stuff will go</p>
        </div>
        <div>
          <main>
            {content}
          </main>
        </div>
      </body>
    </html>
  )
}


export function jsxToHtml(jsx: JSX.Element): string {
  return `<!DOCTYPE HTML>\n${jsx}`;
}


