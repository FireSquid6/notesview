
export interface PageOptions {
  content: string;
}


export function getPage({ content }: PageOptions): JSX.Element {
  return (
    <html>
      <head>

      </head>
      <body>
        <div>
          <p>This is where more stuff will go</p>
        </div>
        <main>
          {content}
        </main>
      </body>
    </html>
  )
}


export function jsxToHtml(jsx: JSX.Element): string {
  return `<!DOCTYPE HTML>\n${jsx}`;
}


