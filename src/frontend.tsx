import type { Node } from "./filemap";
import { Layout } from "./frontend/components/Layout";

export interface PageOptions {
  content: string;
  filename: string;
  filetree: Node
  activePath: string[];
}


export function getPage({ content, filename, filetree, activePath }: PageOptions): JSX.Element {
  return (
    <Layout 
      filetree={filetree}
      filename={filename}
      activePath={activePath}
    >
      {content}
      </Layout>
  )
}


export function jsxToHtml(jsx: JSX.Element): string {
  return `<!DOCTYPE HTML>\n${jsx}`;
}


