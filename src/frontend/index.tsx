import { Layout } from "./components/Layout";

export interface PageOptions {
  content: string;
  filename: string;
  sidebar: SidebarItem[];
}

export interface SidebarItem {
  name: string;
  type: "file" | "folder";
  active: boolean;
  parent: string;
  expanded: boolean;
  level: number;
}

export function getPage({ content, filename, sidebar }: PageOptions): JSX.Element {
  return (
    <Layout filename={filename} sidebar={sidebar}>
      {content}
    </Layout>
  );
}


export function jsxToHtml(jsx: JSX.Element): string {
  return `<!DOCTYPE HTML>\n${jsx}`;
}


