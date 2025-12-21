import { MDSERVE_ROUTE, PACKAGE_FILES_PREFIX } from "../../server";
import { Header } from "./Header";
import { Sidebar } from "./Sidebar";
import type { SidebarItem } from "../index";

interface LayoutProps {
  filename: string;
  sidebar: SidebarItem[];
  children: JSX.Element;
}

export function Layout({ filename, sidebar, children }: LayoutProps): JSX.Element {
  return (
    <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Markdown Preview - {filename}</title>
        <link rel="stylesheet" href={`${PACKAGE_FILES_PREFIX}/highlight.css`} />
        <link rel="stylesheet" href={`${PACKAGE_FILES_PREFIX}/katex.css`} />
        <link rel="stylesheet" href={`${MDSERVE_ROUTE}/main.css`} />
        <script src={`${PACKAGE_FILES_PREFIX}/htmx.js`} />
        <script src={`${PACKAGE_FILES_PREFIX}/katex.js`} />
        <script src={`${MDSERVE_ROUTE}/main.js`} />
      </head>
      <body hx-boost>
        <div class="app-layout">
          <Header filename={filename} />
          <div class="main-layout">
            <Sidebar sidebar={sidebar} />
            <main class="content-wrapper">
              <div class="content">
                {children}
              </div>
            </main>
          </div>
        </div>
      </body>
    </html>
  );
}
