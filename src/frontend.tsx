import { MDSERVE_ROUTE, PACKAGE_FILES_PREFIX } from "./server";

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
          <header class="header">
            <div class="header-content">
              <div class="header-left">
                <button id="sidebar-toggle" class="sidebar-toggle">
                  <span class="toggle-icon">☰</span>
                </button>
                <h1 class="header-title">Markdown Preview</h1>
              </div>
              <div class="filename-display">{filename || "Untitled"}</div>
            </div>
          </header>

          <div class="main-layout">
            <aside id="sidebar" class="sidebar">
              <div class="sidebar-header">
                <h3>Files</h3>
              </div>
              <nav class="file-list">
                {sidebar.map((file) => {
                  if (file.type === 'folder') {
                    return (
                      <div class={`file-item folder-item ${file.expanded ? 'expanded' : ''}`} data-folder={file.name} style={{ paddingLeft: `${file.level * 1.5 + 0.75}rem` }}>
                        <span class="folder-chevron">{file.expanded ? '▼' : '▶'}</span>
                        <span class="file-icon">📁</span>
                        <span class="file-name">{file.name}</span>
                      </div>
                    );
                  } else {
                    const isVisible = !file.parent || sidebar.find(f => f.name === file.parent && f.type === 'folder')?.expanded;
                    const filePath = file.parent ? `${file.parent}/${file.name}` : file.name;
                    return (
                      <a href={`/${encodeURIComponent(filePath)}`}
                        class={`file-item file-link ${file.active ? 'active' : ''}`}
                        data-parent={file.parent}
                        style={{
                          paddingLeft: `${file.level * 1.5 + 0.75}rem`,
                          display: isVisible ? 'flex' : 'none'
                        }}>
                        <span class="file-icon">📄</span>
                        <span class="file-name">{file.name}</span>
                      </a>
                    );
                  }
                })}
              </nav>
            </aside>

            <main class="content-wrapper">
              <div class="content">
                {content}
              </div>
            </main>
          </div>
        </div>
      </body>
    </html>
  )
}


export function jsxToHtml(jsx: JSX.Element): string {
  return `<!DOCTYPE HTML>\n${jsx}`;
}


