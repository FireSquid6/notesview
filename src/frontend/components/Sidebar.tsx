import type { SidebarItem } from "../index";
import { FileItem } from "./FileItem";

interface SidebarProps {
  sidebar: SidebarItem[];
}

export function Sidebar({ sidebar }: SidebarProps): JSX.Element {
  return (
    <aside id="sidebar" class="sidebar">
      <div class="sidebar-header">
        <h3>Files</h3>
      </div>
      <nav class="file-list">
        <a href="/" class="file-item file-link home-link">
          <span class="file-icon">🏠</span>
          <span class="file-name">Home</span>
        </a>
        {sidebar.map((file) => (
          <FileItem file={file} sidebar={sidebar} />
        ))}
      </nav>
    </aside>
  );
}
