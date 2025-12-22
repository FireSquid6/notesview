import type { Node } from "../../filemap";
import { FileItem } from "./FileItem";

interface SidebarProps {
  fileTree: Node;
  activePath: string[];
}

export function Sidebar({ fileTree, activePath }: SidebarProps): JSX.Element {

  return (
    <aside id="sidebar" class="sidebar">
      <div class="sidebar-header">
        <h3>Files</h3>
      </div>
      <nav class="file-list">
        <FileItem
          node={fileTree}
          currentPath={[]}
          activePath={activePath}
          level={0}
        />
      </nav>
    </aside>
  );
}
