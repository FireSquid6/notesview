import type { SidebarItem } from "../index";

interface FileItemProps {
  file: SidebarItem;
  sidebar: SidebarItem[];
}

export function FileItem({ file, sidebar }: FileItemProps): JSX.Element {
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
}
