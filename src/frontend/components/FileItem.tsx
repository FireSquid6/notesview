import type { Node } from "../../filemap";

interface FileItemProps {
  node: Node;
  currentPath: string[];
  activePath: string[];
  level: number;
  parentPath: string[];
}

export function FileItem({ node, currentPath, activePath, level, parentPath }: FileItemProps): JSX.Element {
  const nodePath = [...currentPath, node.name];
  const pathString = nodePath.filter(p => p !== "").join("/");
  
  if (node.type === "directory") {
    const isExpanded = isPrefixOf(nodePath, activePath);
    const href = pathString ? `/${encodeURIComponent(pathString)}` : "/";
    
    return (
      <details open={isExpanded}>
        <summary class={`file-item folder-item ${isExpanded ? 'expanded' : ''}`} 
                 style={{ paddingLeft: `${level * 1.5 + 0.75}rem` }}>
          <span class="folder-chevron">{isExpanded ? '▼' : '▶'}</span>
          <span class="file-icon">📁</span>
          <a href={href} class="file-name folder-link">{node.name}</a>
        </summary>
        {node.children.map((child) => (
          <FileItem 
            node={child} 
            currentPath={nodePath} 
            activePath={activePath} 
            level={level + 1} 
            parentPath={nodePath}
          />
        ))}
      </details>
    );
  } else {
    const isActive = pathsEqual(nodePath, activePath);
    
    return (
      <a href={`/${pathString}`}
         class={`file-item file-link ${isActive ? 'active' : ''}`}
         style={{ paddingLeft: `${level * 1.5 + 0.75}rem` }}>
        <span class="file-icon">📄</span>
        <span class="file-name">{node.name}</span>
      </a>
    );
  }
}

function isPrefixOf(prefix: string[], path: string[]): boolean {
  if (prefix.length > path.length) return false;
  for (let i = 0; i < prefix.length; i++) {
    if (prefix[i] !== path[i]) return false;
  }
  return true;
}

function pathsEqual(path1: string[], path2: string[]): boolean {
  if (path1.length !== path2.length) return false;
  for (let i = 0; i < path1.length; i++) {
    if (path1[i] !== path2[i]) return false;
  }
  return true;
}
