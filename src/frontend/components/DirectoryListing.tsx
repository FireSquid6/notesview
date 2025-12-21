import type { Node } from "../../filemap";

interface DirectoryListingProps {
  node: Node;
  currentPath: string[];
}

export function DirectoryListing({ node, currentPath }: DirectoryListingProps): JSX.Element {
  if (node.type !== "directory") {
    return <div class="directory-listing-error">Error: Not a directory</div>;
  }

  const pathString = currentPath.filter(p => p !== "").join("/");
  const displayPath = pathString || "/";

  return (
    <div class="directory-listing">
      <div class="directory-listing-header">
        <h2>Directory: {displayPath}</h2>
        <p class="directory-listing-count">
          {node.children.length} {node.children.length === 1 ? 'item' : 'items'}
        </p>
      </div>
      
      <div class="directory-listing-content">
        {node.children.length === 0 ? (
          <div class="directory-listing-empty">
            <p>This directory is empty.</p>
          </div>
        ) : (
          <div class="directory-listing-grid">
            {node.children.map((child) => (
              <DirectoryItem 
                node={child} 
                currentPath={currentPath}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

interface DirectoryItemProps {
  node: Node;
  currentPath: string[];
}

function DirectoryItem({ node, currentPath }: DirectoryItemProps): JSX.Element {
  const nodePath = [...currentPath, node.name];
  const pathString = nodePath.filter(p => p !== "").join("/");
  const href = pathString ? `/${encodeURIComponent(pathString)}` : "/";

  return (
    <a href={href} class={`directory-item ${node.type === "directory" ? "directory-item-folder" : "directory-item-file"}`}>
      <div class="directory-item-icon">
        {node.type === "directory" ? "📁" : "📄"}
      </div>
      <div class="directory-item-name">{node.name}</div>
      <div class="directory-item-type">
        {node.type === "directory" ? "Folder" : "File"}
      </div>
    </a>
  );
}
