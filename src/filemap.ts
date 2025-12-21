import fs from "fs";
import path from "path";

export type Node = FileNode | DirectoryNode;

export type ContentData = {
  type: "markdown-file";
  filepath: string;
} | {
  type: "directory-listing";
}

export type FileNode = {
  type: "file";
  name: string;
  content: ContentData;
}

export type DirectoryNode = {
  type: "directory";
  children: Node[];
  name: string;
  content: ContentData;
}


export function getFileTree(rootDirectory: string): Node {
  const indexFile = path.join(rootDirectory, "index.md");
  const children = getDirectoryChildren(rootDirectory);

  const content: ContentData = fs.existsSync(indexFile) ? {
    type: "markdown-file",
    filepath: indexFile,
  } : {
    type: "directory-listing",
  }

  return {
    type: "directory",
    children,
    name: "",
    content,
  }
}

// null indicates that the file path was not found
export function matchFilePath(parts: string[], root: Node): ContentData | null {
  if (parts.length === 0) {
    return root.content;
  }

  let i = 0;
  let current = root;

  while (i < parts.length) {
    const part = parts[i]!;

    if (current.type !== "directory") {
      return null;
    }

    const next = current.children.find(n => n.name === part);

    if (next === undefined) {
      return null;
    }

    current = next;
    i++;
  }

  return current.content;
}

function getDirectoryChildren(directory: string): Node[] {
  const nodes: Node[] = [];
  for (const filename of fs.readdirSync(directory)) {
    const filepath = path.join(directory, filename);

    const stats = fs.statSync(filepath);

    if (stats.isDirectory()) {
      const children = getDirectoryChildren(filepath);
      const indexPath = path.join(filepath, "index.md");

      const content: ContentData = fs.existsSync(indexPath) ? {
        type: "markdown-file",
        filepath: indexPath,
      } : {
        type: "directory-listing",
      }

      nodes.push({
        type: "directory",
        children,
        name: filename,
        content,
      });
    } else {
      if (filename === "index.md") {
        continue;
      }
      const ext = path.extname(filepath);
      if (ext !== ".md") {
        continue;
      }

      nodes.push({
        type: "file",
        name: filename,
        content: {
          type: "markdown-file",
          filepath: filepath,
        }
      });
    }

  }
  return nodes;
}
