import fs from "fs";
import path from "path";
import type { SidebarItem } from "./frontend";

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
        name: path.basename(filename, ".md"),
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
        name: path.basename(filename, ".md"),
        content: {
          type: "markdown-file",
          filepath: filepath,
        }
      });
    }

  }
  return nodes;
}

export function filemapToSidebar(root: Node, expandedPath: string[]): SidebarItem[] {
  const items: SidebarItem[] = [];

  function traverse(node: Node, currentPath: string[], level: number, parentName: string) {
    if (node.type === "directory") {
      if (node.name !== "") {
        const nodePath = [...currentPath, node.name];
        const isExpanded = isPrefixOf(nodePath, expandedPath);

        items.push({
          name: node.name,
          type: "folder",
          active: false,
          parent: parentName,
          expanded: isExpanded,
          level: level
        });
      }

      for (const child of node.children) {
        const newPath = node.name === "" ? currentPath : [...currentPath, node.name];
        const newParent = node.name === "" ? "" : node.name;
        const newLevel = node.name === "" ? level : level + 1;
        traverse(child, newPath, newLevel, newParent);
      }
    } else {
      const nodePath = [...currentPath, node.name];
      const isActive = pathsEqual(nodePath, expandedPath);

      items.push({
        name: node.name,
        type: "file",
        active: isActive,
        parent: parentName,
        expanded: false,
        level: level
      });
    }
  }

  traverse(root, [], 0, "");
  return items;
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

export function printFilemap(node: Node, indent: string = ""): void {
  if (node.type === "directory") {
    if (node.name !== "") {
      console.log(`${indent}📁 ${node.name}/`);
      const nextIndent = indent + "  ";
      for (const child of node.children) {
        printFilemap(child, nextIndent);
      }
    } else {
      for (const child of node.children) {
        printFilemap(child, indent);
      }
    }
  } else {
    console.log(`${indent}📄 ${node.name}`);
  }
}
