import { describe, it, expect, beforeEach, afterEach } from "bun:test";
import fs from "fs";
import path from "path";
import { getFileTree, matchFilePath, type Node, type ContentData } from "../src/filemap";

const TEST_DIR = path.join(process.cwd(), "test-temp");

describe("filemap", () => {
  beforeEach(() => {
    // Clean up and create fresh test directory
    if (fs.existsSync(TEST_DIR)) {
      fs.rmSync(TEST_DIR, { recursive: true });
    }
    fs.mkdirSync(TEST_DIR, { recursive: true });
  });

  afterEach(() => {
    // Clean up test directory
    if (fs.existsSync(TEST_DIR)) {
      fs.rmSync(TEST_DIR, { recursive: true });
    }
  });

  describe("getFileTree", () => {
    it("should return directory listing for empty directory", () => {
      const result = getFileTree(TEST_DIR);
      
      expect(result.type).toBe("directory");
      expect(result.name).toBe("");
      expect(result.children).toEqual([]);
      expect(result.content.type).toBe("directory-listing");
    });

    it("should return markdown content when index.md exists in root", () => {
      const indexPath = path.join(TEST_DIR, "index.md");
      fs.writeFileSync(indexPath, "# Root Index");
      
      const result = getFileTree(TEST_DIR);
      
      expect(result.content.type).toBe("markdown-file");
      if (result.content.type === "markdown-file") {
        expect(result.content.filepath).toBe(indexPath);
      }
    });

    it("should include markdown files in directory", () => {
      fs.writeFileSync(path.join(TEST_DIR, "readme.md"), "# README");
      fs.writeFileSync(path.join(TEST_DIR, "docs.md"), "# Docs");
      fs.writeFileSync(path.join(TEST_DIR, "test.txt"), "ignored"); // Should be ignored
      
      const result = getFileTree(TEST_DIR);
      
      expect(result.children).toHaveLength(2);
      expect(result.children.map(c => c.name).sort()).toEqual(["docs.md", "readme.md"]);
      
      const readmeNode = result.children.find(c => c.name === "readme.md");
      expect(readmeNode?.type).toBe("file");
      expect(readmeNode?.content.type).toBe("markdown-file");
    });

    it("should ignore index.md files when creating file nodes", () => {
      fs.writeFileSync(path.join(TEST_DIR, "index.md"), "# Index");
      fs.writeFileSync(path.join(TEST_DIR, "other.md"), "# Other");
      
      const result = getFileTree(TEST_DIR);
      
      // index.md should be used as directory content, not as a child
      expect(result.children).toHaveLength(1);
      expect(result.children[0].name).toBe("other.md");
      expect(result.content.type).toBe("markdown-file");
    });

    it("should handle nested directories", () => {
      const subDir = path.join(TEST_DIR, "docs");
      fs.mkdirSync(subDir);
      fs.writeFileSync(path.join(subDir, "guide.md"), "# Guide");
      
      const result = getFileTree(TEST_DIR);
      
      expect(result.children).toHaveLength(1);
      const docsNode = result.children[0];
      expect(docsNode.name).toBe("docs");
      expect(docsNode.type).toBe("directory");
      expect(docsNode.children).toHaveLength(1);
      expect(docsNode.children[0].name).toBe("guide.md");
    });

    it("should use index.md in subdirectories", () => {
      const subDir = path.join(TEST_DIR, "docs");
      fs.mkdirSync(subDir);
      fs.writeFileSync(path.join(subDir, "index.md"), "# Docs Index");
      fs.writeFileSync(path.join(subDir, "guide.md"), "# Guide");
      
      const result = getFileTree(TEST_DIR);
      
      const docsNode = result.children[0];
      expect(docsNode.content.type).toBe("markdown-file");
      if (docsNode.content.type === "markdown-file") {
        expect(docsNode.content.filepath).toBe(path.join(subDir, "index.md"));
      }
      // Should still include other markdown files
      expect(docsNode.children).toHaveLength(1);
      expect(docsNode.children[0].name).toBe("guide.md");
    });

    it("should ignore non-markdown files", () => {
      fs.writeFileSync(path.join(TEST_DIR, "readme.md"), "# README");
      fs.writeFileSync(path.join(TEST_DIR, "package.json"), "{}");
      fs.writeFileSync(path.join(TEST_DIR, "script.js"), "console.log('hello')");
      fs.writeFileSync(path.join(TEST_DIR, "style.css"), "body {}");
      
      const result = getFileTree(TEST_DIR);
      
      expect(result.children).toHaveLength(1);
      expect(result.children[0].name).toBe("readme.md");
    });
  });

  describe("matchFilePath", () => {
    let fileTree: Node;

    beforeEach(() => {
      // Create test file structure
      fs.writeFileSync(path.join(TEST_DIR, "index.md"), "# Root");
      fs.writeFileSync(path.join(TEST_DIR, "readme.md"), "# README");
      
      const docsDir = path.join(TEST_DIR, "docs");
      fs.mkdirSync(docsDir);
      fs.writeFileSync(path.join(docsDir, "index.md"), "# Docs Index");
      fs.writeFileSync(path.join(docsDir, "guide.md"), "# Guide");
      
      const apiDir = path.join(docsDir, "api");
      fs.mkdirSync(apiDir);
      fs.writeFileSync(path.join(apiDir, "reference.md"), "# API Reference");
      
      fileTree = getFileTree(TEST_DIR);
    });

    it("should return root content for empty path", () => {
      const result = matchFilePath([], fileTree);
      
      expect(result).not.toBeNull();
      expect(result?.type).toBe("markdown-file");
      if (result?.type === "markdown-file") {
        expect(result.filepath).toBe(path.join(TEST_DIR, "index.md"));
      }
    });

    it("should find files in root directory", () => {
      const result = matchFilePath(["readme.md"], fileTree);
      
      expect(result).not.toBeNull();
      expect(result?.type).toBe("markdown-file");
      if (result?.type === "markdown-file") {
        expect(result.filepath).toBe(path.join(TEST_DIR, "readme.md"));
      }
    });

    it("should find directory content", () => {
      const result = matchFilePath(["docs"], fileTree);
      
      expect(result).not.toBeNull();
      expect(result?.type).toBe("markdown-file");
      if (result?.type === "markdown-file") {
        expect(result.filepath).toBe(path.join(TEST_DIR, "docs", "index.md"));
      }
    });

    it("should find files in subdirectories", () => {
      const result = matchFilePath(["docs", "guide.md"], fileTree);
      
      expect(result).not.toBeNull();
      expect(result?.type).toBe("markdown-file");
      if (result?.type === "markdown-file") {
        expect(result.filepath).toBe(path.join(TEST_DIR, "docs", "guide.md"));
      }
    });

    it("should find deeply nested files", () => {
      const result = matchFilePath(["docs", "api", "reference.md"], fileTree);
      
      expect(result).not.toBeNull();
      expect(result?.type).toBe("markdown-file");
      if (result?.type === "markdown-file") {
        expect(result.filepath).toBe(path.join(TEST_DIR, "docs", "api", "reference.md"));
      }
    });

    it("should return null for non-existent files", () => {
      const result = matchFilePath(["nonexistent.md"], fileTree);
      expect(result).toBeNull();
    });

    it("should return null for non-existent directories", () => {
      const result = matchFilePath(["nonexistent", "file.md"], fileTree);
      expect(result).toBeNull();
    });

    it("should return null when trying to access file as directory", () => {
      const result = matchFilePath(["readme.md", "nonexistent.md"], fileTree);
      expect(result).toBeNull();
    });

    it("should handle directory without index.md", () => {
      // Create a directory without index.md
      const emptyDir = path.join(TEST_DIR, "empty");
      fs.mkdirSync(emptyDir);
      
      const newTree = getFileTree(TEST_DIR);
      const result = matchFilePath(["empty"], newTree);
      
      expect(result).not.toBeNull();
      expect(result?.type).toBe("directory-listing");
    });
  });

  describe("edge cases", () => {
    it("should handle directory with only non-markdown files", () => {
      fs.writeFileSync(path.join(TEST_DIR, "package.json"), "{}");
      fs.writeFileSync(path.join(TEST_DIR, "script.js"), "console.log('hello')");
      
      const result = getFileTree(TEST_DIR);
      
      expect(result.children).toHaveLength(0);
      expect(result.content.type).toBe("directory-listing");
    });

    it("should handle special characters in filenames", () => {
      fs.writeFileSync(path.join(TEST_DIR, "file with spaces.md"), "# Special");
      fs.writeFileSync(path.join(TEST_DIR, "file-with-dashes.md"), "# Dashes");
      fs.writeFileSync(path.join(TEST_DIR, "file_with_underscores.md"), "# Underscores");
      
      const result = getFileTree(TEST_DIR);
      
      expect(result.children).toHaveLength(3);
      const names = result.children.map(c => c.name).sort();
      expect(names).toEqual([
        "file with spaces.md",
        "file-with-dashes.md", 
        "file_with_underscores.md"
      ]);
    });

    it("should handle deeply nested directory structure", () => {
      const deep = path.join(TEST_DIR, "a", "b", "c", "d");
      fs.mkdirSync(deep, { recursive: true });
      fs.writeFileSync(path.join(deep, "deep.md"), "# Deep");
      
      const result = getFileTree(TEST_DIR);
      
      // Navigate through the nested structure
      let current = result;
      for (const dir of ["a", "b", "c", "d"]) {
        expect(current.children).toHaveLength(1);
        current = current.children[0];
        expect(current.name).toBe(dir);
        expect(current.type).toBe("directory");
      }
      
      expect(current.children).toHaveLength(1);
      expect(current.children[0].name).toBe("deep.md");
    });
  });
});