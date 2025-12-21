interface HeaderProps {
  filename: string;
}

export function Header({ filename }: HeaderProps): JSX.Element {
  return (
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
  );
}