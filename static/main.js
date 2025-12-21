
document.addEventListener('DOMContentLoaded', function() {
  const sidebarToggle = document.getElementById('sidebar-toggle');
  const sidebar = document.getElementById('sidebar');

  function isMobileView() {
    return window.innerWidth <= 1024;
  }

  function updateSidebarForResize() {
    if (!isMobileView()) {
      sidebar.classList.remove('collapsed');
      const icon = sidebarToggle?.querySelector('.toggle-icon');
      if (icon) {
        icon.textContent = '☰';
      }
    }
  }

  updateSidebarForResize();
  window.addEventListener('resize', updateSidebarForResize);

  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener('click', function() {
      sidebar.classList.toggle('collapsed');
      const icon = sidebarToggle.querySelector('.toggle-icon');
      if (icon) {
        icon.textContent = sidebar.classList.contains('collapsed') ? '☰' : '✕';
      }
    });
  }

  document.querySelectorAll('.folder-item').forEach(folder => {
    folder.addEventListener('click', function(e) {
      e.preventDefault();
      const expanded = this.classList.contains('expanded');

      this.classList.toggle('expanded');

      const chevron = this.querySelector('.folder-chevron');
      if (chevron) {
        chevron.textContent = expanded ? '▶' : '▼';
      }

      document.querySelectorAll(`[data-parent="\${folderName}"]`).forEach(child => {
        if (expanded) {
          child.style.display = 'none';
        } else {
          child.style.display = 'flex';
        }
      });
    });
  });
});
