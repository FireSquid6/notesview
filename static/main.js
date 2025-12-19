
document.addEventListener('DOMContentLoaded', function() {
  // Sidebar toggle functionality
  const sidebarToggle = document.getElementById('sidebar-toggle');
  const sidebar = document.getElementById('sidebar');

  // Function to check if we're in mobile view
  function isMobileView() {
    return window.innerWidth <= 1024;
  }

  // Function to update sidebar state based on screen size
  function updateSidebarForResize() {
    if (!isMobileView()) {
      // Desktop view - ensure sidebar is visible and reset toggle button
      sidebar.classList.remove('collapsed');
      const icon = sidebarToggle?.querySelector('.toggle-icon');
      if (icon) {
        icon.textContent = '☰';
      }
    }
  }

  // Initial check
  updateSidebarForResize();

  // Add resize listener
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

  // Folder toggle functionality
  document.querySelectorAll('.folder-item').forEach(folder => {
    folder.addEventListener('click', function(e) {
      e.preventDefault();
      const expanded = this.classList.contains('expanded');

      // Toggle folder state
      this.classList.toggle('expanded');

      // Toggle chevron
      const chevron = this.querySelector('.folder-chevron');
      if (chevron) {
        chevron.textContent = expanded ? '▶' : '▼';
      }

      // Show/hide child files
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
