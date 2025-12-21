
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

  // Handle folder expansion and navigation
  document.addEventListener('click', function(e) {
    // Check if the click is directly on the chevron element
    if (e.target.classList.contains('folder-chevron')) {
      // Clicking directly on chevron - toggle expansion
      e.preventDefault();
      e.stopPropagation();
      
      const summary = e.target.closest('summary');
      const details = summary.closest('details');
      
      if (details) {
        details.open = !details.open;
        
        // Update chevron and expanded class
        const expanded = details.open;
        summary.classList.toggle('expanded', expanded);
        e.target.textContent = expanded ? '▼' : '▶';
      }
    }
    // For all other clicks on summary (including folder link), let default behavior happen
    // This means clicking anywhere else will follow the link
  });
});
