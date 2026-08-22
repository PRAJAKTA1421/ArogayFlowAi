const toggle = document.querySelector('.menu-toggle');
const links = document.querySelector('.nav-links');
if (toggle) toggle.addEventListener('click', () => links.classList.toggle('open'));
const dashLinks = document.querySelectorAll('.sidebar nav a');
const dashboardRoutes = {
  'PHC Network': '/phc-network',
  'Medicine Inventory': '/medicine-inventory',
  'Beds Overview': '/beds-overview',
  'Medical Staff': '/medical-staff',
  'AI Predictions': '/ai-predictions',
  'Resource Transfer': '/resource-transfer',
};
document.querySelectorAll('.sidebar nav a').forEach((link) => {
  const label = link.querySelector('span')?.textContent.trim();
  if (label === 'Alerts & Notifications' || label === 'Reports & Analytics') {
    link.remove();
    return;
  }
  if (dashboardRoutes[label]) link.href = dashboardRoutes[label];
});
if (dashLinks.length && location.pathname === '/dashboard') dashLinks[1].addEventListener('click', (event) => { event.preventDefault(); location.href = '/phc-network'; });
if (dashLinks.length && location.pathname === '/dashboard') dashLinks[2].addEventListener('click', (event) => { event.preventDefault(); location.href = '/medicine-inventory'; });
if (dashLinks.length && location.pathname === '/dashboard') dashLinks[3].addEventListener('click', (event) => { event.preventDefault(); location.href = '/beds-overview'; });

// Keep report-page sidebars aligned with the Dashboard navigation.
if (['/medicine-inventory', '/beds-overview'].includes(location.pathname)) {
  const sidebar = document.querySelector('.sidebar');
  const nav = sidebar?.querySelector('nav');
  if (sidebar && nav && !nav.querySelector('.dashboard-extra-link')) {
    nav.insertAdjacentHTML('beforeend', `
      <a class="dashboard-extra-link" href="#">&#9633; <span>My PHC</span></a>
      <a class="dashboard-extra-link" href="/settings">&#9881; <span>Settings</span></a>
    `);
    sidebar.insertAdjacentHTML('beforeend', `
      <div class="health-score"><small>AI Health Resilience Score</small><div class="score">82<em>/100</em></div><strong>Good</strong><p>Network Health Status</p></div>
      <a class="logout" href="/">&#10132; Logout</a>
    `);
  }
}

// Personalize the command header for the PHC that has just signed in.
const currentPhc = document.body.dataset.currentPhc;
if (currentPhc) {
  const locationButton = document.querySelector('.dash-head .location');
  const profile = document.querySelector('.dash-head .profile');
  if (locationButton) locationButton.textContent = `${currentPhc}  ▾`;
  if (profile) {
    const manager = document.body.dataset.currentManager || 'PHC Manager';
    profile.innerHTML = `<span class="profile-avatar" aria-hidden="true">&#128100;</span><span>${manager}<small>${currentPhc}</small></span>⌄`;
  }
}

// The header notification control should always display a bell, not a music symbol.
document.querySelectorAll('.dash-head .bell').forEach((bell) => {
  const badge = bell.querySelector('i');
  bell.replaceChildren(document.createTextNode('🔔'));
  if (badge) bell.append(badge);
});
