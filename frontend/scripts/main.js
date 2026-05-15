/**
 * main.js — shared utilities across all pages
 * - Navbar scroll effect
 * - Dark / Light theme toggle (persisted in localStorage)
 * - Smooth scroll for anchor links
 * - Scroll-reveal animations (IntersectionObserver)
 * - Counter animation for hero stats
 * - Utility functions shared across pages
 */

// ── Theme ─────────────────────────────────────────────────────────────
const html      = document.documentElement;
const toggle    = document.getElementById('theme-toggle');
const themeIcon = document.getElementById('theme-icon');

function applyTheme(theme) {
  html.setAttribute('data-theme', theme);
  // Tailwind dark mode uses 'dark' class on <html>
  if (theme === 'dark') {
    html.classList.add('dark');
    if (themeIcon) themeIcon.textContent = '🌙';
  } else {
    html.classList.remove('dark');
    if (themeIcon) themeIcon.textContent = '☀️';
  }
  localStorage.setItem('fs-theme', theme);
}

// Restore saved theme on load
const savedTheme = localStorage.getItem('fs-theme') || 'dark';
applyTheme(savedTheme);

if (toggle) {
  toggle.addEventListener('click', () => {
    const current = html.getAttribute('data-theme');
    applyTheme(current === 'dark' ? 'light' : 'dark');
  });
}

// ── Navbar scroll glass effect ────────────────────────────────────────
const navbar = document.getElementById('navbar');
if (navbar) {
  let ticking = false;
  window.addEventListener('scroll', () => {
    if (!ticking) {
      requestAnimationFrame(() => {
        if (window.scrollY > 20) navbar.classList.add('scrolled');
        else                     navbar.classList.remove('scrolled');
        ticking = false;
      });
      ticking = true;
    }
  }, { passive: true });
}

// ── Smooth scroll for in-page anchors ────────────────────────────────
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', e => {
    const target = document.querySelector(anchor.getAttribute('href'));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});

// ── Animate elements on scroll (IntersectionObserver) ────────────────
const observerOpts = { threshold: 0.12, rootMargin: '0px 0px -40px 0px' };
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = '1';
      entry.target.classList.add('animate-fade-in');
      observer.unobserve(entry.target);
    }
  });
}, observerOpts);

// Only observe cards that are already visible (not inside hidden sections)
document.querySelectorAll('.glass-card, .stat-card, .feature-card, .step-card')
  .forEach(el => {
    // Skip if inside a hidden parent (e.g. results-section)
    if (el.closest('.hidden') || el.closest('[id$="-section"]')) return;
    el.style.opacity = '0';
    observer.observe(el);
  });

// ── Counter animation for hero stats ─────────────────────────────────
function animateCounter(el, target, duration = 1800) {
  const start = performance.now();
  const isFloat = target % 1 !== 0;

  function update(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    // Ease-out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = eased * target;
    el.textContent = isFloat ? current.toFixed(2) : Math.round(current).toLocaleString();
    if (progress < 1) requestAnimationFrame(update);
    else el.textContent = isFloat ? target.toFixed(2) : target.toLocaleString();
  }
  requestAnimationFrame(update);
}

// Observe counter elements
const counterObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const target = parseFloat(entry.target.dataset.target);
      if (!isNaN(target)) animateCounter(entry.target, target);
      counterObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.5 });

document.querySelectorAll('.counter[data-target]').forEach(el => {
  counterObserver.observe(el);
});

// ── Utility: format number with commas ───────────────────────────────
function fmtNum(n) {
  return Number(n).toLocaleString();
}

// ── Utility: risk badge HTML ──────────────────────────────────────────
function riskBadgeHtml(label, confidence) {
  const cls = label === 'FRAUD'
    ? (confidence === 'HIGH' ? 'badge-fraud' : 'badge-medium')
    : 'badge-safe';
  const icon = label === 'FRAUD' ? '⚠️' : '✅';
  return `<span class="risk-badge ${cls}">${icon} ${label}</span>`;
}
