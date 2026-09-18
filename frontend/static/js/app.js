/**
 * Road Damage Intelligence System — Frontend JavaScript
 * Handles system status checks and active nav highlighting.
 */

// Highlight active nav link
document.addEventListener('DOMContentLoaded', () => {
    const path = window.location.pathname;
    const navMap = {
        '/': 'nav-home',
        '/image-analysis': 'nav-image',
        '/video-analysis': 'nav-video',
        '/history': 'nav-history',
        '/analytics': 'nav-analytics',
        '/model': 'nav-model',
        '/evaluation': 'nav-eval',
    };
    const activeId = navMap[path];
    if (activeId) {
        const el = document.getElementById(activeId);
        if (el) el.classList.add('active');
    }
});

/**
 * Check system status (API, model, database) and update badges.
 */
async function checkSystemStatus() {
    const apiBadge = document.getElementById('status-api');
    const modelBadge = document.getElementById('status-model');
    const dbBadge = document.getElementById('status-db');

    try {
        const response = await fetch('/health');
        const data = await response.json();

        if (apiBadge) {
            apiBadge.textContent = 'Online';
            apiBadge.className = 'status-badge status-completed';
        }

        if (modelBadge) {
            if (data.model_loaded) {
                modelBadge.textContent = 'Loaded';
                modelBadge.className = 'status-badge status-completed';
            } else {
                modelBadge.textContent = 'Not Loaded';
                modelBadge.className = 'status-badge status-failed';
            }
        }

        if (dbBadge) {
            if (data.database_connected) {
                dbBadge.textContent = 'Connected';
                dbBadge.className = 'status-badge status-completed';
            } else {
                dbBadge.textContent = 'Error';
                dbBadge.className = 'status-badge status-failed';
            }
        }
    } catch (err) {
        if (apiBadge) {
            apiBadge.textContent = 'Offline';
            apiBadge.className = 'status-badge status-failed';
        }
        if (modelBadge) {
            modelBadge.textContent = 'Unknown';
            modelBadge.className = 'status-badge status-pending';
        }
        if (dbBadge) {
            dbBadge.textContent = 'Unknown';
            dbBadge.className = 'status-badge status-pending';
        }
    }
}
