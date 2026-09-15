/**
 * CineMatch — AI Movie Recommender
 * Frontend JavaScript Logic
 * Handles: autocomplete search, API calls, rendering, animations
 */

// ── State ──────────────────────────────────────────────────
let allMovies = [];
let selectedMovie = '';
let autocompleteIndex = -1;

// ── DOM Elements ───────────────────────────────────────────
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const autocompleteEl = document.getElementById('autocomplete');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const selectedMovieEl = document.getElementById('selectedMovie');
const recsGrid = document.getElementById('recsGrid');
const heroSection = document.getElementById('hero');
const howItWorks = document.getElementById('howItWorks');
const searchAgainBtn = document.getElementById('searchAgainBtn');
const navbar = document.getElementById('navbar');

// ── Initialize ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadMovieList();
    initParticles();
    initScrollEffects();
    initEventListeners();
});

// ── Load Movie List for Autocomplete ───────────────────────
async function loadMovieList() {
    try {
        const response = await fetch('/api/movies');
        const data = await response.json();
        allMovies = data.movies || [];
        console.log(`✅ Loaded ${allMovies.length} movies for search`);
    } catch (error) {
        console.error('Failed to load movie list:', error);
    }
}

// ── Event Listeners ────────────────────────────────────────
function initEventListeners() {
    // Search input
    searchInput.addEventListener('input', handleSearchInput);
    searchInput.addEventListener('keydown', handleSearchKeydown);
    searchInput.addEventListener('focus', () => {
        if (searchInput.value.length >= 2) showAutocomplete(searchInput.value);
    });

    // Search button
    searchBtn.addEventListener('click', handleRecommend);

    // Search again
    searchAgainBtn.addEventListener('click', resetToSearch);

    // Quick tags
    document.querySelectorAll('.quick-tag').forEach(tag => {
        tag.addEventListener('click', () => {
            const movie = tag.dataset.movie;
            searchInput.value = movie;
            selectedMovie = movie;
            searchBtn.disabled = false;
            hideAutocomplete();
            handleRecommend();
        });
    });

    // Close autocomplete on outside click
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.search-container')) {
            hideAutocomplete();
        }
    });
}

// ── Search Input Handler ───────────────────────────────────
function handleSearchInput(e) {
    const query = e.target.value.trim();
    autocompleteIndex = -1;

    if (query.length < 2) {
        hideAutocomplete();
        searchBtn.disabled = true;
        selectedMovie = '';
        return;
    }

    showAutocomplete(query);
}

// ── Keyboard Navigation ────────────────────────────────────
function handleSearchKeydown(e) {
    const items = autocompleteEl.querySelectorAll('.autocomplete-item');

    if (e.key === 'ArrowDown') {
        e.preventDefault();
        autocompleteIndex = Math.min(autocompleteIndex + 1, items.length - 1);
        updateActiveItem(items);
    } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        autocompleteIndex = Math.max(autocompleteIndex - 1, 0);
        updateActiveItem(items);
    } else if (e.key === 'Enter') {
        e.preventDefault();
        if (autocompleteIndex >= 0 && items[autocompleteIndex]) {
            selectMovie(items[autocompleteIndex].textContent);
        } else if (selectedMovie) {
            handleRecommend();
        }
    } else if (e.key === 'Escape') {
        hideAutocomplete();
    }
}

function updateActiveItem(items) {
    items.forEach((item, i) => {
        item.classList.toggle('active', i === autocompleteIndex);
    });
    if (items[autocompleteIndex]) {
        items[autocompleteIndex].scrollIntoView({ block: 'nearest' });
    }
}

// ── Autocomplete ───────────────────────────────────────────
function showAutocomplete(query) {
    const lowerQuery = query.toLowerCase();
    const matches = allMovies
        .filter(movie => movie.toLowerCase().includes(lowerQuery))
        .slice(0, 12);

    if (matches.length === 0) {
        hideAutocomplete();
        return;
    }

    autocompleteEl.innerHTML = matches.map(movie => {
        const highlighted = highlightMatch(movie, query);
        return `<div class="autocomplete-item">${highlighted}</div>`;
    }).join('');

    autocompleteEl.classList.add('active');

    // Add click handlers
    autocompleteEl.querySelectorAll('.autocomplete-item').forEach(item => {
        item.addEventListener('click', () => {
            selectMovie(item.textContent);
        });
    });
}

function highlightMatch(text, query) {
    const idx = text.toLowerCase().indexOf(query.toLowerCase());
    if (idx === -1) return text;
    const before = text.slice(0, idx);
    const match = text.slice(idx, idx + query.length);
    const after = text.slice(idx + query.length);
    return `${before}<mark>${match}</mark>${after}`;
}

function hideAutocomplete() {
    autocompleteEl.classList.remove('active');
    autocompleteIndex = -1;
}

function selectMovie(title) {
    searchInput.value = title;
    selectedMovie = title;
    searchBtn.disabled = false;
    hideAutocomplete();
    searchInput.focus();
}

// ── Get Recommendations ────────────────────────────────────
async function handleRecommend() {
    if (!selectedMovie && searchInput.value.trim()) {
        selectedMovie = searchInput.value.trim();
    }
    if (!selectedMovie) return;

    // Show loading
    heroSection.style.display = 'none';
    howItWorks.style.display = 'none';
    resultsSection.style.display = 'none';
    loadingSection.style.display = 'block';

    try {
        const response = await fetch('/api/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ movie: selectedMovie })
        });

        const data = await response.json();

        if (data.error) {
            alert(`⚠️ ${data.error}`);
            resetToSearch();
            return;
        }

        renderResults(data);
    } catch (error) {
        console.error('Recommendation failed:', error);
        alert('❌ Failed to get recommendations. Please try again.');
        resetToSearch();
    }
}

// ── Render Results ─────────────────────────────────────────
function renderResults(data) {
    // Hide loading
    loadingSection.style.display = 'none';

    // Render selected movie
    const sel = data.selected;
    selectedMovieEl.innerHTML = `
        <div class="selected-card">
            <div class="selected-poster">
                ${sel.poster
                    ? `<img src="${sel.poster}" alt="${sel.title}" loading="lazy">`
                    : `<div class="poster-placeholder">🎬</div>`
                }
            </div>
            <div class="selected-info">
                <span class="selected-label">Your Selection</span>
                <h2 class="selected-title">${sel.title}</h2>
                <div class="selected-meta">
                    <span class="meta-badge rating">⭐ ${sel.rating}/10</span>
                    ${sel.year ? `<span class="meta-badge">📅 ${sel.year}</span>` : ''}
                </div>
                <div class="selected-genres">
                    ${sel.genres.map(g => `<span class="genre-tag">${g}</span>`).join('')}
                </div>
                <p class="selected-overview">${sel.overview || ''}</p>
            </div>
        </div>
    `;

    // Render recommendation cards
    recsGrid.innerHTML = data.recommendations.map((rec, i) => `
        <div class="movie-card" style="animation-delay: ${0.1 + i * 0.1}s">
            <div class="card-poster">
                ${rec.poster
                    ? `<img src="${rec.poster}" alt="${rec.title}" loading="lazy">`
                    : `<div class="poster-placeholder">🎥</div>`
                }
                <span class="card-rating">⭐ ${rec.rating}</span>
                <span class="card-match">${Math.round(rec.score * 100)}% Match</span>
            </div>
            <div class="card-body">
                <h3 class="card-title">${rec.title}</h3>
                ${rec.year ? `<p class="card-year">${rec.year}</p>` : ''}
                <div class="card-genres">
                    ${rec.genres.slice(0, 3).map(g => `<span class="card-genre">${g}</span>`).join('')}
                </div>
                <p class="card-overview">${rec.overview || ''}</p>
            </div>
        </div>
    `).join('');

    // Show results
    resultsSection.style.display = 'block';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Reset to Search ────────────────────────────────────────
function resetToSearch() {
    resultsSection.style.display = 'none';
    loadingSection.style.display = 'none';
    heroSection.style.display = 'block';
    howItWorks.style.display = 'block';

    searchInput.value = '';
    selectedMovie = '';
    searchBtn.disabled = true;

    window.scrollTo({ top: 0, behavior: 'smooth' });
    searchInput.focus();
}

// ── Particle Animation ─────────────────────────────────────
function initParticles() {
    const container = document.getElementById('particles');
    if (!container) return;

    for (let i = 0; i < 30; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = `${Math.random() * 100}%`;
        particle.style.animationDuration = `${8 + Math.random() * 15}s`;
        particle.style.animationDelay = `${Math.random() * 10}s`;
        particle.style.width = `${1 + Math.random() * 3}px`;
        particle.style.height = particle.style.width;
        particle.style.opacity = `${0.2 + Math.random() * 0.5}`;
        container.appendChild(particle);
    }
}

// ── Scroll Effects ─────────────────────────────────────────
function initScrollEffects() {
    window.addEventListener('scroll', () => {
        const scrollY = window.scrollY;

        // Navbar scroll effect
        if (navbar) {
            navbar.classList.toggle('scrolled', scrollY > 50);
        }
    });
}
