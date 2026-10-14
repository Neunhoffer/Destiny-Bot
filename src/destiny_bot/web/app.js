/*
 * Athena: the browser reader.
 *
 * Plain ES2020 module with no build step and no third-party code. It talks only
 * to the local Athena server (docs/API.md). Book, AI and server text is
 * untrusted, so it is only ever inserted as text (textContent / text nodes),
 * never as HTML.
 *
 * Sections:
 *   1. Helpers: DOM building, icons, storage, formatting
 *   2. Server API, toasts and screen-reader announcements
 *   3. App state, engines and voices
 *   4. Appearance: theme, text size, highlight color
 *   5. Audio output: TTS clip cache, Web Audio, <audio> fallback, browser speech
 *   6. Gapless scheduler for server voices (Web Audio clock drives the highlight)
 *   7. Player: play/pause/seek, sequential path, speed, sleep timer, progress
 *   8. Reader: blocks, sentence spans, highlight, auto-scroll, selection bubble
 *   9. Library, import, paste, contents and bookmarks
 *  10. Study helper (AI)
 *  11. Dialogs, panels and tabs
 *  12. Settings
 *  13. Export ("Save as audio")
 *  14. Keyboard shortcuts and start-up
 */

// =====================================================================
// 1. Helpers
// =====================================================================

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
const enc = encodeURIComponent;
const clamp = (value, lo, hi) => Math.min(hi, Math.max(lo, value));
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

function debounce(fn, ms) {
  let timer = 0;
  const run = (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
  run.cancel = () => clearTimeout(timer);
  return run;
}

/** Build an element. Children may be nodes, strings (added as text) or arrays. */
function el(tag, props = null, ...children) {
  const node = document.createElement(tag);
  if (props) {
    for (const [key, value] of Object.entries(props)) {
      if (value === undefined || value === null || value === false) continue;
      if (key === 'class') node.className = value;
      else if (key === 'text') node.textContent = value;
      else if (key === 'dataset') Object.assign(node.dataset, value);
      else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
      else if (value === true) node.setAttribute(key, '');
      else node.setAttribute(key, String(value));
    }
  }
  appendAll(node, children);
  return node;
}

function appendAll(node, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

/** Plain text with blank-line paragraphs and single-line breaks, built safely. */
function textParagraphs(text, cls = '') {
  const frag = document.createDocumentFragment();
  const paras = String(text || '').trim().split(/\n\s*\n/);
  for (const para of paras) {
    const p = el('p', cls ? { class: cls } : null);
    para.split('\n').forEach((line, i) => {
      if (i) p.append(el('br'));
      p.append(line);
    });
    frag.append(p);
  }
  return frag;
}

const local = {
  get(key, fallback) {
    try {
      const raw = window.localStorage.getItem(`destiny.${key}`);
      return raw === null ? fallback : JSON.parse(raw);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      window.localStorage.setItem(`destiny.${key}`, JSON.stringify(value));
    } catch {
      /* private browsing or storage full: preferences just won't stick */
    }
  },
  clearAll() {
    try {
      for (const key of Object.keys(window.localStorage)) {
        if (key.startsWith('destiny.')) window.localStorage.removeItem(key);
      }
    } catch {
      /* ignore */
    }
  },
};

function truncate(text, max) {
  const clean = String(text || '').replace(/\s+/g, ' ').trim();
  return clean.length > max ? `${clean.slice(0, max - 1).trimEnd()}…` : clean;
}

function formatSize(bytes) {
  if (!Number.isFinite(bytes)) return '';
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const formatSpeed = (value) => `${value.toFixed(2)}×`;
const plural = (n, one, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;
const norm = (text) => String(text || '').replace(/\s+/g, ' ').trim().toLowerCase();

const TYPE_LABELS = {
  pdf: 'PDF', epub: 'EPUB', docx: 'Word', pptx: 'Slides', html: 'Web page', txt: 'Text',
  md: 'Markdown', image: 'Photo', paste: 'Pasted text',
};
const COVER_LABELS = {
  pdf: 'PDF', epub: 'EPUB', docx: 'DOC', pptx: 'PPT', html: 'WEB', txt: 'TXT', md: 'MD', image: 'IMG', paste: 'TXT',
};
const MAX_UPLOAD_BYTES = 300 * 1024 * 1024;

// ----- icons (inline SVG built from trusted constants)

const SVG_NS = 'http://www.w3.org/2000/svg';
const P = (d) => ['path', { d }];
const S = (d) => ['path', { d, fill: 'currentColor' }];
const C = (cx, cy, r, filled = false) => ['circle', filled ? { cx, cy, r, fill: 'currentColor', stroke: 'none' } : { cx, cy, r }];
const R = (x, y, width, height, rx, filled = false) => ['rect', filled ? { x, y, width, height, rx, fill: 'currentColor', stroke: 'none' } : { x, y, width, height, rx }];

const ICONS = {
  menu: [P('M4 7h16M4 12h16M4 17h16')],
  close: [P('M6 6l12 12M18 6L6 18')],
  plus: [P('M12 5v14M5 12h14')],
  paste: [R(8, 3, 8, 4, 1.2), P('M16 5h2a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h2'), P('M8.5 12h7M8.5 16h4.5')],
  upload: [P('M12 15V4M7.5 8.5L12 4l4.5 4.5'), P('M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3')],
  trash: [P('M4 7h16M10 11v6M14 11v6'), P('M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4.5h6V7')],
  lock: [R(5, 10.5, 14, 10, 2.2), P('M8 10.5V7.5a4 4 0 0 1 8 0v3'), P('M12 14.5v2')],
  cloud: [P('M7 18.5a4.5 4.5 0 0 1-.6-8.96A6 6 0 0 1 17.9 9a4.75 4.75 0 0 1-.4 9.5z')],
  settings: [P('M4 7h9M17 7h3M4 17h3M11 17h9'), C(15, 7, 2.2), C(9, 17, 2.2)],
  keyboard: [R(2.5, 6, 19, 12, 2.5), P('M6.5 10h.01M10 10h.01M14 10h.01M17.5 10h.01M7.5 14h9')],
  bulb: [P('M9 17.5h6M10 20.5h4'), P('M12 3.5a5.75 5.75 0 0 0-3.4 10.4c.55.42.9 1.06.9 1.76v.34h5v-.34c0-.7.35-1.34.9-1.76A5.75 5.75 0 0 0 12 3.5z')],
  play: [S('M8 5.5v13l10.5-6.5z')],
  pause: [R(6.5, 5, 3.6, 14, 1.2, true), R(13.9, 5, 3.6, 14, 1.2, true)],
  prev: [S('M11 6.5v11L4.5 12z'), S('M19.5 6.5v11L13 12z')],
  next: [S('M13 6.5v11l6.5-5.5z'), S('M4.5 6.5v11l6.5-5.5z')],
  prevSection: [P('M6 6v12'), S('M18 6.5v11L9.5 12z')],
  nextSection: [P('M18 6v12'), S('M6 6.5v11l8.5-5.5z')],
  bookmark: [P('M7 3.5h10a1 1 0 0 1 1 1V20.5l-6-4-6 4V4.5a1 1 0 0 1 1-1z')],
  bookmarkFilled: [S('M7 3.5h10a1 1 0 0 1 1 1V20.5l-6-4-6 4V4.5a1 1 0 0 1 1-1z')],
  moon: [P('M19.5 14.5A7.5 7.5 0 0 1 9.5 4.5a7.5 7.5 0 1 0 10 10z')],
  voice: [C(9, 8, 3.2), P('M3.5 19.5c.7-3 2.9-4.8 5.5-4.8s4.8 1.8 5.5 4.8'), P('M16.5 6.5a4 4 0 0 1 0 5M19 4a7.5 7.5 0 0 1 0 10')],
  speaker: [P('M4 9.5h3l4.5-4v13l-4.5-4H4a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1z'), P('M15.5 9a4.2 4.2 0 0 1 0 6M18.5 6.5a8 8 0 0 1 0 11')],
  stop: [R(6.5, 6.5, 11, 11, 2, true)],
  download: [P('M12 4v11M7.5 10.5L12 15l4.5-4.5M5 19.5h14')],
  book: [P('M12 6.5C10 5 7 4.5 4 5v13c3-.5 6 0 8 1.5 2-1.5 5-2 8-1.5V5c-3-.5-6 0-8 1.5zM12 6.5v13')],
  check: [P('M5 12.5l4.5 4.5L19 7.5')],
  alert: [P('M12 4.5l8.5 15h-17z'), P('M12 10v4'), P('M12 16.8v.2')],
  info: [C(12, 12, 8.5), P('M12 11v5.5'), P('M12 7.8v.2')],
  image: [R(3.5, 5, 17, 14, 2), C(9, 10, 1.6), P('M20.5 15.5l-5-5-9 8.5')],
  summary: [P('M5 6h14M5 10h14M5 14h10M5 18h6')],
  quiz: [R(4, 4, 16, 16, 3.5), P('M8.5 12.2l2.4 2.4 4.6-5')],
  table: [R(3.5, 5, 17, 14, 2), P('M3.5 10h17M3.5 14.5h17M10 5v14')],
  sigma: [P('M17 5H7l5.5 7L7 19h10')],
  send: [P('M4.5 12L20 4.5 15 20l-3-6.5z'), P('M12 13.5l8-9')],
  copy: [R(8.5, 8.5, 11, 11, 2), P('M15.5 8.5V6.5a2 2 0 0 0-2-2h-7a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h2')],
  palette: [P('M12 3.5a8.5 8.5 0 1 0 0 17c1.1 0 1.7-.8 1.7-1.6 0-.5-.2-.9-.5-1.2-.3-.3-.5-.7-.5-1.2 0-.9.7-1.6 1.6-1.6h1.9a4.3 4.3 0 0 0 4.3-4.3c0-4-3.8-7.1-8.5-7.1z'), C(7.6, 11, 1.1, true), C(10, 7.4, 1.1, true), C(14.6, 7.4, 1.1, true)],
  file: [P('M7 3.5h7l4.5 4.5v12a.5.5 0 0 1-.5.5H7a.5.5 0 0 1-.5-.5V4a.5.5 0 0 1 .5-.5z'), P('M14 3.5V8h4.5')],
  music: [P('M9 18V6l10-2v12'), C(6.5, 18, 2.5), C(16.5, 16, 2.5)],
  chat: [P('M5 5h14a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1h-7l-4 3.5V16H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1z')],
};

function icon(name) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('class', 'icon');
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('focusable', 'false');
  for (const [tag, attrs] of ICONS[name] || []) {
    const shape = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs)) shape.setAttribute(key, String(value));
    svg.append(shape);
  }
  return svg;
}

function setIcon(holder, name) {
  if (!holder || (holder.dataset.icon === name && holder.firstElementChild)) return;
  holder.dataset.icon = name;
  holder.replaceChildren(icon(name));
}

function hydrateIcons(root = document) {
  for (const holder of $$('[data-icon]', root)) {
    if (!holder.firstElementChild) holder.append(icon(holder.dataset.icon));
  }
}

// =====================================================================
// 2. Server API, toasts and announcements
// =====================================================================

const TOKEN = $('meta[name="destiny-token"]')?.getAttribute('content') || '';

class ApiError extends Error {
  constructor(status, detail) {
    super(detail);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

const STATUS_MESSAGES = {
  0: "Can't reach Athena. Make sure it's still running, then try again.",
  401: 'Athena was restarted. Reload this page to continue.',
  403: "That isn't allowed right now. Check Settings → Privacy & AI.",
  404: "That couldn't be found. It may have been deleted.",
  413: 'That file is too big to import (the limit is 300 MB).',
  500: 'Something went wrong inside Athena. Please try again.',
};

function detailFrom(status, data) {
  const detail = data && data.detail;
  if (typeof detail === 'string' && detail.trim()) return detail.trim();
  if (Array.isArray(detail) && detail.length) {
    return detail.map((d) => (d && d.msg ? String(d.msg) : String(d))).join('; ');
  }
  return STATUS_MESSAGES[status] || `Something went wrong (error ${status}). Please try again.`;
}

async function api(path, { method = 'GET', body, signal, keepalive = false, raw = false } = {}) {
  const headers = { 'X-Destiny-Token': TOKEN };
  const init = { method, headers, signal, keepalive, cache: 'no-store', credentials: 'same-origin' };
  if (body !== undefined) {
    headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(path, init);
  } catch (err) {
    if (err && err.name === 'AbortError') throw err;
    throw new ApiError(0, STATUS_MESSAGES[0]);
  }
  if (!res.ok) {
    let data = null;
    try {
      data = await res.json();
    } catch {
      /* not JSON */
    }
    if (res.status === 401) showSessionExpired();
    throw new ApiError(res.status, detailFrom(res.status, data));
  }
  if (raw) return res;
  if (res.status === 204) return null;
  return res.json();
}

function errorText(err) {
  if (!err) return 'Something went wrong.';
  if (err instanceof ApiError) return err.detail;
  return err.message || String(err);
}

function toast(message, { kind = 'info', action = null, timeout = null } = {}) {
  const box = $('#toasts');
  const iconName = kind === 'error' ? 'alert' : kind === 'success' ? 'check' : 'info';
  const node = el('div', { class: `toast toast-${kind}`, role: kind === 'error' ? 'alert' : 'status' },
    icon(iconName), el('div', { class: 'toast-msg', text: message }));
  let timer = 0;
  const close = () => {
    clearTimeout(timer);
    node.classList.add('is-leaving');
    setTimeout(() => node.remove(), 220);
  };
  if (action) {
    node.append(el('button', {
      type: 'button', class: 'btn btn-small btn-secondary',
      onclick: () => { close(); action.run(); },
    }, action.label));
  }
  node.append(el('button', { type: 'button', class: 'icon-btn icon-btn-small', 'aria-label': 'Dismiss', onclick: close }, icon('close')));
  box.append(node);
  while (box.children.length > 4) box.firstElementChild.remove();
  const ms = timeout ?? (action || kind === 'error' ? 12000 : 5000);
  const arm = (delay) => { clearTimeout(timer); timer = setTimeout(close, delay); };
  arm(ms);
  node.addEventListener('mouseenter', () => clearTimeout(timer));
  node.addEventListener('mouseleave', () => arm(4000));
  node.addEventListener('focusin', () => clearTimeout(timer));
  return close;
}

let announceTimer = 0;
function announce(text) {
  const live = $('#live');
  clearTimeout(announceTimer);
  live.textContent = '';
  announceTimer = setTimeout(() => { live.textContent = text; }, 80);
}

let sessionExpiredShown = false;
function showSessionExpired() {
  if (sessionExpiredShown) return;
  sessionExpiredShown = true;
  hardStop();
  player.playing = false;
  updatePlayUI();
  $('#banners').prepend(el('div', { class: 'notice notice-error', role: 'alert' },
    icon('alert'),
    el('div', { class: 'notice-body' },
      el('p', { class: 'notice-title', text: 'Please reload this page' }),
      el('p', { text: 'Athena was restarted, so this page lost its connection. Reloading keeps all your books and progress.' }),
      el('div', { class: 'notice-actions' },
        el('button', { type: 'button', class: 'btn btn-primary btn-small', onclick: () => window.location.reload() }, 'Reload page')))));
}

// =====================================================================
// 3. App state, engines and voices
// =====================================================================

const state = {
  status: null, // GET /api/status
  settings: null, // Settings.public_dict()
  speed: 1,
  books: [],
  book: null, // BookSummary of the open book
  toc: [],
  warnings: [],
  dismissedWarnings: new Set(),
  section: null, // {section, blocks, utterances, prev_id, next_id}
  index: 0, // current utterance
  bookmarks: [],
  serverVoices: {}, // engine name -> [Voice]
  navSeq: 0,
};

const ENGINE_BLURBS = {
  piper: 'Natural-sounding voices that run entirely on this computer.',
  system: 'The voices built into your computer. Works offline.',
  edge: 'Very natural Microsoft voices. Needs the internet.',
  browser: 'Voices built into this web browser.',
};

function engineInfo(name) {
  return state.status?.engines?.find((e) => e.name === name) || null;
}

function engineAllowed(info) {
  return !!info && info.available && (!info.online || !!state.settings?.online_tts_allowed);
}

function hasSpeechSynthesis() {
  return 'speechSynthesis' in window && typeof window.SpeechSynthesisUtterance === 'function';
}

/** The engine that will actually speak: the saved choice when usable, else the server's pick. */
function activeEngine() {
  const pref = state.settings?.tts_engine || 'auto';
  if (pref === 'browser') return 'browser';
  if (pref !== 'auto' && engineAllowed(engineInfo(pref))) return pref;
  return state.status?.resolved_engine || 'browser';
}

function activeVoice(engine = activeEngine()) {
  const chosen = state.settings?.voices?.[engine] || '';
  if (engine === 'browser') return pickBrowserVoice(chosen)?.voiceURI || '';
  const list = state.serverVoices[engine] || [];
  if (chosen && (!list.length || list.some((v) => v.id === chosen))) return chosen;
  return engineInfo(engine)?.default_voice || list[0]?.id || '';
}

function engineLabel(name) {
  return engineInfo(name)?.label || (name === 'browser' ? 'Browser voice' : name);
}

function browserVoicesAll() {
  return hasSpeechSynthesis() ? window.speechSynthesis.getVoices() : [];
}

/** Browser voices that keep text on this computer, plus online ones when allowed. */
function usableBrowserVoices() {
  const all = browserVoicesAll();
  if (state.settings?.online_tts_allowed) return all;
  return all.filter((v) => v.localService !== false);
}

function pickBrowserVoice(uri) {
  const voices = usableBrowserVoices();
  if (!voices.length) return null;
  const lang = (navigator.language || 'en').slice(0, 2).toLowerCase();
  return voices.find((v) => v.voiceURI === uri)
    || voices.find((v) => v.default && v.lang?.toLowerCase().startsWith(lang))
    || voices.find((v) => v.lang?.toLowerCase().startsWith(lang))
    || voices[0];
}

async function refreshStatus() {
  try {
    state.status = await api('/api/status');
  } catch (err) {
    toast(errorText(err), { kind: 'error' });
  }
  await refreshServerVoices();
  renderVoicePicker();
  renderPrivacyChip();
  renderStudy();
}

async function refreshServerVoices() {
  const engines = (state.status?.engines || []).filter((e) => e.name !== 'browser' && engineAllowed(e));
  const results = await Promise.all(engines.map(async (e) => {
    try {
      const data = await api(`/api/voices?engine=${enc(e.name)}`);
      return [e.name, data.voices || []];
    } catch {
      return [e.name, []];
    }
  }));
  state.serverVoices = Object.fromEntries(results);
}

async function saveSettings(changes) {
  try {
    state.settings = await api('/api/settings', { method: 'PUT', body: changes });
    return true;
  } catch (err) {
    toast(errorText(err), { kind: 'error' });
    return false;
  }
}

// =====================================================================
// 4. Appearance
// =====================================================================

const HIGHLIGHTS = [
  ['sun', 'Sunny'], ['mint', 'Mint'], ['sky', 'Sky'], ['rose', 'Rose'], ['lavender', 'Lavender'],
];

function applyAppearance(overrides = {}) {
  const root = document.documentElement;
  const theme = overrides.theme ?? state.settings?.theme ?? local.get('theme', 'system');
  if (theme === 'light' || theme === 'dark') root.dataset.theme = theme;
  else delete root.dataset.theme;
  local.set('theme', theme);
  const scale = clamp(Number(overrides.fontScale ?? state.settings?.font_scale ?? 1) || 1, 0.7, 2);
  root.style.setProperty('--font-scale', String(scale));
  const highlight = local.get('highlight', 'sun');
  if (highlight && highlight !== 'sun') root.dataset.highlight = highlight;
  else delete root.dataset.highlight;
  root.classList.toggle('comfortable', !!local.get('comfortable', false));
}

function renderPrivacyChip() {
  const s = state.settings;
  const online = !!(s && (s.online_tts_allowed || s.ai_enabled));
  const chip = $('#privacy-chip');
  chip.classList.toggle('is-online', online);
  setIcon($('#privacy-chip-icon'), online ? 'cloud' : 'lock');
  $('#privacy-chip-long').textContent = online ? 'Online features on' : 'Private: everything stays on this computer';
  $('#privacy-chip-short').textContent = online ? 'Online' : 'Offline';
  chip.setAttribute('aria-label', online
    ? 'Online features are on. Open privacy settings.'
    : 'Private: everything stays on this computer. Open privacy settings.');
  chip.title = online ? 'Some text is sent online. Click to review.' : 'Nothing leaves this computer. Click for privacy settings.';
}

// =====================================================================
// 5. Audio output
// =====================================================================

const FADE_SECONDS = 0.008;
const SECTION_GAP_SECONDS = 0.5;

/** Natural pauses: shortened a little at high speed, never lengthened at low speed. */
function pauseSeconds(utterance, rate) {
  return (utterance.pause_after_ms || 0) / Math.max(1, rate * 0.8) / 1000;
}

/** The one reusable <audio> element, used only when Web Audio isn't available. */
const audio = new Audio();
audio.preload = 'auto';
audio.preservesPitch = true;
audio.mozPreservesPitch = true;
audio.webkitPreservesPitch = true;
let audioWaiter = null;
audio.addEventListener('ended', () => {
  const waiter = audioWaiter;
  audioWaiter = null;
  waiter?.resolve('ended');
});
audio.addEventListener('error', () => {
  const waiter = audioWaiter;
  audioWaiter = null;
  waiter?.reject(new Error("This audio couldn't be played. Try another voice in Settings → Voice."));
});

function playUrl(url) {
  return new Promise((resolve, reject) => {
    audioWaiter?.resolve('cancelled');
    const waiter = { resolve, reject };
    audioWaiter = waiter;
    audio.defaultPlaybackRate = 1; // the voice engine already produced the speed
    audio.src = url;
    audio.playbackRate = 1;
    audio.preservesPitch = true;
    audio.play().catch((err) => {
      if (audioWaiter !== waiter) return;
      audioWaiter = null;
      reject(err);
    });
  });
}

function stopAudioElement() {
  const waiter = audioWaiter;
  audioWaiter = null;
  if (!audio.paused) audio.pause();
  waiter?.resolve('cancelled');
}

/** Fetched speech clips keyed by engine/voice/rate/text. LRU with bounded parallel prefetch. */
const clips = {
  entries: new Map(),
  waiting: [],
  running: 0,
  maxParallel: 2,
  limit: 40,

  key(text, engine, voice, rate) {
    return [engine, voice || '', rate.toFixed(2), text].join('␟');
  },

  get(text, engine, voice, rate, { urgent = false } = {}) {
    const key = this.key(text, engine, voice, rate);
    let entry = this.entries.get(key);
    if (entry) {
      this.entries.delete(key);
      this.entries.set(key, entry);
      if (urgent && !entry.started) {
        entry.urgent = true;
        this.start(entry);
      }
      return entry;
    }
    entry = {
      key, text, engine, voice, rate, urgent, started: false, pinned: 0,
      controller: new AbortController(), blob: null, url: null,
    };
    entry.promise = new Promise((resolve, reject) => {
      entry.resolve = resolve;
      entry.reject = reject;
    });
    entry.promise.catch(() => {}); // prefetch failures are reported when the clip is needed
    this.entries.set(key, entry);
    if (urgent) this.start(entry);
    else {
      this.waiting.push(entry);
      this.pump();
    }
    this.evict();
    return entry;
  },

  start(entry) {
    if (entry.started) return;
    entry.started = true;
    const queued = this.waiting.indexOf(entry);
    if (queued >= 0) this.waiting.splice(queued, 1);
    const counted = !entry.urgent;
    if (counted) this.running += 1;
    api('/api/tts', {
      method: 'POST',
      body: { text: entry.text, engine: entry.engine, voice: entry.voice || undefined, rate: entry.rate },
      signal: entry.controller.signal,
      raw: true,
    })
      .then((res) => res.blob())
      .then((blob) => {
        entry.blob = blob;
        entry.resolve(blob);
      }, (err) => {
        if (this.entries.get(entry.key) === entry) this.entries.delete(entry.key);
        entry.reject(err);
      })
      .finally(() => {
        if (counted) {
          this.running -= 1;
          this.pump();
        }
      });
  },

  pump() {
    while (this.running < this.maxParallel && this.waiting.length) this.start(this.waiting[0]);
  },

  evict() {
    for (const entry of this.entries.values()) {
      if (this.entries.size <= this.limit) break;
      if (!entry.pinned) this.drop(entry);
    }
  },

  drop(entry) {
    if (this.entries.get(entry.key) === entry) this.entries.delete(entry.key);
    const queued = this.waiting.indexOf(entry);
    if (queued >= 0) this.waiting.splice(queued, 1);
    if (!entry.blob) {
      entry.controller.abort();
      entry.reject(new DOMException('Dropped', 'AbortError'));
    }
    if (entry.url) URL.revokeObjectURL(entry.url);
    entry.url = null;
  },

  dropWhere(predicate) {
    for (const entry of [...this.entries.values()]) if (!entry.pinned && predicate(entry)) this.drop(entry);
  },

  clear() {
    for (const entry of [...this.entries.values()]) {
      entry.pinned = 0;
      this.drop(entry);
    }
  },

  /** Decode on demand; only the compact encoded clips are cached, never the PCM. */
  async decoded(entry, ctx) {
    const blob = await entry.promise;
    const bytes = await blob.arrayBuffer();
    try {
      return await ctx.decodeAudioData(bytes);
    } catch (err) {
      const failure = new Error("This voice's audio couldn't be decoded.");
      failure.isDecodeError = true;
      failure.cause = err;
      throw failure;
    }
  },

  objectUrl(entry) {
    if (!entry.url) entry.url = URL.createObjectURL(entry.blob);
    return entry.url;
  },
};

/** One AudioContext for the whole app, created on the first user gesture. */
const webAudio = {
  ctx: null,
  master: null,
  broken: false,

  supported() {
    return !this.broken && !!(window.AudioContext || window.webkitAudioContext);
  },

  /** Create the context if needed (it never resumes a context we paused on purpose). */
  ensure() {
    if (!this.supported()) return null;
    if (!this.ctx) {
      try {
        const Ctx = window.AudioContext || window.webkitAudioContext;
        this.ctx = new Ctx({ latencyHint: 'playback' });
        this.master = this.ctx.createGain();
        this.master.connect(this.ctx.destination);
        this.ctx.addEventListener('statechange', () => {
          if (this.ctx.state === 'interrupted' && player.playing) pause();
        });
      } catch {
        this.broken = true;
        this.ctx = null;
        return null;
      }
    }
    return this.ctx;
  },

  resume() {
    if (this.ctx && this.ctx.state !== 'running') return this.ctx.resume().catch(() => {});
    return Promise.resolve();
  },

  /** The context time of the sample reaching the speakers right now. */
  heardTime() {
    const ctx = this.ctx;
    return ctx ? ctx.currentTime - (ctx.outputLatency || 0) : 0;
  },

  /** Schedule one buffer with tiny fades so joins never click. */
  playAt(buffer, startAt) {
    const ctx = this.ctx;
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    const gain = ctx.createGain();
    const end = startAt + buffer.duration;
    const fade = Math.min(FADE_SECONDS, buffer.duration / 4);
    gain.gain.setValueAtTime(0, startAt);
    gain.gain.linearRampToValueAtTime(1, startAt + fade);
    gain.gain.setValueAtTime(1, Math.max(startAt + fade, end - fade));
    gain.gain.linearRampToValueAtTime(0, end);
    source.connect(gain);
    gain.connect(this.master);
    source.onended = () => {
      try {
        source.disconnect();
        gain.disconnect();
      } catch {
        /* already disconnected */
      }
    };
    source.start(startAt);
    return { source, gain, start: startAt, end };
  },

  /** Fade out and stop a scheduled source (also cancels one that hasn't started). */
  silence(node) {
    const ctx = this.ctx;
    if (!ctx || !node) return;
    const t = ctx.currentTime;
    try {
      node.gain.gain.cancelScheduledValues(t);
      node.gain.gain.setValueAtTime(t >= node.start && t < node.end ? node.gain.gain.value : 0, t);
      node.gain.gain.linearRampToValueAtTime(0, t + 0.012);
      node.source.stop(t + 0.015);
    } catch {
      /* already stopped */
    }
  },
};

/** The browser's own speech engine (Web Speech API). */
const browserSpeech = {
  current: null,
  keepAlive: 0,

  speak(text, { voiceURI, rate }) {
    return new Promise((resolve, reject) => {
      const synth = window.speechSynthesis;
      const wasBusy = synth.speaking || synth.pending;
      this.cancel();
      let done = false;
      let watchdog = 0;
      const token = {};
      const finish = (result, error) => {
        if (done) return;
        done = true;
        clearInterval(this.keepAlive);
        clearTimeout(watchdog);
        if (this.current === token) this.current = null;
        if (error) reject(error);
        else resolve(result);
      };
      token.finish = finish;
      this.current = token;
      const utter = new SpeechSynthesisUtterance(text);
      const voice = pickBrowserVoice(voiceURI);
      if (voice) {
        utter.voice = voice;
        utter.lang = voice.lang;
      }
      utter.rate = clamp(rate, 0.5, 3);
      utter.onend = () => finish('ended');
      utter.onerror = (event) => {
        if (event.error === 'canceled' || event.error === 'interrupted') finish('cancelled');
        else finish(null, new Error(`The browser voice stopped (${event.error || 'unknown error'}). Try another voice in Settings → Voice.`));
      };
      const begin = () => {
        if (done) return;
        synth.speak(utter);
        // Chrome stops long utterances after ~15 s unless nudged.
        this.keepAlive = setInterval(() => {
          if (synth.speaking && !synth.paused) synth.resume();
        }, 5000);
        // Some browsers never fire `end`; don't hang forever.
        const expected = 2500 + (text.length * 95) / Math.max(0.5, rate);
        const check = () => {
          if (done) return;
          if (!synth.speaking && !synth.pending) finish('ended');
          else watchdog = setTimeout(check, 1500);
        };
        watchdog = setTimeout(check, expected * 1.6);
      };
      // Chrome can drop a speak() issued in the same tick as cancel().
      if (wasBusy) setTimeout(begin, 60);
      else begin();
    });
  },

  cancel() {
    const current = this.current;
    this.current = null;
    clearInterval(this.keepAlive);
    if (hasSpeechSynthesis() && (window.speechSynthesis.speaking || window.speechSynthesis.pending)) {
      window.speechSynthesis.cancel();
    }
    current?.finish('cancelled');
  },
};

/** Wait briefly for the browser to list its voices; false if it has none we may use. */
async function ensureBrowserVoices() {
  if (!hasSpeechSynthesis()) return false;
  if (usableBrowserVoices().length) return true;
  await new Promise((resolve) => {
    const timer = setTimeout(resolve, 1500);
    window.speechSynthesis.addEventListener('voiceschanged', () => {
      clearTimeout(timer);
      resolve();
    }, { once: true });
  });
  return usableBrowserVoices().length > 0;
}

function noBrowserVoicesMessage() {
  let message;
  if (!hasSpeechSynthesis()) {
    message = "This browser can't speak text. Choose another voice in Settings → Voice.";
  } else if (browserVoicesAll().length) {
    message = 'This browser only has online voices. Turn on “Online voices” in Settings → Privacy & AI, or choose another voice.';
  } else {
    message = "This browser didn't provide any voices. Choose another voice in Settings → Voice, such as a natural Piper voice.";
  }
  toast(message, { kind: 'error', action: { label: 'Voice settings', run: () => openSettings('voice') } });
  announce(message);
}

/** Split long text (an AI answer) into voice-sized pieces at sentence ends. */
function speechChunks(text, max = 320) {
  const sentences = String(text || '').replace(/\s+/g, ' ').trim().match(/[^.!?]+[.!?]+["')\]]*\s*|[^.!?]+$/g) || [];
  const chunks = [];
  let current = '';
  for (const raw of sentences) {
    let sentence = raw.trim();
    while (sentence.length > max) {
      const cut = sentence.lastIndexOf(' ', max) > max / 2 ? sentence.lastIndexOf(' ', max) : max;
      chunks.push(sentence.slice(0, cut).trim());
      sentence = sentence.slice(cut).trim();
    }
    if (current && (current.length + sentence.length + 1) > max) {
      chunks.push(current);
      current = sentence;
    } else {
      current = current ? `${current} ${sentence}` : sentence;
    }
  }
  if (current) chunks.push(current);
  return chunks.filter((c) => c.trim());
}

/** Reads AI answers and voice samples aloud with the current voice (pauses the book first). */
const speaker = {
  gen: 0,
  button: null,
  label: '',
  nodes: [],

  async toggle(text, button = null) {
    if (button && this.button === button) {
      this.stop();
      return;
    }
    this.stop();
    const chunks = speechChunks(text);
    if (!chunks.length) return;
    stopBookForSpeaker();
    const gen = ++this.gen;
    const live = () => gen === this.gen;
    this.setButton(button);
    const engine = activeEngine();
    const voice = activeVoice(engine);
    const rate = state.speed;
    try {
      if (engine === 'browser') {
        if (!(await ensureBrowserVoices())) {
          noBrowserVoicesMessage();
          return;
        }
        for (const chunk of chunks) {
          if (!live()) return;
          const result = await browserSpeech.speak(chunk, { voiceURI: voice, rate });
          if (result === 'cancelled') return;
        }
      } else if (webAudio.ensure()) {
        await this.playWebAudio(chunks, engine, voice, rate, live);
      } else {
        for (const chunk of chunks) {
          if (!live()) return;
          const result = await playClipElement(chunk, engine, voice, rate, live);
          if (result === 'cancelled') return;
        }
      }
    } catch (err) {
      if (!live() || err?.name === 'AbortError') return;
      if (err?.isDecodeError) {
        webAudio.broken = true;
        this.stop();
        this.toggle(text, button);
        return;
      }
      toast(errorText(err), { kind: 'error' });
    } finally {
      if (live()) this.reset();
    }
  },

  async playWebAudio(chunks, engine, voice, rate, live) {
    const ctx = webAudio.ctx;
    await webAudio.resume();
    const entries = chunks.map((chunk, i) => clips.get(chunk, engine, voice, rate, { urgent: i === 0 }));
    let at = ctx.currentTime + 0.05;
    let lastEnded = Promise.resolve();
    for (const entry of entries) {
      entry.pinned += 1;
      let buffer;
      try {
        buffer = await clips.decoded(entry, ctx);
      } finally {
        entry.pinned -= 1;
      }
      if (!live()) return;
      at = Math.max(at, ctx.currentTime + 0.02);
      const node = webAudio.playAt(buffer, at);
      this.nodes.push(node);
      lastEnded = new Promise((resolve) => node.source.addEventListener('ended', resolve, { once: true }));
      at = node.end + 0.3;
    }
    await lastEnded;
  },

  setButton(button) {
    this.button = button;
    if (!button) return;
    this.label = button.querySelector('.btn-text')?.textContent || '';
    button.classList.add('is-speaking');
    button.setAttribute('aria-pressed', 'true');
    setIcon(button.querySelector('.i'), 'stop');
    const text = button.querySelector('.btn-text');
    if (text) text.textContent = 'Stop';
  },

  reset() {
    const button = this.button;
    this.button = null;
    for (const node of this.nodes) webAudio.silence(node);
    this.nodes = [];
    if (!button) return;
    button.classList.remove('is-speaking');
    button.setAttribute('aria-pressed', 'false');
    setIcon(button.querySelector('.i'), 'speaker');
    const text = button.querySelector('.btn-text');
    if (text) text.textContent = this.label || 'Read aloud';
  },

  stop() {
    this.gen += 1;
    stopAudioElement();
    browserSpeech.cancel();
    this.reset();
  },
};

/** Fallback path: play one clip through the <audio> element. */
async function playClipElement(text, engine, voice, rate, live) {
  const entry = clips.get(text, engine, voice, rate, { urgent: true });
  entry.pinned += 1;
  try {
    await entry.promise;
    if (!live()) return 'cancelled';
    return await playUrl(clips.objectUrl(entry));
  } finally {
    entry.pinned -= 1;
  }
}

// =====================================================================
// 6. Gapless scheduler for server voices
// =====================================================================

const AHEAD = 3;
const sectionCache = new Map();

function fetchSectionData(bookId, sectionId) {
  const key = `${bookId}|${sectionId}`;
  let promise = sectionCache.get(key);
  if (!promise) {
    promise = api(`/api/books/${enc(bookId)}/sections/${enc(sectionId)}`).then((data) => {
      data.blocks = data.blocks || [];
      data.utterances = data.utterances || [];
      return data;
    });
    promise.catch(() => {
      if (sectionCache.get(key) === promise) sectionCache.delete(key);
    });
    sectionCache.set(key, promise);
    while (sectionCache.size > 8) sectionCache.delete(sectionCache.keys().next().value);
  }
  return promise;
}

/**
 * Schedules decoded sentences back to back on the AudioContext clock and moves
 * the highlight when each one actually starts. Pause/resume suspend the clock.
 */
const sched = {
  active: false,
  session: 0,
  fillGen: 0,
  filling: false,
  items: [], // {payload, index, u, source, gain, start, end, next}
  cursor: null, // {payload, index}: the next utterance to schedule
  timeEnd: 0,
  current: null,
  ended: null, // null | 'book' | 'sleep'
  timer: 0,
  engine: '',
  voice: '',
  rate: 1,

  start(index) {
    const ctx = webAudio.ctx;
    this.teardown();
    this.active = true;
    this.engine = activeEngine();
    this.voice = activeVoice(this.engine);
    this.rate = state.speed;
    this.cursor = { payload: state.section, index };
    this.timeEnd = ctx.currentTime + 0.06;
    webAudio.resume();
    this.timer = setInterval(() => this.tick(), 40);
    clips.dropWhere((e) => e.rate !== this.rate || e.engine !== this.engine || e.voice !== this.voice);
    this.fill();
  },

  teardown() {
    this.session += 1;
    this.fillGen += 1;
    this.filling = false;
    clearInterval(this.timer);
    this.timer = 0;
    for (const item of this.items) webAudio.silence(item);
    this.items = [];
    this.current = null;
    this.cursor = null;
    this.ended = null;
    this.active = false;
  },

  canResume() {
    return this.active && this.items.length > 0 && this.engine === activeEngine()
      && this.voice === activeVoice(this.engine);
  },

  aheadCount(now) {
    let count = 0;
    for (const item of this.items) if (item.start > now) count += 1;
    return count;
  },

  async fill() {
    if (this.filling || !this.active) return;
    this.filling = true;
    const session = this.session;
    const fillGen = this.fillGen;
    const valid = () => this.active && session === this.session && fillGen === this.fillGen;
    try {
      while (valid() && this.cursor && this.aheadCount(webAudio.ctx.currentTime) < AHEAD) {
        const next = await this.resolveCursor(valid);
        if (!valid()) return;
        if (!next) break;
        this.prefetch(next);
        const entry = clips.get(next.u.speech, this.engine, this.voice, this.rate, { urgent: true });
        entry.pinned += 1;
        let buffer;
        try {
          buffer = await clips.decoded(entry, webAudio.ctx);
        } finally {
          entry.pinned -= 1;
        }
        if (!valid()) return;
        this.schedule(next, buffer);
        this.cursor = { payload: next.payload, index: next.index + 1 };
      }
    } catch (err) {
      if (valid()) this.failed(err);
    } finally {
      if (session === this.session && fillGen === this.fillGen) this.filling = false;
    }
  },

  /** The utterance at the cursor, crossing into the next section when needed. */
  async resolveCursor(valid) {
    let { payload, index } = this.cursor;
    for (let hops = 0; hops < 200; hops += 1) {
      if (index < payload.utterances.length) return { payload, index, u: payload.utterances[index] };
      if (sleepTimer.mode === 'section') {
        this.ended = 'sleep';
        this.cursor = null;
        return null;
      }
      if (!payload.next_id || !state.book) {
        this.ended = 'book';
        this.cursor = null;
        return null;
      }
      const nextPayload = await fetchSectionData(state.book.id, payload.next_id);
      if (!valid()) return null;
      payload = nextPayload;
      index = 0;
      this.cursor = { payload, index };
    }
    return null;
  },

  prefetch(next) {
    const list = next.payload.utterances;
    for (let i = next.index + 1; i <= next.index + AHEAD && i < list.length; i += 1) {
      clips.get(list[i].speech, this.engine, this.voice, this.rate);
    }
  },

  schedule(next, buffer) {
    const ctx = webAudio.ctx;
    const prev = this.items[this.items.length - 1];
    let startAt = Math.max(this.timeEnd, ctx.currentTime + 0.03);
    if (prev && prev.payload !== next.payload) startAt += SECTION_GAP_SECONDS;
    const node = webAudio.playAt(buffer, startAt);
    const item = { ...node, payload: next.payload, index: next.index, u: next.u, next: node.end + pauseSeconds(next.u, this.rate) };
    this.items.push(item);
    this.timeEnd = item.next;
  },

  tick() {
    if (!this.active || !webAudio.ctx) return;
    const now = webAudio.ctx.currentTime;
    const heard = webAudio.heardTime();
    let current = null;
    for (const item of this.items) {
      if (item.start <= heard + 0.005) current = item;
      else break;
    }
    if (current && current !== this.current) {
      this.current = current;
      onScheduledStart(current);
      if (!this.active) return;
    }
    if (this.items.length > 1 && this.items[0] !== current && this.items[0].next < now) {
      this.items = this.items.filter((item) => item === current || item.next >= now);
    }
    if (!this.filling && this.cursor && this.aheadCount(now) < AHEAD) this.fill();
    if (this.ended && !this.filling && !this.cursor) {
      const last = this.items[this.items.length - 1];
      if (!last || heard >= last.end + 0.05) {
        const reason = this.ended;
        this.teardown();
        onReachedEnd(reason);
      }
    }
  },

  /** Speed changed: finish the current sentence, re-fetch everything after it at the new rate. */
  changeRate(rate) {
    if (!this.active || rate === this.rate) return;
    const now = webAudio.ctx.currentTime;
    this.rate = rate;
    this.fillGen += 1;
    this.filling = false;
    const keep = this.items.filter((item) => item.start <= now);
    const drop = this.items.filter((item) => item.start > now);
    for (const item of drop) webAudio.silence(item);
    this.items = keep;
    if (drop.length) {
      this.cursor = { payload: drop[0].payload, index: drop[0].index };
      this.ended = null;
    }
    const last = keep[keep.length - 1];
    this.timeEnd = last ? Math.max(now, last.end + pauseSeconds(last.u, rate)) : now + 0.05;
    clips.dropWhere((entry) => entry.rate !== rate);
    this.fill();
  },

  /** Sleep timer "end of section": drop anything already scheduled past this section. */
  stopAfterSection() {
    if (!this.active) return;
    const base = this.current ? this.current.payload : state.section;
    const beyond = this.items.filter((item) => item.payload !== base);
    const cursorBeyond = this.cursor && this.cursor.payload !== base;
    if (!beyond.length && !cursorBeyond) return;
    for (const item of beyond) webAudio.silence(item);
    this.items = this.items.filter((item) => item.payload === base);
    this.fillGen += 1;
    this.filling = false;
    this.cursor = { payload: base, index: base.utterances.length };
    const last = this.items[this.items.length - 1];
    this.timeEnd = last ? last.next : webAudio.ctx.currentTime + 0.05;
    this.ended = null;
    this.fill();
  },

  failed(err) {
    if (err?.name === 'AbortError') return;
    if (err?.isDecodeError) {
      // Fall back to the <audio> element so playback never breaks.
      webAudio.broken = true;
      const index = this.current && this.current.payload === state.section ? this.current.index : state.index;
      hardStop();
      player.playing = false;
      setCurrent(index, { scroll: false });
      play();
      return;
    }
    handlePlaybackError(err);
  },
};

function onScheduledStart(item) {
  if (item.payload !== state.section) showSection(item.payload, item.index, { announceIt: true });
  setCurrent(item.index, { scroll: true });
  noteProgress();
}

// =====================================================================
// 7. Player
// =====================================================================

const player = {
  playing: false,
  mode: null, // 'webaudio' | 'sequential'
  gen: 0,
  sinceSave: 0,
  lastSaved: '',
};

/** Stop every kind of output and invalidate pending callbacks. */
function hardStop() {
  player.gen += 1;
  sched.teardown();
  stopAudioElement();
  browserSpeech.cancel();
}

function stopBookForSpeaker() {
  const wasPlaying = player.playing;
  hardStop();
  player.playing = false;
  if (wasPlaying) {
    updatePlayUI();
    saveProgress();
  }
}

function play() {
  if (!state.section || player.playing) return;
  speaker.stop();
  const engine = activeEngine();
  if (engine === 'browser') {
    startBrowserPlayback();
    return;
  }
  const ctx = webAudio.ensure();
  player.playing = true;
  updatePlayUI();
  announce('Playing');
  if (ctx) {
    player.mode = 'webaudio';
    if (sched.canResume()) {
      webAudio.resume();
    } else {
      hardStop();
      sched.start(state.index);
    }
  } else {
    player.mode = 'sequential';
    hardStop();
    runSequential(player.gen);
  }
}

async function startBrowserPlayback() {
  hardStop();
  if (!(await ensureBrowserVoices())) {
    noBrowserVoicesMessage();
    return;
  }
  if (player.playing || !state.section) return;
  player.playing = true;
  player.mode = 'sequential';
  updatePlayUI();
  announce('Playing');
  runSequential(player.gen);
}

function pause({ announceIt = true } = {}) {
  if (!player.playing) return;
  player.playing = false;
  if (player.mode === 'webaudio' && sched.active && webAudio.ctx) {
    webAudio.ctx.suspend().catch(() => {});
  } else {
    hardStop();
  }
  updatePlayUI();
  if (announceIt) announce('Paused');
  saveProgress();
}

function togglePlay() {
  if (player.playing) pause();
  else play();
}

/** Restart output from the current sentence (after a voice change). */
function restartPlayback() {
  const was = player.playing;
  hardStop();
  player.playing = false;
  if (was) play();
  else updatePlayUI();
}

/** Browser voices and the <audio> fallback: one sentence at a time. */
async function runSequential(gen) {
  const live = () => gen === player.gen && player.playing;
  while (live()) {
    const sec = state.section;
    if (!sec) return;
    if (state.index >= sec.utterances.length || !sec.utterances.length) {
      if (sleepTimer.mode === 'section') {
        onReachedEnd('sleep');
        return;
      }
      if (!sec.next_id) {
        onReachedEnd('book');
        return;
      }
      const ok = await loadSection(sec.next_id, 0, { announceIt: true });
      if (!live()) return;
      if (!ok) {
        pause({ announceIt: false });
        return;
      }
      continue;
    }
    const u = sec.utterances[state.index];
    setCurrent(state.index, { scroll: true });
    const engine = activeEngine();
    const voice = activeVoice(engine);
    const rate = state.speed;
    let result;
    try {
      if (engine === 'browser') {
        result = await browserSpeech.speak(u.speech, { voiceURI: voice, rate });
      } else {
        for (let i = state.index + 1; i <= state.index + AHEAD && i < sec.utterances.length; i += 1) {
          clips.get(sec.utterances[i].speech, engine, voice, rate);
        }
        result = await playClipElement(u.speech, engine, voice, rate, live);
      }
    } catch (err) {
      if (live() && err?.name !== 'AbortError') handlePlaybackError(err);
      return;
    }
    if (!live() || result === 'cancelled') return;
    const pauseMs = pauseSeconds(u, rate) * 1000;
    if (pauseMs > 0) {
      await wait(pauseMs);
      if (!live()) return;
    }
    if (state.section === sec) {
      state.index += 1;
      noteProgress();
    }
  }
}

function handlePlaybackError(err) {
  hardStop();
  player.playing = false;
  updatePlayUI();
  saveProgress();
  if (err?.name === 'NotAllowedError') {
    toast('Your browser blocked the sound. Press Play to start reading.', { kind: 'info' });
    return;
  }
  const message = errorText(err);
  announce(`Reading stopped. ${message}`);
  if (activeEngine() !== 'browser' && hasSpeechSynthesis()) {
    toast(message, { kind: 'error', action: { label: 'Use browser voice instead', run: useBrowserVoice } });
  } else {
    toast(message, { kind: 'error', action: { label: 'Voice settings', run: () => openSettings('voice') } });
  }
}

async function useBrowserVoice() {
  if (await saveSettings({ tts_engine: 'browser' })) {
    renderVoicePicker();
    play();
  }
}

function onReachedEnd(reason) {
  hardStop();
  player.playing = false;
  updatePlayUI();
  saveProgress();
  if (reason === 'sleep') {
    setSleepMode('off', { quiet: true });
    toast('Sleep timer: reading paused at the end of the section. Sleep well!');
    announce('Paused by the sleep timer');
  } else {
    const title = state.book?.title || 'this book';
    toast(`You've reached the end of “${title}”.`, { kind: 'success' });
    announce('End of the book');
  }
}

function seekTo(index, { play: shouldPlay = null } = {}) {
  if (!state.section) return;
  const resume = shouldPlay ?? player.playing;
  hardStop();
  player.playing = false;
  if (resume) webAudio.ensure(); // still inside the user's gesture
  setCurrent(index, { scroll: true, force: true });
  saveProgressSoon();
  if (resume) play();
  else updatePlayUI();
}

async function goToSection(sectionId, { index = 0, play: shouldPlay = null, focusContent = false } = {}) {
  if (!state.book || !sectionId) return;
  const resume = shouldPlay ?? player.playing;
  hardStop();
  player.playing = false;
  updatePlayUI();
  if (resume) webAudio.ensure();
  const ok = await loadSection(sectionId, index, { announceIt: true });
  if (!ok) return;
  saveProgress();
  if (focusContent) $('#section-title')?.focus({ preventScroll: true });
  if (resume) play();
}

function prevSentence() {
  if (!state.section) return;
  if (state.index > 0) seekTo(state.index - 1);
  else if (state.section.prev_id) goToSection(state.section.prev_id, { index: -1 });
}

function nextSentence() {
  if (!state.section) return;
  if (state.index < state.section.utterances.length - 1) seekTo(state.index + 1);
  else if (state.section.next_id) goToSection(state.section.next_id);
}

function prevSection() {
  if (state.section?.prev_id) goToSection(state.section.prev_id);
  else if (state.section) seekTo(0);
}

function nextSection() {
  if (state.section?.next_id) goToSection(state.section.next_id);
}

function updatePlayUI() {
  const button = $('#btn-play');
  const has = !!state.section;
  button.disabled = !has;
  setIcon(button.querySelector('.i'), player.playing ? 'pause' : 'play');
  button.setAttribute('aria-label', player.playing ? 'Pause' : 'Play');
  button.title = player.playing ? 'Pause (Space)' : 'Play (Space)';
  for (const id of ['btn-prev', 'btn-next', 'btn-prev-section', 'btn-next-section', 'btn-bookmark']) {
    $(`#${id}`).disabled = !has;
  }
  if (has) {
    $('#btn-prev-section').disabled = !state.section.prev_id && state.index === 0;
    $('#btn-next-section').disabled = !state.section.next_id;
  }
  document.body.classList.toggle('is-playing', player.playing);
  if ('mediaSession' in navigator) {
    try {
      navigator.mediaSession.playbackState = player.playing ? 'playing' : (has ? 'paused' : 'none');
    } catch {
      /* unsupported */
    }
  }
}

function updateStatus() {
  const sec = state.section;
  $('#now-title').textContent = state.book ? state.book.title : 'Nothing open';
  let status;
  if (!state.book) status = 'Add or open something to start listening';
  else if (!sec) status = 'Loading…';
  else if (!sec.utterances.length) status = `Nothing to read aloud here · ${sec.section.title}`;
  else status = `Sentence ${state.index + 1} of ${sec.utterances.length} · ${sec.section.title}`;
  $('#now-status').textContent = status;
}

// ----- speed

const applySpeedSoon = debounce(() => {
  if (sched.active) sched.changeRate(state.speed);
  else clips.dropWhere((entry) => entry.rate !== state.speed);
}, 250);
const saveSpeedSoon = debounce(() => saveSettings({ speed: state.speed }), 700);

function setSpeed(value, { save = true, apply = true } = {}) {
  const speed = Math.round(clamp(Number(value) || 1, 0.5, 3) * 20) / 20;
  state.speed = speed;
  const slider = $('#speed');
  if (Number(slider.value) !== speed) slider.value = String(speed);
  $('#speed-value').textContent = formatSpeed(speed);
  slider.setAttribute('aria-valuetext', `${speed.toFixed(2)} times`);
  if (apply) applySpeedSoon();
  if (save) saveSpeedSoon();
}

function changeSpeed(delta) {
  setSpeed(state.speed + delta);
  announce(`Speed ${state.speed.toFixed(2)}`);
}

// ----- progress

function noteProgress() {
  player.sinceSave += 1;
  if (player.sinceSave >= 5) saveProgress();
}

const saveProgressSoon = debounce(() => saveProgress(), 1200);

function saveProgress({ beacon = false } = {}) {
  if (!state.book || !state.section) return;
  player.sinceSave = 0;
  const body = { section_id: state.section.section.id, utterance: state.index };
  const key = `${state.book.id}|${body.section_id}|${body.utterance}`;
  if (key === player.lastSaved && !beacon) return;
  player.lastSaved = key;
  const summary = state.books.find((b) => b.id === state.book.id);
  const progress = { ...body, updated_at: new Date().toISOString() };
  if (summary) summary.progress = progress;
  state.book.progress = progress;
  api(`/api/books/${enc(state.book.id)}/progress`, { method: 'PUT', body, keepalive: beacon })
    .then(() => { if (!beacon) renderLibraryProgress(); })
    .catch(() => {
      player.lastSaved = '';
    });
}

// ----- sleep timer

const sleepTimer = { mode: 'off', remaining: 0, last: 0, timer: 0 };

function setSleepMode(mode, { quiet = false } = {}) {
  clearInterval(sleepTimer.timer);
  sleepTimer.timer = 0;
  const minutes = Number(mode);
  if (Number.isFinite(minutes) && minutes > 0) {
    sleepTimer.mode = minutes;
    sleepTimer.remaining = minutes * 60000;
    sleepTimer.last = performance.now();
    sleepTimer.timer = setInterval(tickSleep, 1000);
  } else {
    sleepTimer.mode = mode === 'section' ? 'section' : 'off';
  }
  if (sleepTimer.mode === 'section' && sched.active) sched.stopAfterSection();
  renderSleep();
  if (!quiet) {
    announce(sleepTimer.mode === 'off' ? 'Sleep timer off'
      : sleepTimer.mode === 'section' ? 'Reading will pause at the end of this section'
        : `Sleep timer set for ${minutes} minutes`);
  }
}

function tickSleep() {
  const now = performance.now();
  if (player.playing) sleepTimer.remaining -= now - sleepTimer.last;
  sleepTimer.last = now;
  if (sleepTimer.remaining <= 0) {
    setSleepMode('off', { quiet: true });
    pause({ announceIt: false });
    toast('Sleep timer finished, so reading is paused. Sleep well!');
    announce('Paused by the sleep timer');
  }
  renderSleep();
}

function renderSleep() {
  const button = $('#btn-sleep');
  const left = $('#sleep-left');
  let label = 'Sleep timer: off';
  let short = '';
  if (sleepTimer.mode === 'section') {
    label = 'Sleep timer: pause at the end of this section';
    short = 'End';
  } else if (typeof sleepTimer.mode === 'number') {
    const seconds = Math.max(0, Math.ceil(sleepTimer.remaining / 1000));
    const minutes = Math.ceil(seconds / 60);
    short = seconds >= 60 ? `${minutes}m` : `0:${String(seconds).padStart(2, '0')}`;
    label = `Sleep timer: ${seconds >= 60 ? plural(minutes, 'minute') : plural(seconds, 'second')} left`;
  }
  left.textContent = short;
  button.classList.toggle('is-active', sleepTimer.mode !== 'off');
  button.setAttribute('aria-label', label);
  button.title = label;
  for (const option of $$('[data-sleep]')) {
    const value = option.dataset.sleep;
    const active = String(sleepTimer.mode) === value;
    option.setAttribute('aria-pressed', active ? 'true' : 'false');
  }
}

function toggleSleepMenu(open = $('#sleep-menu').hidden) {
  $('#sleep-menu').hidden = !open;
  $('#btn-sleep').setAttribute('aria-expanded', open ? 'true' : 'false');
  if (open) $('#sleep-menu [aria-pressed="true"]')?.focus();
}

// ----- media session

function setupMediaSession() {
  if (!('mediaSession' in navigator)) return;
  const handlers = {
    play: () => play(),
    pause: () => pause(),
    nexttrack: () => nextSentence(),
    previoustrack: () => prevSentence(),
  };
  for (const [action, handler] of Object.entries(handlers)) {
    try {
      navigator.mediaSession.setActionHandler(action, handler);
    } catch {
      /* unsupported action */
    }
  }
}

function updateMediaSession() {
  if (!('mediaSession' in navigator) || typeof window.MediaMetadata !== 'function') return;
  if (!state.book || !state.section) {
    navigator.mediaSession.metadata = null;
    return;
  }
  navigator.mediaSession.metadata = new window.MediaMetadata({
    title: state.section.section.title,
    album: state.book.title,
    artist: state.book.author || 'Athena',
    artwork: [{ src: '/static/icon.svg', sizes: '512x512', type: 'image/svg+xml' }],
  });
}

// =====================================================================
// 8. Reader
// =====================================================================

const reader = {
  targets: [], // utterance index -> {node, whole}
  lit: [],
  autoScrollPausedAt: -Infinity,
  programmaticUntil: 0,
};

function isNotesBlock(block) {
  const meta = block.meta || {};
  return block.notes === true || meta.speaker_notes === true || meta.notes === true
    || meta.role === 'notes' || meta.role === 'speaker_notes';
}

function pageLabel(page) {
  return state.book?.source_type === 'pptx' ? `Slide ${page}` : `Page ${page}`;
}

/** Wrap each utterance's span of `text` so it can be highlighted. */
function fillSentences(container, text, utterances, blockNode) {
  const sorted = [...(utterances || [])].sort((a, b) => a.start - b.start || a.index - b.index);
  let pos = 0;
  for (const u of sorted) {
    const fits = u.end > u.start && u.start >= pos && u.end <= text.length;
    if (!fits) {
      reader.targets[u.index] = { node: blockNode, whole: true };
      continue;
    }
    if (u.start > pos) container.append(text.slice(pos, u.start));
    const span = el('span', { class: 'u', dataset: { u: String(u.index) } }, text.slice(u.start, u.end));
    container.append(span);
    reader.targets[u.index] = { node: span, whole: false };
    pos = u.end;
  }
  if (pos < text.length) container.append(text.slice(pos));
}

function markWholeBlock(node, utterances) {
  if (!utterances?.length) return;
  node.dataset.uFirst = String(Math.min(...utterances.map((u) => u.index)));
  for (const u of utterances) reader.targets[u.index] = { node, whole: true };
}

function blockTools(kind, blockIndex) {
  if (!aiReady()) return null;
  const label = kind === 'table' ? 'Explain this table' : 'Explain this equation';
  return el('div', { class: 'block-tools' },
    el('button', {
      type: 'button', class: 'btn btn-small btn-ghost',
      onclick: () => explainBlock(blockIndex),
    }, el('span', { class: 'i' }, icon('bulb')), label));
}

function renderBlock(block, index, utterances) {
  const text = block.text || '';
  const kind = block.kind;
  let node;
  switch (kind) {
    case 'heading': {
      const level = clamp(Number(block.level) || 1, 1, 3);
      node = el(`h${level + 1}`);
      fillSentences(node, text, utterances, node);
      break;
    }
    case 'list_item':
      node = el('li');
      fillSentences(node, text, utterances, node);
      break;
    case 'quote':
      node = el('blockquote');
      fillSentences(node, text, utterances, node);
      break;
    case 'caption':
      node = el('p', { class: 'caption' });
      fillSentences(node, text, utterances, node);
      break;
    case 'footnote':
      node = el('p', { class: 'footnote' });
      fillSentences(node, text, utterances, node);
      break;
    case 'equation':
      node = el('div', { class: 'equation', role: 'math', 'aria-label': text });
      fillSentences(node, text, utterances, node);
      break;
    case 'figure': {
      const caption = el('span', { class: 'figure-text' });
      node = el('figure', { class: 'figure' }, el('span', { class: 'figure-icon', 'aria-hidden': 'true', text: '🖼' }), caption);
      if (text) fillSentences(caption, text, utterances, node);
      else {
        caption.textContent = 'Image (no description in the file)';
        markWholeBlock(node, utterances);
      }
      break;
    }
    case 'code':
      node = el('pre', { class: 'code', tabindex: '0', 'aria-label': 'Code' }, el('code', { text }));
      markWholeBlock(node, utterances);
      break;
    case 'table': {
      node = el('div', { class: 'table-wrap', tabindex: '0', role: 'region', 'aria-label': 'Table' });
      const rows = Array.isArray(block.rows) ? block.rows.filter((r) => Array.isArray(r)) : [];
      if (rows.length) {
        const table = el('table');
        const width = Math.max(...rows.map((r) => r.length));
        const bodyRows = rows.length > 1 ? rows.slice(1) : rows;
        if (rows.length > 1) {
          table.append(el('thead', null, el('tr', null,
            Array.from({ length: width }, (_, i) => el('th', { scope: 'col', text: rows[0][i] ?? '' })))));
        }
        table.append(el('tbody', null, bodyRows.map((row) => el('tr', null,
          Array.from({ length: width }, (_, i) => el('td', { text: row[i] ?? '' }))))));
        node.append(table);
      } else {
        node.append(el('pre', { class: 'code', text }));
      }
      markWholeBlock(node, utterances);
      break;
    }
    default:
      node = el('p');
      fillSentences(node, text, utterances, node);
  }
  node.dataset.block = String(index);
  return node;
}

function renderSection() {
  const host = $('#article-host');
  if (!host || !state.section) return;
  const { section, blocks, utterances } = state.section;
  reader.targets = new Array(utterances.length);
  reader.lit = [];
  const byBlock = new Map();
  for (const u of utterances) {
    if (!byBlock.has(u.block)) byBlock.set(u.block, []);
    byBlock.get(u.block).push(u);
  }

  const article = el('article', { class: 'article', 'aria-labelledby': 'section-title' });
  const first = blocks[0];
  const titleFromBlock = first && first.kind === 'heading' && norm(first.text) === norm(section.title);
  const title = el('h1', { class: 'section-title', id: 'section-title', tabindex: '-1' });
  if (titleFromBlock) {
    fillSentences(title, first.text, byBlock.get(0), title);
    title.dataset.block = '0';
  } else {
    title.textContent = section.title;
  }
  const announced = byBlock.get(-1);
  if (announced) markWholeBlock(title, announced);
  article.append(title);

  const isPaged = ['pdf', 'pptx'].includes(state.book?.source_type);
  let lastPage = null;
  let list = null;
  let notes = null;
  for (let i = titleFromBlock ? 1 : 0; i < blocks.length; i += 1) {
    const block = blocks[i];
    const node = renderBlock(block, i, byBlock.get(i));
    const isItem = block.kind === 'list_item';
    const isNote = isNotesBlock(block);
    if (isPaged && block.page && block.page !== lastPage && !(isItem && list) && !(isNote && notes)) {
      article.append(el('p', { class: 'page-marker', text: pageLabel(block.page) }));
      lastPage = block.page;
      list = null;
    }
    if (isNote) {
      if (!notes) {
        notes = el('aside', { class: 'notes', 'aria-label': 'Speaker notes' }, el('span', { class: 'notes-label', text: 'Speaker notes' }));
        article.append(notes);
      }
      notes.append(node.tagName === 'LI' ? el('p', null, ...node.childNodes) : node);
      list = null;
      continue;
    }
    notes = null;
    if (isItem) {
      if (!list) {
        list = el('ul', { class: 'list' });
        article.append(list);
      }
      list.append(node);
      continue;
    }
    list = null;
    article.append(node);
    if (block.kind === 'table' || block.kind === 'equation') {
      const tools = blockTools(block.kind, i);
      if (tools) article.append(tools);
    }
  }
  if (!utterances.length) {
    article.append(el('p', { class: 'hint', text: 'Nothing in this section is read aloud with your current reading settings.' }));
  }
  host.replaceChildren(article);
  renderSectionNav();
}

function renderSectionNav() {
  const nav = $('#section-nav');
  if (!nav || !state.section) return;
  nav.replaceChildren();
  const titleOf = (id) => state.toc.find((t) => t.id === id)?.title || '';
  const { prev_id: prevId, next_id: nextId } = state.section;
  if (prevId) {
    nav.append(el('button', { type: 'button', class: 'section-nav-btn prev', onclick: () => goToSection(prevId, { focusContent: true }) },
      el('span', { class: 'section-nav-dir', text: '← Previous' }), el('span', { class: 'section-nav-title', text: titleOf(prevId) })));
  }
  if (nextId) {
    nav.append(el('button', { type: 'button', class: 'section-nav-btn next', onclick: () => goToSection(nextId, { focusContent: true }) },
      el('span', { class: 'section-nav-dir', text: 'Next →' }), el('span', { class: 'section-nav-title', text: titleOf(nextId) })));
  }
  nav.hidden = !prevId && !nextId;
}

function highlight(index) {
  for (const node of reader.lit) node.classList.remove('is-current', 'is-current-block');
  reader.lit = [];
  const target = reader.targets[index];
  if (!target || !target.node) return;
  target.node.classList.add(target.whole ? 'is-current-block' : 'is-current');
  reader.lit.push(target.node);
}

function setCurrent(index, { scroll = true, force = false } = {}) {
  const sec = state.section;
  if (!sec) return;
  const count = sec.utterances.length;
  state.index = count ? clamp(index, 0, count - 1) : 0;
  highlight(state.index);
  updateStatus();
  updatePlayUI();
  updateStudyActions();
  updateBookmarkButton();
  if (scroll) scrollToCurrent({ force });
}

function scrollToCurrent({ force = false, instant = false } = {}) {
  const target = reader.targets[state.index];
  const node = target?.node;
  const box = $('#reader');
  if (!node || !box.contains(node)) return;
  if (!force && performance.now() - reader.autoScrollPausedAt < 6000) return;
  const r = node.getBoundingClientRect();
  const c = box.getBoundingClientRect();
  const margin = Math.min(80, c.height * 0.12);
  if (r.top >= c.top + margin && r.bottom <= c.bottom - margin) return;
  const smooth = !instant && !reducedMotion.matches;
  reader.programmaticUntil = performance.now() + (smooth ? 1200 : 200);
  // Scroll only the reading column (scrollIntoView could also shift the page itself).
  const offset = r.height > c.height * 0.6 ? 24 : (c.height - r.height) / 2;
  box.scrollTo({ top: Math.max(0, box.scrollTop + (r.top - c.top) - offset), behavior: smooth ? 'smooth' : 'auto' });
}

function scrollIntoPanel(node, panel) {
  if (!node || !panel || panel.hidden) return;
  const r = node.getBoundingClientRect();
  const c = panel.getBoundingClientRect();
  if (r.top < c.top) panel.scrollTop -= c.top - r.top + 8;
  else if (r.bottom > c.bottom) panel.scrollTop += r.bottom - c.bottom + 8;
}

function scrollThreadToEnd() {
  const box = $('#study-thread');
  box.scrollTop = box.scrollHeight;
}

function markUserScroll() {
  reader.autoScrollPausedAt = performance.now();
}

function utteranceText(u) {
  if (!u || !state.section) return '';
  if (u.block < 0) return state.section.section.title;
  const block = state.section.blocks[u.block];
  if (!block) return u.speech || '';
  const text = block.text || '';
  if (u.end > u.start && u.end <= text.length) return text.slice(u.start, u.end);
  return text || u.speech || '';
}

function currentUtterance() {
  return state.section?.utterances[state.index] || null;
}

function currentBlock() {
  const u = currentUtterance();
  return u && u.block >= 0 ? state.section.blocks[u.block] || null : null;
}

function currentPage() {
  return currentBlock()?.page || state.section?.section?.page_start || null;
}

/** Text the user selected inside the reading column, if any. */
function readerSelection() {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed || !sel.rangeCount) return null;
  const range = sel.getRangeAt(0);
  const article = $('.article');
  if (!article || !article.contains(range.commonAncestorContainer)) return null;
  const text = sel.toString().replace(/\s+/g, ' ').trim();
  return text.length >= 2 ? { text, range } : null;
}

function utteranceAtRange(range) {
  const start = range.startContainer.nodeType === Node.ELEMENT_NODE ? range.startContainer : range.startContainer.parentElement;
  const direct = start?.closest('[data-u]') || start?.closest('[data-u-first]');
  if (direct) return Number(direct.dataset.u ?? direct.dataset.uFirst);
  for (const span of $$('.article [data-u]')) if (range.intersectsNode(span)) return Number(span.dataset.u);
  return null;
}

function updateSelectionBubble() {
  const bubble = $('#sel-bubble');
  const selection = readerSelection();
  if (!selection) {
    bubble.hidden = true;
    return;
  }
  const rect = selection.range.getBoundingClientRect();
  bubble.hidden = false;
  const { width, height } = bubble.getBoundingClientRect();
  const box = $('#reader').getBoundingClientRect();
  if (rect.bottom < box.top || rect.top > box.bottom) {
    bubble.hidden = true;
    return;
  }
  let top = rect.top - height - 10;
  if (top < box.top + 4) top = rect.bottom + 10;
  top = clamp(top, box.top + 4, box.bottom - height - 4);
  const left = clamp(rect.left + rect.width / 2 - width / 2, 8, window.innerWidth - width - 8);
  bubble.style.top = `${Math.round(top)}px`;
  bubble.style.left = `${Math.round(left)}px`;
}

function onReaderClick(event) {
  if (event.target.closest('button, a, input, textarea, select')) return;
  if (readerSelection()) return;
  const span = event.target.closest('.article [data-u]');
  if (span) {
    seekTo(Number(span.dataset.u), { play: true });
    return;
  }
  const block = event.target.closest('.article [data-u-first]');
  if (block) seekTo(Number(block.dataset.uFirst), { play: true });
}

// ----- the main column

function renderReader() {
  const body = $('#reader-body');
  if (state.book) {
    body.replaceChildren(
      el('div', { class: 'book-bar' },
        el('div', { class: 'book-bar-text' },
          el('span', { class: 'book-bar-title', text: state.book.title }),
          [TYPE_LABELS[state.book.source_type] || state.book.source_type, state.book.author].filter(Boolean).join(' · ')),
        el('button', { type: 'button', class: 'btn btn-secondary btn-small', id: 'btn-export', onclick: openExport },
          el('span', { class: 'i' }, icon('download')), 'Save as audio')),
      el('div', { id: 'book-notices' }),
      el('div', { id: 'article-host' }, el('p', { class: 'loading-line', text: 'Opening…' })),
      el('nav', { class: 'section-nav', id: 'section-nav', 'aria-label': 'Sections', hidden: true }));
    renderWarnings();
    if (state.section) renderSection();
    return;
  }
  body.replaceChildren(state.books.length ? renderChooser() : renderWelcome());
}

function renderWelcome() {
  return el('section', { class: 'welcome', 'aria-labelledby': 'welcome-title' },
    el('img', { class: 'welcome-art', src: '/static/icon.svg', alt: '', width: '84', height: '84' }),
    el('h1', { id: 'welcome-title', text: 'Welcome to Athena' }),
    el('p', { class: 'welcome-lead', text: 'Add a textbook, lecture slides or your notes, and Athena will read them aloud, highlighting each sentence so you can follow along.' }),
    el('div', { class: 'welcome-actions' },
      el('button', { type: 'button', class: 'btn btn-primary', onclick: chooseFiles }, el('span', { class: 'i' }, icon('plus')), 'Add material'),
      el('button', { type: 'button', class: 'btn btn-secondary', onclick: openPaste }, el('span', { class: 'i' }, icon('paste')), 'Paste text')),
    el('p', { class: 'welcome-drop', text: 'Or drag a file anywhere onto this window.' }),
    el('ul', { class: 'features' },
      feature('lock', 'Private', 'Everything stays on this computer. Nothing is uploaded unless you turn on an online feature.'),
      feature('book', 'Reads your files', 'PDF, EPUB, Word, PowerPoint, web pages, text, and photos of handouts.'),
      feature('bulb', 'Study helper', 'Optional: explanations, summaries and quizzes about what you are reading.')));
}

function feature(iconName, title, text) {
  return el('li', { class: 'feature' }, icon(iconName), el('p', { class: 'feature-title', text: title }), el('p', { text }));
}

function renderChooser() {
  return el('section', { class: 'welcome', 'aria-labelledby': 'choose-title' },
    el('img', { class: 'welcome-art', src: '/static/icon.svg', alt: '', width: '84', height: '84' }),
    el('h1', { id: 'choose-title', text: 'What would you like to read?' }),
    el('p', { class: 'welcome-lead', text: 'Pick something from your library, or add something new.' }),
    el('ul', { class: 'book-list chooser-list' }, state.books.slice(0, 6).map((book) => bookItem(book, { deletable: false }))),
    el('div', { class: 'welcome-actions' },
      el('button', { type: 'button', class: 'btn btn-primary', onclick: chooseFiles }, el('span', { class: 'i' }, icon('plus')), 'Add material'),
      el('button', { type: 'button', class: 'btn btn-secondary', onclick: openPaste }, el('span', { class: 'i' }, icon('paste')), 'Paste text')));
}

function renderWarnings() {
  const host = $('#book-notices');
  if (!host) return;
  host.replaceChildren();
  const warnings = (state.warnings || []).filter(Boolean);
  if (!state.book || !warnings.length || state.dismissedWarnings.has(state.book.id)) return;
  const bookId = state.book.id;
  host.append(el('div', { class: 'notice notice-warn import-notice', role: 'note' },
    icon('info'),
    el('div', { class: 'notice-body' },
      el('p', { class: 'notice-title', text: warnings.length === 1 ? 'A note about this file' : 'A few notes about this file' }),
      warnings.length === 1 ? el('p', { text: warnings[0] }) : el('ul', null, warnings.map((w) => el('li', { text: w })))),
    el('button', {
      type: 'button', class: 'icon-btn icon-btn-small', 'aria-label': 'Dismiss these notes',
      onclick: () => {
        state.dismissedWarnings.add(bookId);
        renderWarnings();
      },
    }, icon('close'))));
}

async function checkVoiceBanner() {
  const host = $('#banners');
  $('#voice-banner')?.remove();
  if (local.get('voiceBannerDismissed', false) || !engineInfo('piper')) return;
  let catalog = [];
  try {
    catalog = (await api('/api/voices/catalog')).voices || [];
  } catch {
    return;
  }
  if (!catalog.length || catalog.some((v) => v.installed)) return;
  const dismiss = () => {
    local.set('voiceBannerDismissed', true);
    $('#voice-banner')?.remove();
  };
  host.append(el('div', { class: 'notice notice-accent', id: 'voice-banner', role: 'note' },
    icon('voice'),
    el('div', { class: 'notice-body' },
      el('p', { class: 'notice-title', text: 'Try a natural-sounding voice' }),
      el('p', { text: 'Download a free voice that runs entirely on this computer (about 60 MB). It sounds much more natural than the built-in one.' }),
      el('div', { class: 'notice-actions' },
        el('button', { type: 'button', class: 'btn btn-primary btn-small', onclick: () => openSettings('voice') }, 'Choose a voice'),
        el('button', { type: 'button', class: 'btn btn-ghost btn-small', onclick: dismiss }, 'Not now')))));
}

// =====================================================================
// 9. Library, import, paste, contents and bookmarks
// =====================================================================

function sectionNumber(id) {
  const match = /^s(\d+)$/.exec(id || '');
  return match ? Number(match[1]) : null;
}

function progressInfo(book) {
  const p = book.progress;
  if (!p) return { text: 'Not started', fraction: 0 };
  const index = sectionNumber(p.section_id);
  const count = book.section_count || 0;
  if (index === null || !count) return { text: 'In progress', fraction: 0.02 };
  return { text: `Section ${index + 1} of ${count}`, fraction: Math.max(0.03, index / count) };
}

function bookItem(book, { deletable = true } = {}) {
  const info = progressInfo(book);
  const bar = el('span');
  bar.style.width = `${Math.round(info.fraction * 100)}%`;
  const parts = [TYPE_LABELS[book.source_type] || book.source_type];
  if (book.section_count) parts.push(plural(book.section_count, 'section'));
  const item = el('li', { class: `book-item${state.book?.id === book.id ? ' is-open' : ''}`, dataset: { bookId: book.id } },
    el('button', {
      type: 'button', class: 'book-open', 'aria-current': state.book?.id === book.id ? 'true' : null,
      onclick: () => openBook(book.id, { focusContent: true }),
    },
    el('span', { class: 'book-cover', 'aria-hidden': 'true', text: COVER_LABELS[book.source_type] || 'DOC' }),
    el('span', { class: 'book-text' },
      el('span', { class: 'book-title', text: book.title }),
      el('span', { class: 'book-sub', text: `${parts.join(' · ')} · ${info.text}` }),
      el('span', { class: 'book-progress', 'aria-hidden': 'true' }, bar))));
  if (deletable) {
    item.append(el('button', {
      type: 'button', class: 'icon-btn icon-btn-small book-delete', 'aria-label': `Delete “${book.title}”`, title: 'Delete',
      onclick: () => deleteBook(book),
    }, icon('trash')));
  }
  return item;
}

function renderLibrary() {
  const list = $('#library-list');
  list.replaceChildren(...state.books.map((book) => bookItem(book)));
  $('#library-empty').hidden = state.books.length > 0;
}

/** Refresh progress hints in place, so keyboard focus in the list is never lost. */
const renderLibraryProgress = debounce(() => {
  for (const item of $$('.book-item[data-book-id]')) {
    const book = state.books.find((b) => b.id === item.dataset.bookId);
    if (!book) continue;
    const info = progressInfo(book);
    const parts = [TYPE_LABELS[book.source_type] || book.source_type];
    if (book.section_count) parts.push(plural(book.section_count, 'section'));
    const sub = item.querySelector('.book-sub');
    if (sub) sub.textContent = `${parts.join(' · ')} · ${info.text}`;
    const bar = item.querySelector('.book-progress > span');
    if (bar) bar.style.width = `${Math.round(info.fraction * 100)}%`;
  }
}, 400);

async function refreshLibrary() {
  try {
    state.books = (await api('/api/library')).books || [];
  } catch (err) {
    toast(errorText(err), { kind: 'error' });
  }
  renderLibrary();
  if (!state.book) renderReader();
}

async function openBook(id, { warnings = null, focusContent = false } = {}) {
  hardStop();
  speaker.stop();
  player.playing = false;
  updatePlayUI();
  let data;
  try {
    data = await api(`/api/books/${enc(id)}`);
  } catch (err) {
    toast(errorText(err), { kind: 'error' });
    if (err.status === 404) refreshLibrary();
    return false;
  }
  state.book = data.book;
  state.toc = data.toc || [];
  state.warnings = warnings ?? data.warnings ?? [];
  state.section = null;
  state.index = 0;
  state.bookmarks = [];
  sectionCache.clear();
  local.set('lastBook', id);
  const known = state.books.findIndex((b) => b.id === id);
  if (known >= 0) state.books[known] = data.book;
  else state.books.unshift(data.book);
  renderLibrary();
  renderToc();
  renderBookmarks();
  renderReader();
  updateStatus();
  updatePlayUI();
  selectSidebarTab('contents');
  loadBookmarks();
  const progress = data.book.progress;
  const resumable = progress && state.toc.some((t) => t.id === progress.section_id);
  const sectionId = resumable ? progress.section_id : state.toc[0]?.id;
  if (!sectionId) {
    $('#article-host')?.replaceChildren(el('p', { class: 'loading-line', text: 'This file has no readable text.' }));
    return true;
  }
  const ok = await loadSection(sectionId, resumable ? progress.utterance : 0);
  if (!ok) {
    if (state.book?.id === id && !state.section) {
      $('#article-host')?.replaceChildren(el('p', { class: 'loading-line', text: "This part of the book couldn't be opened. Try another section from Contents." }));
    }
    return false;
  }
  closeOverlays();
  announce(resumable ? `Opened ${state.book.title}. Resuming where you left off.` : `Opened ${state.book.title}`);
  if (focusContent) $('#section-title')?.focus({ preventScroll: true });
  return true;
}

function closeBook() {
  hardStop();
  speaker.stop();
  player.playing = false;
  state.book = null;
  state.section = null;
  state.toc = [];
  state.bookmarks = [];
  state.index = 0;
  local.set('lastBook', null);
  renderToc();
  renderBookmarks();
  renderReader();
  renderStudy();
  updateStatus();
  updatePlayUI();
  updateMediaSession();
}

async function loadSection(sectionId, index = 0, { announceIt = false } = {}) {
  if (!state.book) return false;
  const seq = ++state.navSeq;
  const bookId = state.book.id;
  let data;
  try {
    data = await fetchSectionData(bookId, sectionId);
  } catch (err) {
    if (seq === state.navSeq) toast(errorText(err), { kind: 'error' });
    return false;
  }
  if (seq !== state.navSeq || state.book?.id !== bookId) return false;
  showSection(data, index, { announceIt, scrollTo: true });
  return true;
}

/** Make `data` the visible section, with `index` as the current sentence (-1 = last). */
function showSection(data, index, { announceIt = false, scrollTo = false } = {}) {
  state.section = data;
  const count = data.utterances.length;
  const target = index < 0 ? count - 1 : index;
  state.index = count ? clamp(target, 0, count - 1) : 0;
  if (!$('#article-host')) renderReader();
  renderSection();
  renderTocCurrent();
  updateMediaSession();
  renderStudy();
  const box = $('#reader');
  reader.programmaticUntil = performance.now() + 300;
  box.scrollTop = 0;
  setCurrent(state.index, { scroll: false });
  if (scrollTo && state.index > 0) scrollToCurrent({ force: true, instant: true });
  if (announceIt) announce(`Now reading: ${data.section.title}`);
}

async function reloadSectionUtterances() {
  if (!state.book || !state.section) return;
  const old = currentUtterance();
  const sectionId = state.section.section.id;
  const was = player.playing;
  hardStop();
  player.playing = false;
  sectionCache.clear();
  clips.clear();
  const seq = ++state.navSeq;
  let data;
  try {
    data = await fetchSectionData(state.book.id, sectionId);
  } catch (err) {
    toast(errorText(err), { kind: 'error' });
    updatePlayUI();
    return;
  }
  if (seq !== state.navSeq) return;
  let index = 0;
  if (old) {
    const list = data.utterances;
    const same = list.filter((u) => u.block === old.block && u.start <= old.start);
    const found = same.length ? same[same.length - 1] : list.find((u) => u.block >= old.block);
    index = found ? found.index : Math.max(0, list.length - 1);
  }
  showSection(data, index, { scrollTo: true });
  if (was) play();
  else updatePlayUI();
}

// ----- contents

function renderToc() {
  const nav = $('#toc');
  nav.replaceChildren();
  $('#toc-empty').hidden = state.toc.length > 0;
  if (!state.toc.length) return;
  const root = el('ul', { class: 'toc-list' });
  const stack = [{ level: 0, li: null, list: root }];
  for (const entry of state.toc) {
    const level = Math.max(1, Number(entry.level) || 1);
    while (stack.length > 1 && stack[stack.length - 1].level >= level) stack.pop();
    const parent = stack[stack.length - 1];
    let list = parent.list;
    if (!list) {
      list = el('ul', { class: 'toc-list' });
      parent.li.append(list);
      parent.list = list;
    }
    const isSlides = state.book?.source_type === 'pptx';
    const page = entry.page_start ? (isSlides ? `Slide ${entry.page_start}` : `p. ${entry.page_start}`) : '';
    const spokenPage = entry.page_start ? `, ${isSlides ? 'slide' : 'page'} ${entry.page_start}` : '';
    const button = el('button', {
      type: 'button', class: `toc-link level-${Math.min(level, 3)}`, dataset: { sid: entry.id },
      onclick: () => {
        goToSection(entry.id, { focusContent: true });
        closeOverlays();
      },
    }, el('span', { class: 'toc-title', text: entry.title }),
    page ? el('span', { class: 'toc-page', 'aria-hidden': 'true', text: page }) : null,
    spokenPage ? el('span', { class: 'visually-hidden', text: spokenPage }) : null);
    const li = el('li', null, button);
    list.append(li);
    stack.push({ level, li, list: null });
  }
  nav.append(root);
  renderTocCurrent();
}

function renderTocCurrent() {
  const id = state.section?.section?.id;
  for (const link of $$('#toc .toc-link')) {
    if (link.dataset.sid === id) {
      link.setAttribute('aria-current', 'location');
      scrollIntoPanel(link, $('#panel-contents'));
    } else {
      link.removeAttribute('aria-current');
    }
  }
}

// ----- bookmarks

async function loadBookmarks() {
  if (!state.book) return;
  const bookId = state.book.id;
  try {
    const data = await api(`/api/books/${enc(bookId)}/bookmarks`);
    if (state.book?.id !== bookId) return;
    state.bookmarks = data.bookmarks || [];
  } catch {
    state.bookmarks = [];
  }
  renderBookmarks();
  updateBookmarkButton();
}

function renderBookmarks() {
  const list = $('#bookmark-list');
  const sorted = [...state.bookmarks].sort((a, b) =>
    (sectionNumber(a.section_id) ?? 0) - (sectionNumber(b.section_id) ?? 0) || a.utterance - b.utterance);
  list.replaceChildren(...sorted.map((bm) => {
    const where = state.toc.find((t) => t.id === bm.section_id)?.title || '';
    return el('li', { class: 'bookmark-item' },
      el('button', {
        type: 'button', class: 'bookmark-open',
        onclick: () => {
          if (state.section?.section.id === bm.section_id) seekTo(bm.utterance);
          else goToSection(bm.section_id, { index: bm.utterance });
          closeOverlays();
        },
      }, icon('bookmarkFilled'), el('span', null,
        el('span', { class: 'bookmark-label', text: bm.label || 'Bookmark' }),
        el('span', { class: 'bookmark-where', text: `${where} · sentence ${bm.utterance + 1}` }))),
      el('button', {
        type: 'button', class: 'icon-btn icon-btn-small', 'aria-label': `Delete bookmark “${truncate(bm.label, 40)}”`,
        onclick: () => removeBookmark(bm),
      }, icon('trash')));
  }));
  $('#bookmark-empty').hidden = state.bookmarks.length > 0;
}

function bookmarkHere() {
  if (!state.section) return null;
  const id = state.section.section.id;
  return state.bookmarks.find((bm) => bm.section_id === id && bm.utterance === state.index) || null;
}

function updateBookmarkButton() {
  const button = $('#btn-bookmark');
  const marked = !!bookmarkHere();
  button.classList.toggle('is-marked', marked);
  setIcon(button.querySelector('.i'), marked ? 'bookmarkFilled' : 'bookmark');
  button.setAttribute('aria-label', marked ? 'Remove the bookmark on this sentence' : 'Bookmark this sentence');
  button.setAttribute('aria-pressed', marked ? 'true' : 'false');
}

async function toggleBookmark() {
  if (!state.book || !state.section) return;
  const existing = bookmarkHere();
  if (existing) {
    await removeBookmark(existing);
    return;
  }
  const u = currentUtterance();
  const label = truncate(utteranceText(u) || state.section.section.title, 90);
  const bookId = state.book.id;
  try {
    const data = await api(`/api/books/${enc(bookId)}/bookmarks`, {
      method: 'POST', body: { section_id: state.section.section.id, utterance: state.index, label },
    });
    if (state.book?.id !== bookId) return;
    state.bookmarks.push(data.bookmark);
    renderBookmarks();
    updateBookmarkButton();
    announce('Bookmark added');
    toast('Bookmark added.', { kind: 'success', action: { label: 'Show bookmarks', run: () => showBookmarksPanel() } });
  } catch (err) {
    toast(errorText(err), { kind: 'error' });
  }
}

async function removeBookmark(bm) {
  if (!state.book) return;
  try {
    await api(`/api/books/${enc(state.book.id)}/bookmarks/${enc(bm.id)}`, { method: 'DELETE' });
    state.bookmarks = state.bookmarks.filter((b) => b.id !== bm.id);
    renderBookmarks();
    updateBookmarkButton();
    announce('Bookmark removed');
  } catch (err) {
    toast(errorText(err), { kind: 'error' });
  }
}

function showBookmarksPanel() {
  selectSidebarTab('bookmarks');
  if (mqNarrow.matches) setSidebarOpen(true);
}

// ----- delete

async function deleteBook(book) {
  const ok = await confirmDialog({
    title: `Delete “${truncate(book.title, 60)}”?`,
    message: 'This removes the book, your place in it and its bookmarks from Athena. The original file on your computer is not touched.',
    okLabel: 'Delete',
  });
  if (!ok) return;
  try {
    await api(`/api/library/${enc(book.id)}`, { method: 'DELETE' });
  } catch (err) {
    toast(errorText(err), { kind: 'error' });
    return;
  }
  if (state.book?.id === book.id) closeBook();
  state.books = state.books.filter((b) => b.id !== book.id);
  renderLibrary();
  if (!state.book) renderReader();
  toast(`Deleted “${truncate(book.title, 60)}”.`);
}

// ----- import

function chooseFiles() {
  $('#file-input').click();
}

async function importFiles(files) {
  for (const file of files) await importOne(file);
}

function addTask(name) {
  const bar = el('div', { class: 'progress-bar' });
  const message = el('p', { class: 'task-msg', text: 'Uploading…' });
  const node = el('div', { class: 'task', role: 'group', 'aria-label': `Adding ${name}` },
    el('div', { class: 'task-head' }, el('span', { class: 'i' }, icon('file')), el('span', { class: 'task-name', text: name })),
    el('div', { class: 'progress', role: 'progressbar', 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-label': `Progress for ${name}` }, bar),
    message);
  $('#tasks').append(node);
  const progress = node.querySelector('[role="progressbar"]');
  return {
    update(fraction, text) {
      const pct = Math.round(clamp(fraction, 0, 1) * 100);
      bar.style.width = `${pct}%`;
      progress.setAttribute('aria-valuenow', String(pct));
      if (text) message.textContent = text;
    },
    done(text) {
      this.update(1, text);
      setTimeout(() => node.remove(), 2500);
    },
    fail(text) {
      node.classList.add('is-error');
      message.textContent = text;
      node.querySelector('.task-head').append(el('button', {
        type: 'button', class: 'icon-btn icon-btn-small', 'aria-label': 'Dismiss', onclick: () => node.remove(),
      }, icon('close')));
    },
  };
}

function uploadWithProgress(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/library/import');
    xhr.setRequestHeader('X-Destiny-Token', TOKEN);
    xhr.responseType = 'json';
    xhr.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    });
    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300 && xhr.response) resolve(xhr.response);
      else {
        if (xhr.status === 401) showSessionExpired();
        reject(new ApiError(xhr.status, detailFrom(xhr.status, xhr.response)));
      }
    });
    xhr.addEventListener('error', () => reject(new ApiError(0, STATUS_MESSAGES[0])));
    const form = new FormData();
    form.append('file', file, file.name);
    xhr.send(form);
  });
}

async function pollJob(jobId, onUpdate, { interval = 600, signal = null } = {}) {
  for (;;) {
    if (signal?.aborted) throw new DOMException('Stopped', 'AbortError');
    const job = await api(`/api/jobs/${enc(jobId)}`);
    onUpdate?.(job);
    if (['done', 'error', 'cancelled'].includes(job.status)) return job;
    await wait(interval);
  }
}

async function importOne(file) {
  if (file.size > MAX_UPLOAD_BYTES) {
    toast(`“${file.name}” is too big to import (the limit is 300 MB).`, { kind: 'error' });
    return;
  }
  const task = addTask(file.name);
  announce(`Adding ${file.name}`);
  try {
    const { job_id: jobId } = await uploadWithProgress(file, (f) => task.update(f, `Uploading… ${Math.round(f * 100)}%`));
    task.update(0, 'Reading the file…');
    const job = await pollJob(jobId, (j) => task.update(j.progress || 0, j.message || 'Reading the file…'));
    if (job.status === 'done') {
      const book = job.result?.book;
      task.done('Added to your library');
      await refreshLibrary();
      if (book) {
        await openBook(book.id, { warnings: job.result?.warnings || [], focusContent: true });
        toast(`Added “${truncate(book.title, 60)}”.`, { kind: 'success' });
      }
    } else if (job.status === 'cancelled') {
      task.fail('Cancelled');
    } else {
      const message = job.error || `Couldn't read “${file.name}”.`;
      task.fail(message);
      toast(message, { kind: 'error' });
    }
  } catch (err) {
    const message = errorText(err);
    task.fail(message);
    toast(message, { kind: 'error' });
  }
}

// ----- paste

function openPaste() {
  $('#paste-name').value = '';
  $('#paste-text').value = '';
  updatePasteCount();
  openDialog($('#paste-dialog'), { focus: $('#paste-text') });
}

function updatePasteCount() {
  const text = $('#paste-text').value.trim();
  const words = text ? text.split(/\s+/).length : 0;
  $('#paste-count').textContent = words ? `${plural(words, 'word')}` : '';
  $('#paste-submit').disabled = !text;
}

async function submitPaste(event) {
  event.preventDefault();
  const text = $('#paste-text').value.trim();
  if (!text) return;
  const firstLine = text.split('\n')[0].replace(/^#+\s*/, '');
  const title = $('#paste-name').value.trim() || truncate(firstLine, 60) || 'Pasted text';
  const button = $('#paste-submit');
  button.disabled = true;
  try {
    const data = await api('/api/library/paste', { method: 'POST', body: { title, text } });
    $('#paste-dialog').close('done');
    await refreshLibrary();
    await openBook(data.book.id, { focusContent: true });
    toast(`Added “${truncate(data.book.title, 60)}”.`, { kind: 'success' });
  } catch (err) {
    toast(errorText(err), { kind: 'error' });
  } finally {
    button.disabled = false;
  }
}

// ----- drag and drop

function setupDragAndDrop() {
  const overlay = $('#drop-overlay');
  let depth = 0;
  const hasFiles = (event) => Array.from(event.dataTransfer?.types || []).includes('Files');
  window.addEventListener('dragenter', (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    depth += 1;
    overlay.hidden = false;
  });
  window.addEventListener('dragover', (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
  });
  window.addEventListener('dragleave', (event) => {
    if (!hasFiles(event)) return;
    depth = Math.max(0, depth - 1);
    if (!depth) overlay.hidden = true;
  });
  window.addEventListener('drop', (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    depth = 0;
    overlay.hidden = true;
    const files = Array.from(event.dataTransfer.files || []);
    if (files.length) importFiles(files);
  });
}

// =====================================================================
// 10. Study helper (AI)
// =====================================================================

const study = { threads: new Map(), pendingSelection: '' };

function aiReady() {
  return !!(state.status?.ai?.enabled && state.status?.ai?.configured);
}

function threadKey() {
  return state.book && state.section ? `${state.book.id}|${state.section.section.id}` : '';
}

function currentThread() {
  const key = threadKey();
  if (!key) return null;
  if (!study.threads.has(key)) study.threads.set(key, []);
  return study.threads.get(key);
}

function renderStudy() {
  const context = $('#study-context');
  context.textContent = state.section ? `About: ${state.section.section.title}` : 'Open something to read first';
  updateStudyActions();
  const box = $('#study-thread');
  box.replaceChildren();
  const thread = currentThread();
  if (!thread || !thread.length) {
    box.append(studyIntro());
    return;
  }
  for (const item of thread) {
    item.node = renderStudyItem(item, thread);
    box.append(item.node);
  }
  box.scrollTop = box.scrollHeight;
}

function studyIntro() {
  const node = aiReady()
    ? el('div', { class: 'study-intro' },
      icon('chat'),
      el('p', { text: 'Ask Athena anything about what you are reading. Select some text and choose Explain, or use the buttons above.' }))
    : setupCard('');
  node.dataset.intro = '1';
  return node;
}

function setupCard(detail) {
  return el('div', { class: 'card card-setup' },
    el('p', { class: 'card-title' }, icon('bulb'), 'Study helper'),
    el('h3', { text: 'Turn on the Study helper' }),
    el('p', { class: 'card-body', text: 'Ask Athena to explain tricky passages in plain words, answer your questions, summarize sections and quiz you. It is powered by Claude, from Anthropic.' }),
    el('div', { class: 'privacy-note' }, icon('lock'),
      el('span', { text: 'When you use it, the text you ask about (and the section you are reading) is sent to Anthropic. Everything else stays on this computer.' })),
    detail ? el('p', { class: 'server-detail', text: detail }) : null,
    el('div', { class: 'card-foot' },
      el('button', { type: 'button', class: 'btn btn-primary btn-small', onclick: () => openSettings('privacy') }, 'Open Settings')));
}

function readAloudButton(text, label = 'Read aloud') {
  return el('button', {
    type: 'button', class: 'btn btn-small btn-secondary read-aloud', 'aria-pressed': 'false',
    onclick: (event) => speaker.toggle(text, event.currentTarget),
  }, el('span', { class: 'i' }, icon('speaker')), el('span', { class: 'btn-text', text: label }));
}

function renderStudyItem(item, thread) {
  switch (item.type) {
    case 'user':
      return el('div', { class: 'bubble-user', text: item.text });
    case 'loading':
      return el('div', { class: 'card', role: 'status' },
        el('div', { class: 'thinking' }, el('span', { class: 'dots', 'aria-hidden': 'true' }, el('span'), el('span'), el('span')),
          el('span', { text: `${item.label || 'Thinking'}…` })));
    case 'setup':
      return setupCard(item.detail);
    case 'error':
      return el('div', { class: 'card card-error', role: 'alert' },
        el('p', { class: 'card-title' }, icon('alert'), 'Something went wrong'),
        el('p', { class: 'card-body', text: item.text }),
        item.retry ? el('div', { class: 'card-foot' },
          el('button', {
            type: 'button', class: 'btn btn-small btn-secondary',
            onclick: () => {
              removeItem(thread, item);
              item.retry();
            },
          }, 'Try again')) : null);
    case 'quiz':
      return renderQuiz(item, thread);
    default:
      return el('div', { class: 'card answer-card' },
        el('p', { class: 'card-title' }, icon(item.icon || 'bulb'), item.title || 'Answer'),
        item.quote ? el('p', { class: 'card-quote', text: item.quote }) : null,
        el('div', { class: 'card-body' }, textParagraphs(item.text)),
        el('div', { class: 'card-foot' }, readAloudButton(item.text)));
  }
}

function isVisibleThread(thread) {
  return thread && thread === study.threads.get(threadKey());
}

function pushItem(thread, item) {
  thread.push(item);
  if (isVisibleThread(thread)) {
    const box = $('#study-thread');
    box.querySelector('[data-intro]')?.remove();
    item.node = renderStudyItem(item, thread);
    box.append(item.node);
    scrollThreadToEnd();
  }
  return item;
}

function replaceItem(thread, item, next) {
  const index = thread.indexOf(item);
  if (index < 0) return;
  thread[index] = next;
  if (isVisibleThread(thread) && item.node?.isConnected) {
    next.node = renderStudyItem(next, thread);
    item.node.replaceWith(next.node);
    scrollIntoPanel(next.node, $('#study-thread'));
  }
}

function removeItem(thread, item) {
  const index = thread.indexOf(item);
  if (index >= 0) thread.splice(index, 1);
  item.node?.remove();
}

async function runAI({ path, body, title, iconName = 'bulb', quote = '', userText = '', chat = false, toItem = null }) {
  if (!state.book || !state.section) {
    toast('Open something to read first.');
    return;
  }
  setStudyOpen(true);
  const thread = currentThread();
  if (userText) pushItem(thread, { type: 'user', text: userText, chat });
  const loading = pushItem(thread, { type: 'loading', label: title });
  const retry = () => runAI({ path, body, title, iconName, quote, userText: '', chat, toItem });
  try {
    const data = await api(path, { method: 'POST', body });
    const item = toItem ? toItem(data) : { type: 'answer', title, icon: iconName, quote, text: String(data.text || ''), chat };
    replaceItem(thread, loading, item);
    announce(`${title} is ready`);
  } catch (err) {
    if (err.status === 409) {
      replaceItem(thread, loading, { type: 'setup', detail: errorText(err) });
      refreshStatus();
    } else {
      replaceItem(thread, loading, { type: 'error', text: errorText(err), retry });
    }
  }
}

function explainText(passage) {
  const text = truncate(passage, 6000);
  if (!text) {
    toast('Select some text, or start reading, and then choose Explain.');
    return;
  }
  runAI({
    path: '/api/ai/explain', title: 'Explanation', iconName: 'bulb', quote: truncate(text, 220),
    body: { book_id: state.book.id, section_id: state.section.section.id, passage: text },
  });
}

function explainCurrent() {
  const selected = study.pendingSelection || readerSelection()?.text || '';
  study.pendingSelection = '';
  explainText(selected || utteranceText(currentUtterance()));
}

function explainBlock(blockIndex) {
  const block = state.section?.blocks[blockIndex];
  if (!block) return;
  runAI({
    path: '/api/ai/narrate', title: block.kind === 'table' ? 'About this table' : 'About this equation',
    iconName: block.kind === 'table' ? 'table' : 'sigma', quote: truncate(block.text, 160),
    body: { book_id: state.book.id, section_id: state.section.section.id, block: blockIndex },
  });
}

function summarize(length) {
  runAI({
    path: '/api/ai/summarize', title: length === 'detailed' ? 'Full summary' : 'Quick summary', iconName: 'summary',
    body: { book_id: state.book.id, section_id: state.section.section.id, length },
  });
}

function describePage() {
  const page = currentPage();
  if (!page) return;
  runAI({
    path: '/api/ai/describe-page', title: `Figures on page ${page}`, iconName: 'image',
    body: { book_id: state.book.id, page },
  });
}

function askQuestion(question) {
  const thread = currentThread();
  if (!thread) {
    toast('Open something to read first.');
    return;
  }
  const history = [];
  for (let i = 0; i < thread.length - 1; i += 1) {
    const a = thread[i];
    const b = thread[i + 1];
    if (a.type === 'user' && a.chat && b.type === 'answer' && b.chat) {
      history.push({ role: 'user', content: a.text }, { role: 'assistant', content: b.text });
    }
  }
  runAI({
    path: '/api/ai/ask', title: 'Answer', iconName: 'chat', userText: question, chat: true,
    body: { book_id: state.book.id, section_id: state.section.section.id, question, history: history.slice(-12) },
  });
}

function startQuiz() {
  runAI({
    path: '/api/ai/quiz', title: 'Quiz', iconName: 'quiz',
    body: { book_id: state.book.id, section_id: state.section.section.id, count: 5 },
    toItem: (data) => {
      const questions = (data.questions || []).filter((q) => q && Array.isArray(q.choices) && q.choices.length);
      if (!questions.length) return { type: 'error', text: "The quiz didn't come back right. Please try again." };
      return { type: 'quiz', questions, current: 0, picks: [], finished: false };
    },
  });
}

function rerenderItem(thread, item) {
  if (!isVisibleThread(thread) || !item.node?.isConnected) return;
  const node = renderStudyItem(item, thread);
  item.node.replaceWith(node);
  item.node = node;
}

function renderQuiz(item, thread) {
  const total = item.questions.length;
  if (item.finished) {
    const score = item.picks.filter((pick, i) => pick === item.questions[i].answer_index).length;
    const verdict = score === total ? 'Perfect score. Great work!' : score >= total / 2 ? 'Nice work. Review the ones you missed.' : 'Keep going. Re-listening to the section can help.';
    return el('div', { class: 'card quiz-card' },
      el('p', { class: 'card-title' }, icon('quiz'), 'Quiz results'),
      el('p', { class: 'quiz-score', text: `You got ${score} of ${total} right` }),
      el('p', { class: 'card-body', text: verdict }),
      el('div', { class: 'card-foot' },
        el('button', { type: 'button', class: 'btn btn-small btn-primary', onclick: startQuiz }, 'New quiz'),
        el('button', {
          type: 'button', class: 'btn btn-small btn-secondary',
          onclick: () => {
            item.finished = false;
            item.current = 0;
            item.review = true;
            rerenderItem(thread, item);
          },
        }, 'Review answers')));
  }
  const q = item.questions[item.current];
  const picked = item.picks[item.current];
  const answered = picked !== undefined;
  const letters = 'ABCDEFGH';
  const questionSpeech = `${q.question} ${q.choices.map((c, i) => `${letters[i]}: ${c}.`).join(' ')}`;
  const choices = el('ul', { class: 'quiz-choices' }, q.choices.map((choice, i) => {
    let cls = 'quiz-choice';
    let mark = '';
    if (answered && i === q.answer_index) {
      cls += ' is-correct';
      mark = i === picked ? 'Your answer: correct' : 'Correct answer';
    } else if (answered && i === picked) {
      cls += ' is-wrong';
      mark = 'Your answer';
    }
    return el('li', null, el('button', {
      type: 'button', class: cls, disabled: answered,
      onclick: () => {
        item.picks[item.current] = i;
        rerenderItem(thread, item);
        announce(i === q.answer_index ? 'Correct!' : `Not quite. The answer is ${letters[q.answer_index]}.`);
        item.node?.querySelector('.quiz-feedback')?.focus({ preventScroll: false });
      },
    }, el('span', { class: 'quiz-letter', 'aria-hidden': 'true', text: letters[i] || '?' }),
    el('span', { class: 'visually-hidden', text: `Choice ${letters[i] || ''}: ` }),
    el('span', { text: choice }),
    mark ? el('span', { class: 'mark', text: mark }) : null));
  }));
  let feedback = null;
  if (answered) {
    const right = picked === q.answer_index;
    const isLast = item.current >= total - 1;
    const explanation = String(q.explanation || '');
    feedback = el('div', { class: 'quiz-feedback', tabindex: '-1' },
      el('p', { class: `quiz-verdict ${right ? 'good' : 'bad'}`, text: right ? '✓ Correct!' : `✗ Not quite. The answer is ${letters[q.answer_index]}: ${q.choices[q.answer_index]}` }),
      explanation ? el('p', { text: explanation }) : null,
      el('div', { class: 'card-foot' },
        explanation ? readAloudButton(explanation, 'Read explanation') : null,
        el('button', {
          type: 'button', class: 'btn btn-small btn-primary',
          onclick: () => {
            if (isLast) item.finished = true;
            else item.current += 1;
            rerenderItem(thread, item);
            item.node?.querySelector('.quiz-choice, .quiz-score')?.focus?.();
          },
        }, isLast ? (item.review ? 'Back to results' : 'See my score') : 'Next question')));
  }
  return el('div', { class: 'card quiz-card' },
    el('p', { class: 'card-title' }, icon('quiz'), 'Quiz'),
    el('p', { class: 'quiz-progress', text: `Question ${item.current + 1} of ${total}` }),
    el('p', { class: 'quiz-q', text: q.question }),
    choices,
    answered ? null : el('div', { class: 'card-foot' }, readAloudButton(questionSpeech, 'Read question')),
    feedback);
}

function updateStudyActions() {
  const has = !!(state.book && state.section);
  for (const id of ['ai-explain', 'ai-summary-short', 'ai-summary-long', 'ai-quiz']) $(`#${id}`).disabled = !has;
  $('#ask-input').disabled = !has;
  const block = has ? currentBlock() : null;
  const narrate = $('#ai-narrate');
  if (block && (block.kind === 'table' || block.kind === 'equation')) {
    narrate.hidden = false;
    const isTable = block.kind === 'table';
    $('#ai-narrate-label').textContent = isTable ? 'Explain this table' : 'Explain this equation';
    const holder = narrate.querySelector('.i');
    if (holder.dataset.icon !== (isTable ? 'table' : 'sigma')) setIcon(holder, isTable ? 'table' : 'sigma');
  } else {
    narrate.hidden = true;
  }
  const describe = $('#ai-describe');
  const page = has && state.book.source_type === 'pdf' ? currentPage() : null;
  describe.hidden = !page;
  if (page) $('#ai-describe-label').textContent = `Describe figures on page ${page}`;
}

function setupStudy() {
  const actions = $('#study-actions');
  actions.addEventListener('pointerdown', () => {
    study.pendingSelection = readerSelection()?.text || '';
  });
  $('#ai-explain').addEventListener('click', explainCurrent);
  $('#ai-summary-short').addEventListener('click', () => summarize('short'));
  $('#ai-summary-long').addEventListener('click', () => summarize('detailed'));
  $('#ai-quiz').addEventListener('click', startQuiz);
  $('#ai-narrate').addEventListener('click', () => {
    const u = currentUtterance();
    if (u && u.block >= 0) explainBlock(u.block);
  });
  $('#ai-describe').addEventListener('click', describePage);
  const input = $('#ask-input');
  const autosize = () => {
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight + 2, 160)}px`;
  };
  input.addEventListener('input', autosize);
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $('#ask-form').requestSubmit();
    }
  });
  $('#ask-form').addEventListener('submit', (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    input.value = '';
    autosize();
    askQuestion(question);
  });
}

// =====================================================================
// 11. Dialogs, panels and tabs
// =====================================================================

const mqNarrow = window.matchMedia('(max-width: 899px)');
const mqStudyOverlay = window.matchMedia('(max-width: 1199px)');

function openDialog(dialog, { focus = null } = {}) {
  if (dialog.open) return;
  const opener = document.activeElement;
  $('#sel-bubble').hidden = true;
  toggleSleepMenu(false);
  dialog.returnValue = '';
  dialog.showModal();
  const target = focus || dialog.querySelector('[autofocus]')
    || dialog.querySelector('input:not([type="hidden"]):not([disabled]), textarea:not([disabled]), select:not([disabled])')
    || dialog.querySelector('[role="tab"][aria-selected="true"]')
    || dialog.querySelector('button:not([disabled])');
  target?.focus();
  dialog.addEventListener('close', () => {
    if (opener && opener.isConnected && typeof opener.focus === 'function') opener.focus({ preventScroll: true });
  }, { once: true });
}

function setupDialogs() {
  for (const dialog of $$('dialog')) {
    let downOnBackdrop = false;
    dialog.addEventListener('pointerdown', (event) => {
      downOnBackdrop = event.target === dialog;
    });
    dialog.addEventListener('click', (event) => {
      if (event.target === dialog && downOnBackdrop) dialog.close('dismiss');
      else if (event.target.closest('[data-close]')) dialog.close('dismiss');
    });
  }
}

function confirmDialog({ title, message, okLabel = 'OK' }) {
  const dialog = $('#confirm-dialog');
  $('#confirm-title').textContent = title;
  $('#confirm-message').textContent = message;
  $('#confirm-ok').textContent = okLabel;
  return new Promise((resolve) => {
    dialog.addEventListener('close', () => resolve(dialog.returnValue === 'ok'), { once: true });
    openDialog(dialog, { focus: $('#confirm-cancel') });
  });
}

function setupTabs(tablist, onSelect) {
  const tabs = $$('[role="tab"]', tablist);
  const select = (tab, focus = false) => {
    for (const t of tabs) {
      const on = t === tab;
      t.setAttribute('aria-selected', on ? 'true' : 'false');
      t.tabIndex = on ? 0 : -1;
      const panel = document.getElementById(t.getAttribute('aria-controls'));
      if (panel) panel.hidden = !on;
    }
    if (focus) tab.focus();
    onSelect?.(tab);
  };
  tablist.addEventListener('click', (event) => {
    const tab = event.target.closest('[role="tab"]');
    if (tab) select(tab);
  });
  tablist.addEventListener('keydown', (event) => {
    const index = tabs.indexOf(document.activeElement);
    if (index < 0) return;
    let next = null;
    if (['ArrowRight', 'ArrowDown'].includes(event.key)) next = tabs[(index + 1) % tabs.length];
    else if (['ArrowLeft', 'ArrowUp'].includes(event.key)) next = tabs[(index - 1 + tabs.length) % tabs.length];
    else if (event.key === 'Home') next = tabs[0];
    else if (event.key === 'End') next = tabs[tabs.length - 1];
    if (next) {
      event.preventDefault();
      event.stopPropagation();
      select(next, true);
    }
  });
  return (id) => {
    const tab = tabs.find((t) => t.id === id);
    if (tab) select(tab);
  };
}

let selectSidebarTabById = () => {};
function selectSidebarTab(name) {
  selectSidebarTabById(`tab-${name}`);
}

function setSidebarOpen(open) {
  if (!mqNarrow.matches) open = false;
  if (open) setStudyOpen(false);
  document.body.classList.toggle('sidebar-open', open);
  syncPanels();
  if (open) $('#btn-sidebar-close').focus();
}

function setStudyOpen(open, { focus = false } = {}) {
  const was = document.body.classList.contains('study-open');
  if (open && mqNarrow.matches) document.body.classList.remove('sidebar-open');
  document.body.classList.toggle('study-open', open);
  syncPanels();
  if (open && !was) {
    renderStudy();
    if (focus) $('#btn-study-close').focus();
  }
  if (!open && was && $('#study').contains(document.activeElement)) $('#btn-study').focus();
}

function closeOverlays() {
  if (mqNarrow.matches && document.body.classList.contains('sidebar-open')) {
    document.body.classList.remove('sidebar-open');
    syncPanels();
  }
  if (mqStudyOverlay.matches && document.body.classList.contains('study-open')) setStudyOpen(false);
}

function syncPanels() {
  const sidebarOpen = document.body.classList.contains('sidebar-open');
  const studyOpen = document.body.classList.contains('study-open');
  $('#sidebar').inert = mqNarrow.matches && !sidebarOpen;
  $('#study').inert = !studyOpen;
  $('#btn-sidebar').setAttribute('aria-expanded', sidebarOpen ? 'true' : 'false');
  $('#btn-study').setAttribute('aria-expanded', studyOpen ? 'true' : 'false');
  $('#scrim').hidden = !((mqNarrow.matches && sidebarOpen) || (mqStudyOverlay.matches && studyOpen));
}

// =====================================================================
// 12. Settings
// =====================================================================

const READING_FIELDS = [
  { key: 'read_headings', type: 'bool', label: 'Read headings', hint: 'Say chapter and section headings before the text under them.' },
  { key: 'announce_section_title', type: 'bool', label: 'Announce section names', hint: 'Say the name of a section when it does not start with a heading.' },
  { key: 'read_captions', type: 'bool', label: 'Read captions', hint: 'Read the captions under figures and tables, like “Figure 2.1 …”.' },
  { key: 'read_figures', type: 'bool', label: 'Describe pictures', hint: 'Read the description of a picture when the file includes one.' },
  { key: 'read_footnotes', type: 'bool', label: 'Read footnotes', hint: 'Off by default, so footnotes don’t interrupt the flow.' },
  {
    key: 'tables', type: 'choice', label: 'Tables', hint: 'How tables are read out loud.',
    options: [['rows', 'Read each row'], ['announce', 'Just say there is a table'], ['skip', 'Skip tables']],
  },
  { key: 'equations', type: 'flag', on: 'read', off: 'skip', label: 'Read equations', hint: 'Read math formulas out loud as well as possible.' },
  {
    key: 'code', type: 'choice', label: 'Computer code', hint: 'What to do with blocks of programming code.',
    options: [['announce', 'Say “code block” and move on'], ['read', 'Read the code'], ['skip', 'Skip it silently']],
  },
  { key: 'citations', type: 'flag', on: 'remove', off: 'keep', label: 'Skip citations', hint: 'Leave out references like “(Smith et al., 2019)” and “[12]” when reading.' },
  {
    key: 'urls', type: 'choice', label: 'Web addresses', hint: 'How links like “https://www.example.com/page” are read.',
    options: [['domain', 'Say just the website name'], ['full', 'Read the whole address'], ['skip', 'Skip web addresses']],
  },
  { key: 'expand_abbreviations', type: 'bool', label: 'Expand abbreviations', hint: 'Say “for example” for “e.g.” and “Figure” for “Fig.”.' },
  {
    key: 'max_chars', type: 'range', min: 80, max: 2000, step: 20, label: 'Longest phrase',
    hint: 'Very long sentences are split into pieces no longer than this. Shorter pieces start sooner.',
  },
];

const settingsUI = { selectTab: () => {}, voiceJobs: new Map(), catalog: [] };
const reloadUtterancesSoon = debounce(() => reloadSectionUtterances(), 300);

function openSettings(tab = 'voice') {
  renderSettings();
  const dialog = $('#settings-dialog');
  if (!dialog.open) openDialog(dialog);
  settingsUI.selectTab(`stab-${tab}`);
  $(`#stab-${tab}`)?.focus();
}

function renderSettings() {
  if (!state.settings) return;
  renderEngineList();
  renderSettingsVoice();
  renderReadingOptions();
  renderAppearance();
  renderPrivacy();
  loadCatalog();
}

function renderEngineList() {
  const box = $('#engine-list');
  for (const child of [...box.children]) if (child.tagName !== 'LEGEND') child.remove();
  const resolved = state.status?.resolved_engine;
  const choice = state.settings.tts_engine || 'auto';
  const options = [{
    name: 'auto', label: 'Automatic', available: true,
    note: `Uses the best voice available on this computer${resolved ? ` (right now: ${engineLabel(resolved)})` : ''}.`,
  }];
  for (const e of state.status?.engines || []) {
    let note = ENGINE_BLURBS[e.name] || '';
    let badge = '';
    if (!e.available) {
      note = e.reason || 'Not available on this computer.';
      badge = 'Not available';
    } else if (e.online && !state.settings.online_tts_allowed) {
      note = 'Turn on “Online voices” in Privacy & AI to use these voices.';
      badge = 'Off in Privacy settings';
    } else if (e.name === 'browser' && !hasSpeechSynthesis()) {
      note = "This browser can't speak text.";
      badge = 'Not available';
    }
    options.push({ name: e.name, label: e.label, online: e.online, available: !badge, badge, note });
  }
  for (const option of options) {
    const id = `engine-${option.name}`;
    const input = el('input', {
      type: 'radio', name: 'engine', id, value: option.name, disabled: !option.available,
      'aria-describedby': `${id}-note`,
    });
    input.checked = option.name === choice;
    const card = el('label', { class: `engine-card${input.checked ? ' is-selected' : ''}${option.available ? '' : ' is-disabled'}`, for: id },
      input,
      el('span', { class: 'engine-text' },
        el('span', { class: 'engine-name' }, option.label,
          option.online ? el('span', { class: 'badge badge-warn' }, '☁ Online') : null,
          option.badge ? el('span', { class: 'badge', text: option.badge }) : null,
          option.name === resolved && choice === 'auto' ? el('span', { class: 'badge badge-accent', text: 'In use' }) : null),
        el('span', { class: 'engine-note', id: `${id}-note`, text: option.note })));
    input.addEventListener('change', async () => {
      if (!input.checked) return;
      if (await saveSettings({ tts_engine: option.name })) {
        await refreshStatus();
        renderEngineList();
        renderSettingsVoice();
        restartPlayback();
      }
    });
    box.append(card);
  }
}

function renderSettingsVoice() {
  const select = $('#set-voice');
  const hint = $('#set-voice-hint');
  const pref = state.settings.tts_engine || 'auto';
  const engine = pref === 'auto' ? (state.status?.resolved_engine || 'browser') : pref;
  select.replaceChildren();
  select.dataset.engine = engine;
  if (engine === 'browser') {
    const voices = usableBrowserVoices();
    for (const v of voices) select.append(el('option', { value: v.voiceURI, text: `${v.name}${v.localService === false ? ' ☁' : ''} (${v.lang})` }));
    select.value = pickBrowserVoice(state.settings.voices?.browser)?.voiceURI || '';
    hint.textContent = voices.length
      ? 'Browser voices marked ☁ send text to an online service.'
      : (hasSpeechSynthesis() ? 'This browser has no voices available right now.' : "This browser can't speak text.");
  } else {
    const voices = state.serverVoices[engine] || [];
    for (const v of voices) select.append(el('option', { value: v.id, text: `${v.name}${v.online ? ' ☁' : ''}` }));
    select.value = activeVoice(engine);
    hint.textContent = voices.length ? '' : (engine === 'piper' ? 'No natural voices installed yet. Download one below.' : 'No voices found for this engine.');
  }
  select.disabled = !select.options.length;
  if (!select.options.length) select.append(el('option', { value: '', text: 'No voices available' }));
}

async function loadCatalog() {
  const list = $('#catalog-list');
  if (!engineInfo('piper')) {
    list.replaceChildren(el('li', { class: 'hint', text: 'Natural voices are not available in this copy of Athena.' }));
    return;
  }
  try {
    settingsUI.catalog = (await api('/api/voices/catalog')).voices || [];
  } catch (err) {
    list.replaceChildren(el('li', { class: 'hint', text: errorText(err) }));
    return;
  }
  renderCatalog();
}

function renderCatalog() {
  const list = $('#catalog-list');
  const catalog = settingsUI.catalog;
  const banner = $('#catalog-banner');
  banner.replaceChildren();
  if (catalog.length && !catalog.some((v) => v.installed)) {
    banner.append(el('div', { class: 'notice notice-accent' }, icon('info'),
      el('div', { class: 'notice-body' }, el('p', {
        text: 'You don’t have a natural voice yet. We recommend Lessac: clear, friendly and about 60 MB. After downloading, everything stays offline.',
      }))));
  }
  list.replaceChildren(...catalog.map((voice) => {
    const recommended = voice.id === 'en_US-lessac-medium';
    const job = settingsUI.voiceJobs.get(voice.id);
    let side;
    if (voice.installed) {
      side = el('span', { class: 'badge badge-success' }, icon('check'), 'Installed');
    } else if (job && !['done', 'error', 'cancelled'].includes(job.status)) {
      const bar = el('div', { class: 'progress-bar' });
      bar.style.width = `${Math.round((job.progress || 0) * 100)}%`;
      side = el('div', null,
        el('div', { class: 'catalog-status', text: job.message || 'Downloading…' }),
        el('div', { class: 'progress', role: 'progressbar', 'aria-label': `Downloading ${voice.name}`, 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': String(Math.round((job.progress || 0) * 100)) }, bar));
    } else {
      side = el('div', null,
        job && job.status === 'error' ? el('div', { class: 'catalog-status', text: job.error || 'Download failed.' }) : null,
        el('button', {
          type: 'button', class: `btn btn-small ${recommended ? 'btn-primary' : 'btn-secondary'}`,
          'aria-label': `Download ${voice.name}`, onclick: () => downloadVoice(voice),
        }, el('span', { class: 'i' }, icon('download')), job && job.status === 'error' ? 'Try again' : 'Download'));
    }
    const meta = [voice.language, voice.quality ? `${voice.quality} quality` : '', voice.size_mb ? `${Math.round(voice.size_mb)} MB` : ''].filter(Boolean).join(' · ');
    return el('li', { class: `catalog-item${recommended ? ' is-recommended' : ''}`, dataset: { voiceId: voice.id } },
      el('div', { class: 'catalog-text' },
        el('div', { class: 'catalog-name' }, voice.name, recommended ? el('span', { class: 'badge badge-accent', text: 'Recommended' }) : null),
        el('p', { class: 'catalog-meta', text: meta }),
        voice.description ? el('p', { class: 'catalog-desc', text: voice.description }) : null),
      el('div', { class: 'catalog-side' }, side));
  }));
}

async function downloadVoice(voice) {
  let jobId;
  try {
    jobId = (await api('/api/voices/download', { method: 'POST', body: { voice_id: voice.id } })).job_id;
  } catch (err) {
    toast(errorText(err), { kind: 'error' });
    return;
  }
  settingsUI.voiceJobs.set(voice.id, { status: 'running', progress: 0, message: 'Starting download…' });
  renderCatalog();
  announce(`Downloading ${voice.name}`);
  try {
    const job = await pollJob(jobId, (j) => {
      settingsUI.voiceJobs.set(voice.id, j);
      if ($('#settings-dialog').open) renderCatalog();
    }, { interval: 700 });
    if (job.status === 'done') {
      settingsUI.voiceJobs.delete(voice.id);
      toast(`${voice.name} is ready to use.`, { kind: 'success' });
      announce(`${voice.name} installed`);
      if (local.get('voiceBannerDismissed', false) === false) $('#voice-banner')?.remove();
      await refreshStatus();
      await loadCatalog();
      if ($('#settings-dialog').open) {
        renderEngineList();
        renderSettingsVoice();
      }
      checkVoiceBanner();
    } else if (job.status === 'error') {
      toast(job.error || `Couldn't download ${voice.name}.`, { kind: 'error' });
      renderCatalog();
    } else {
      settingsUI.voiceJobs.delete(voice.id);
      renderCatalog();
    }
  } catch (err) {
    settingsUI.voiceJobs.set(voice.id, { status: 'error', error: errorText(err) });
    renderCatalog();
  }
}

function renderReadingOptions() {
  const box = $('#reading-options');
  const values = state.settings.reading || {};
  box.replaceChildren(...READING_FIELDS.map((field) => {
    const id = `ro-${field.key}`;
    const hintId = `${id}-hint`;
    let control;
    if (field.type === 'bool' || field.type === 'flag') {
      control = el('input', { type: 'checkbox', role: 'switch', class: 'switch', id, 'aria-describedby': hintId });
      control.checked = field.type === 'bool' ? !!values[field.key] : values[field.key] === field.on;
      control.addEventListener('change', () => {
        const value = field.type === 'bool' ? control.checked : (control.checked ? field.on : field.off);
        saveReading({ [field.key]: value });
      });
    } else if (field.type === 'choice') {
      control = el('select', { class: 'select', id, 'aria-describedby': hintId },
        field.options.map(([value, label]) => el('option', { value, text: label })));
      control.value = values[field.key];
      control.addEventListener('change', () => saveReading({ [field.key]: control.value }));
    } else {
      const output = el('output', { for: id, text: `${values[field.key]} letters` });
      const range = el('input', { type: 'range', id, min: field.min, max: field.max, step: field.step, 'aria-describedby': hintId });
      range.value = String(values[field.key] ?? 400);
      range.addEventListener('input', () => {
        output.textContent = `${range.value} letters`;
      });
      range.addEventListener('change', () => saveReading({ [field.key]: Number(range.value) }));
      control = el('div', { class: 'option-range' }, range, output);
    }
    return el('div', { class: 'option' },
      el('div', { class: 'option-text' },
        el('label', { class: 'option-label', for: id, text: field.label }),
        el('p', { class: 'hint', id: hintId, text: field.hint })),
      control);
  }));
}

async function saveReading(changes) {
  if (await saveSettings({ reading: changes })) reloadUtterancesSoon();
  else renderReadingOptions();
}

function renderAppearance() {
  const theme = state.settings.theme || 'system';
  for (const radio of $$('input[name="theme"]')) radio.checked = radio.value === theme;
  const scale = clamp(state.settings.font_scale || 1, 0.8, 1.6);
  $('#font-scale').value = String(scale);
  $('#font-scale-value').textContent = `${Math.round(scale * 100)}%`;
  $('#comfortable').checked = !!local.get('comfortable', false);
  const current = local.get('highlight', 'sun');
  const swatches = $('#highlight-choice');
  swatches.replaceChildren(...HIGHLIGHTS.map(([value, label]) => {
    const input = el('input', { type: 'radio', name: 'highlight', value, 'aria-label': label });
    input.checked = value === current;
    input.addEventListener('change', () => {
      local.set('highlight', value);
      applyAppearance();
      renderAppearanceSwatchColors();
    });
    return el('label', { class: 'swatch' }, input, el('span', null, el('i', { class: `swatch-dot hl-${value}` }), label));
  }));
  renderAppearanceSwatchColors();
}

/** Each swatch previews its own color by temporarily reading the theme's token values. */
function renderAppearanceSwatchColors() {
  const root = document.documentElement;
  const saved = root.dataset.highlight;
  for (const dot of $$('.swatch-dot')) {
    const value = dot.className.match(/hl-(\w+)/)?.[1] || 'sun';
    if (value === 'sun') delete root.dataset.highlight;
    else root.dataset.highlight = value;
    const styles = getComputedStyle(root);
    dot.style.backgroundColor = styles.getPropertyValue('--hl-bg').trim();
    dot.style.borderColor = styles.getPropertyValue('--hl-line').trim();
  }
  if (saved) root.dataset.highlight = saved;
  else delete root.dataset.highlight;
}

function renderPrivacy() {
  const s = state.settings;
  $('#set-online-tts').checked = !!s.online_tts_allowed;
  $('#set-ai').checked = !!s.ai_enabled;
  $('#ai-settings').hidden = !s.ai_enabled;
  const keySet = !!s.anthropic_api_key_set;
  const keyState = $('#api-key-state');
  keyState.replaceChildren(keySet ? el('span', { class: 'badge badge-success' }, icon('check'), 'Saved') : el('span', { class: 'badge', text: 'Not set' }));
  $('#api-key-remove').disabled = !keySet;
  $('#api-key').placeholder = keySet ? 'A key is saved. Paste a new one to replace it.' : 'sk-ant-…';
  $('#ai-model').value = s.ai_model || '';
  $('#data-dir').textContent = state.status?.data_dir || 'Unknown';
}

function setupSettings() {
  settingsUI.selectTab = setupTabs($('#settings-tabs'), (tab) => {
    if (tab.id === 'stab-voice') loadCatalog();
  });
  $('#set-voice').addEventListener('change', async (event) => {
    const engine = event.target.dataset.engine;
    if (await saveSettings({ voices: { [engine]: event.target.value } })) {
      renderVoicePicker();
      restartPlayback();
    }
  });
  $('#btn-test-voice').addEventListener('click', (event) => {
    speaker.toggle('Hello! This is how I will sound when I read your books to you.', event.currentTarget);
  });
  for (const radio of $$('input[name="theme"]')) {
    radio.addEventListener('change', async () => {
      applyAppearance({ theme: radio.value });
      await saveSettings({ theme: radio.value });
      applyAppearance();
      renderAppearanceSwatchColors();
    });
  }
  const saveScale = debounce((value) => saveSettings({ font_scale: value }), 500);
  $('#font-scale').addEventListener('input', (event) => {
    const value = Math.round(Number(event.target.value) * 100) / 100;
    $('#font-scale-value').textContent = `${Math.round(value * 100)}%`;
    if (state.settings) state.settings.font_scale = value;
    applyAppearance({ fontScale: value });
    saveScale(value);
  });
  $('#comfortable').addEventListener('change', (event) => {
    local.set('comfortable', event.target.checked);
    applyAppearance();
  });
  $('#set-online-tts').addEventListener('change', async (event) => {
    if (await saveSettings({ online_tts_allowed: event.target.checked })) {
      await refreshStatus();
      renderEngineList();
      renderSettingsVoice();
      if (player.playing) restartPlayback();
    }
    renderPrivacy();
  });
  $('#set-ai').addEventListener('change', async (event) => {
    await saveSettings({ ai_enabled: event.target.checked });
    renderPrivacy();
    await refreshStatus();
    if (state.section) renderSection();
    if (state.section) setCurrent(state.index, { scroll: false });
  });
  $('#api-key-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const input = $('#api-key');
    const key = input.value.trim();
    if (!key) {
      toast('Paste your API key into the box first.');
      input.focus();
      return;
    }
    if (await saveSettings({ anthropic_api_key: key })) {
      input.value = '';
      toast('API key saved on this computer.', { kind: 'success' });
      await refreshStatus();
    }
    renderPrivacy();
  });
  $('#api-key-remove').addEventListener('click', async () => {
    if (await saveSettings({ anthropic_api_key: '' })) {
      toast('API key removed.');
      await refreshStatus();
    }
    renderPrivacy();
  });
  $('#ai-model').addEventListener('change', async (event) => {
    const model = event.target.value.trim();
    if (!model) {
      event.target.value = state.settings.ai_model || '';
      return;
    }
    if (await saveSettings({ ai_model: model })) toast('Model saved.', { kind: 'success' });
    renderPrivacy();
  });
  $('#copy-data-dir').addEventListener('click', async () => {
    const path = state.status?.data_dir || '';
    try {
      await navigator.clipboard.writeText(path);
      toast('Folder path copied.', { kind: 'success' });
    } catch {
      const range = document.createRange();
      range.selectNodeContents($('#data-dir'));
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      toast('Press Ctrl+C (or ⌘C) to copy the selected path.');
    }
  });
  $('#btn-wipe').addEventListener('click', () => {
    $('#wipe-input').value = '';
    $('#wipe-confirm').disabled = true;
    openDialog($('#wipe-dialog'), { focus: $('#wipe-input') });
  });
  $('#wipe-input').addEventListener('input', (event) => {
    $('#wipe-confirm').disabled = event.target.value.trim() !== 'DELETE';
  });
  $('#wipe-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    if ($('#wipe-input').value.trim() !== 'DELETE') return;
    $('#wipe-confirm').disabled = true;
    try {
      await api('/api/wipe', { method: 'POST', body: { confirm: 'DELETE' } });
    } catch (err) {
      toast(errorText(err), { kind: 'error' });
      $('#wipe-confirm').disabled = false;
      return;
    }
    hardStop();
    clips.clear();
    local.clearAll();
    $('#wipe-dialog').close('done');
    $('#settings-dialog').close('done');
    toast('All your Athena data was erased.', { kind: 'success' });
    setTimeout(() => window.location.reload(), 1200);
  });
}

// ----- the player's voice picker

function renderVoicePicker() {
  const select = $('#voice-select');
  select.replaceChildren();
  if (!state.status) {
    select.append(el('option', { value: '', text: 'Voices unavailable' }));
    select.disabled = true;
    return;
  }
  select.disabled = false;
  for (const e of state.status.engines || []) {
    if (!engineAllowed(e)) continue;
    const group = el('optgroup', { label: `${e.label}${e.online ? ' ☁' : ''}` });
    if (e.name === 'browser') {
      if (!hasSpeechSynthesis()) continue;
      const voices = usableBrowserVoices();
      if (!voices.length) group.append(el('option', { value: 'browser::', text: 'Browser voice (none found)' }));
      for (const v of voices) group.append(el('option', { value: `browser::${v.voiceURI}`, text: `${v.name}${v.localService === false ? ' ☁' : ''}` }));
    } else {
      for (const v of state.serverVoices[e.name] || []) {
        group.append(el('option', { value: `${e.name}::${v.id}`, text: `${v.name}${v.online || e.online ? ' ☁' : ''}` }));
      }
    }
    if (group.children.length) select.append(group);
  }
  select.append(el('option', { value: '__more__', text: 'More voices…' }));
  const engine = activeEngine();
  const wanted = `${engine}::${activeVoice(engine)}`;
  select.value = wanted;
  if (select.value !== wanted) {
    select.prepend(el('option', { value: wanted, text: engineLabel(engine) }));
    select.value = wanted;
  }
  select.dataset.current = select.value;
  select.title = `Voice: ${select.selectedOptions[0]?.textContent || ''}`;
}

async function onVoicePicked(event) {
  const select = event.target;
  const value = select.value;
  if (value === '__more__') {
    select.value = select.dataset.current || '';
    openSettings('voice');
    return;
  }
  const split = value.indexOf('::');
  const engine = value.slice(0, split);
  const voice = value.slice(split + 2);
  const changes = { tts_engine: engine };
  if (voice) changes.voices = { [engine]: voice };
  if (await saveSettings(changes)) {
    renderVoicePicker();
    announce(`Voice: ${select.selectedOptions[0]?.textContent || engine}`);
    restartPlayback();
  } else {
    select.value = select.dataset.current || '';
  }
}

// =====================================================================
// 13. Export ("Save as audio")
// =====================================================================

const exporter = { jobId: null, polling: false, files: null };

function exportEngines() {
  return (state.status?.engines || []).filter((e) => e.name !== 'browser' && engineAllowed(e));
}

function openExport() {
  if (!state.book) return;
  const dialog = $('#export-dialog');
  if (!exporter.polling && !exporter.files) resetExportForm();
  openDialog(dialog, { focus: exporter.polling || exporter.files ? $('#export-cancel') : $('#export-all') });
}

function resetExportForm() {
  exporter.jobId = null;
  exporter.files = null;
  $('#export-setup').hidden = false;
  $('#export-progress').hidden = true;
  $('#export-start').hidden = false;
  $('#export-start').disabled = false;
  $('#export-cancel').textContent = 'Cancel';
  $('#export-all').checked = true;
  const box = $('#export-sections');
  box.replaceChildren(...state.toc.map((entry) => {
    const input = el('input', { type: 'checkbox', value: entry.id });
    input.checked = true;
    input.addEventListener('change', syncExportAll);
    return el('label', { class: `check level-${Math.min(3, entry.level || 1)}` }, input, el('span', { text: entry.title }));
  }));
  const select = $('#export-voice');
  select.replaceChildren();
  for (const e of exportEngines()) {
    const voices = state.serverVoices[e.name] || [];
    const group = el('optgroup', { label: `${e.label}${e.online ? ' ☁' : ''}` });
    for (const v of voices) group.append(el('option', { value: `${e.name}::${v.id}`, text: `${v.name}${v.online || e.online ? ' ☁' : ''}` }));
    if (!voices.length) group.append(el('option', { value: `${e.name}::`, text: `${e.label} (default voice)` }));
    select.append(group);
  }
  const engine = activeEngine();
  const preferred = engine !== 'browser' ? `${engine}::${activeVoice(engine)}` : '';
  if (preferred) select.value = preferred;
  if (!select.value && select.options.length) select.selectedIndex = 0;
  const none = !select.options.length;
  select.disabled = none;
  $('#export-start').disabled = none;
  $('#export-note').textContent = none
    ? 'Saving audio needs a natural or system voice. The browser voice can only read aloud inside this page. Get a natural voice in Settings → Voice.'
    : 'Larger books can take a while. You can keep reading while it works.';
  $('#export-rate').value = String(state.speed);
  $('#export-rate-value').textContent = formatSpeed(state.speed);
}

function syncExportAll() {
  const boxes = $$('#export-sections input');
  $('#export-all').checked = boxes.every((b) => b.checked);
  $('#export-start').disabled = !boxes.some((b) => b.checked) || $('#export-voice').disabled;
}

function exportDownloadUrl(url) {
  return `${url}${url.includes('?') ? '&' : '?'}token=${enc(TOKEN)}`;
}

function renderExportFiles(files) {
  const host = $('#export-files');
  host.replaceChildren(
    el('p', { class: 'hint', text: 'Your audio is ready. Click a file to save it.' }),
    el('ul', { class: 'file-list' }, files.map((file) => el('li', null,
      el('a', { class: 'file-link', href: exportDownloadUrl(file.url || ''), download: file.name },
        icon(/\.m3u8?$/i.test(file.name) ? 'file' : 'music'),
        el('span', { class: 'file-name', text: file.name }),
        el('span', { class: 'file-size', text: formatSize(file.size) }))))));
}

async function startExport(event) {
  event.preventDefault();
  if (!state.book || exporter.polling) return;
  const selected = $$('#export-sections input').filter((b) => b.checked).map((b) => b.value);
  if (!selected.length) {
    toast('Choose at least one section.');
    return;
  }
  const all = selected.length === state.toc.length;
  const [engine, voice] = ($('#export-voice').value || '').split('::');
  const format = $('input[name="export-format"]:checked')?.value || 'mp3';
  const rate = Math.round(clamp(Number($('#export-rate').value) || 1, 0.5, 3) * 20) / 20;
  const body = { book_id: state.book.id, section_ids: all ? null : selected, format, rate };
  if (engine) body.engine = engine;
  if (voice) body.voice = voice;
  $('#export-start').disabled = true;
  let jobId;
  try {
    jobId = (await api('/api/export', { method: 'POST', body })).job_id;
  } catch (err) {
    toast(errorText(err), { kind: 'error' });
    $('#export-start').disabled = false;
    return;
  }
  exporter.jobId = jobId;
  exporter.polling = true;
  $('#export-setup').hidden = true;
  $('#export-progress').hidden = false;
  $('#export-start').hidden = true;
  $('#export-files').replaceChildren();
  $('#export-message').textContent = 'Starting…';
  $('#export-bar').style.width = '0%';
  announce('Saving audio started');
  try {
    const job = await pollJob(jobId, (j) => {
      $('#export-message').textContent = j.message || 'Working…';
      $('#export-bar').style.width = `${Math.round((j.progress || 0) * 100)}%`;
      $('#export-bar-wrap').setAttribute('aria-valuenow', String(Math.round((j.progress || 0) * 100)));
    }, { interval: 800 });
    exporter.polling = false;
    $('#export-cancel').textContent = 'Close';
    if (job.status === 'done') {
      exporter.files = job.result?.files || [];
      $('#export-message').textContent = `Done: ${plural(exporter.files.filter((f) => !/\.m3u8?$/i.test(f.name)).length, 'file')} saved.`;
      $('#export-bar').style.width = '100%';
      renderExportFiles(exporter.files);
      announce('Your audio files are ready');
      if (!$('#export-dialog').open) {
        toast('Your audio files are ready.', { kind: 'success', action: { label: 'Show files', run: openExport } });
      }
    } else if (job.status === 'cancelled') {
      $('#export-message').textContent = 'Cancelled.';
      exporter.files = [];
    } else {
      $('#export-message').textContent = job.error || 'Saving audio failed.';
      exporter.files = [];
      toast(job.error || 'Saving audio failed.', { kind: 'error' });
    }
  } catch (err) {
    exporter.polling = false;
    exporter.files = [];
    $('#export-cancel').textContent = 'Close';
    $('#export-message').textContent = errorText(err);
  }
}

async function cancelOrCloseExport() {
  const dialog = $('#export-dialog');
  if (exporter.polling && exporter.jobId) {
    try {
      await api(`/api/jobs/${enc(exporter.jobId)}/cancel`, { method: 'POST' });
      $('#export-message').textContent = 'Cancelling…';
    } catch (err) {
      toast(errorText(err), { kind: 'error' });
    }
    return;
  }
  dialog.close('dismiss');
  if (exporter.files) {
    exporter.files = null;
    exporter.jobId = null;
  }
}

function setupExport() {
  $('#export-form').addEventListener('submit', startExport);
  $('#export-cancel').addEventListener('click', cancelOrCloseExport);
  $('#export-all').addEventListener('change', (event) => {
    for (const box of $$('#export-sections input')) box.checked = event.target.checked;
    syncExportAll();
  });
  $('#export-rate').addEventListener('input', (event) => {
    $('#export-rate-value').textContent = formatSpeed(Number(event.target.value));
  });
  $('#export-bar-wrap').setAttribute('role', 'progressbar');
  $('#export-bar-wrap').setAttribute('aria-label', 'Saving audio');
  $('#export-bar-wrap').setAttribute('aria-valuemin', '0');
  $('#export-bar-wrap').setAttribute('aria-valuemax', '100');
}

// =====================================================================
// 14. Keyboard shortcuts and start-up
// =====================================================================

function onKeyDown(event) {
  if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey) return;
  if ($$('dialog').some((d) => d.open)) return; // dialogs handle their own keys (Esc closes them)
  const target = event.target instanceof Element ? event.target : document.body;
  if (event.key === 'Escape') {
    if (!$('#sleep-menu').hidden) {
      toggleSleepMenu(false);
      $('#btn-sleep').focus();
      return;
    }
    if (!$('#sel-bubble').hidden) {
      $('#sel-bubble').hidden = true;
      return;
    }
    if (document.body.classList.contains('sidebar-open') || (mqStudyOverlay.matches && document.body.classList.contains('study-open'))) {
      const wasSidebar = document.body.classList.contains('sidebar-open');
      closeOverlays();
      (wasSidebar ? $('#btn-sidebar') : $('#btn-study')).focus();
    }
    return;
  }
  if (target.closest('input, textarea, select, [contenteditable=""], [contenteditable="true"]')) return;
  if (target.closest('[role="tablist"]') && event.key.startsWith('Arrow')) return;
  switch (event.key) {
    case ' ':
    case 'Spacebar':
      if (target.closest('button, a, summary') && !target.closest('[data-space-plays]')) return;
      event.preventDefault();
      togglePlay();
      break;
    case 'ArrowLeft':
      event.preventDefault();
      if (event.shiftKey) prevSection();
      else prevSentence();
      break;
    case 'ArrowRight':
      event.preventDefault();
      if (event.shiftKey) nextSection();
      else nextSentence();
      break;
    case '+':
    case '=':
      event.preventDefault();
      changeSpeed(0.1);
      break;
    case '-':
    case '_':
      event.preventDefault();
      changeSpeed(-0.1);
      break;
    case 'b':
    case 'B':
      event.preventDefault();
      toggleBookmark();
      break;
    case '?':
      event.preventDefault();
      openDialog($('#shortcuts-dialog'));
      break;
    default:
      break;
  }
}

function bindUI() {
  // Sidebar, drawer, header
  $('#btn-sidebar').addEventListener('click', () => setSidebarOpen(!document.body.classList.contains('sidebar-open')));
  $('#btn-sidebar-close').addEventListener('click', () => {
    setSidebarOpen(false);
    $('#btn-sidebar').focus();
  });
  $('#btn-study').addEventListener('click', () => setStudyOpen(!document.body.classList.contains('study-open'), { focus: mqStudyOverlay.matches }));
  $('#btn-study-close').addEventListener('click', () => setStudyOpen(false));
  $('#scrim').addEventListener('click', closeOverlays);
  $('#btn-settings').addEventListener('click', () => openSettings('voice'));
  $('#btn-shortcuts').addEventListener('click', () => openDialog($('#shortcuts-dialog')));
  $('#privacy-chip').addEventListener('click', () => openSettings('privacy'));
  const onBreakpoint = () => {
    if (!mqNarrow.matches) document.body.classList.remove('sidebar-open');
    syncPanels();
  };
  mqNarrow.addEventListener('change', onBreakpoint);
  mqStudyOverlay.addEventListener('change', onBreakpoint);
  selectSidebarTabById = setupTabs($('#sidebar-tabs'), (tab) => {
    if (tab.id === 'tab-contents') renderTocCurrent();
  });

  // Library
  $('#btn-add').addEventListener('click', chooseFiles);
  $('#btn-paste').addEventListener('click', openPaste);
  $('#file-input').addEventListener('change', (event) => {
    const files = Array.from(event.target.files || []);
    event.target.value = '';
    if (files.length) importFiles(files);
  });
  $('#paste-form').addEventListener('submit', submitPaste);
  $('#paste-text').addEventListener('input', updatePasteCount);
  setupDragAndDrop();

  // Player
  $('#btn-play').addEventListener('click', togglePlay);
  $('#btn-prev').addEventListener('click', prevSentence);
  $('#btn-next').addEventListener('click', nextSentence);
  $('#btn-prev-section').addEventListener('click', prevSection);
  $('#btn-next-section').addEventListener('click', nextSection);
  $('#btn-bookmark').addEventListener('click', toggleBookmark);
  $('#speed').addEventListener('input', (event) => setSpeed(event.target.value));
  $('#voice-select').addEventListener('change', onVoicePicked);
  $('#btn-sleep').addEventListener('click', () => toggleSleepMenu());
  for (const option of $$('[data-sleep]')) {
    option.addEventListener('click', () => {
      setSleepMode(option.dataset.sleep);
      toggleSleepMenu(false);
      $('#btn-sleep').focus();
    });
  }
  document.addEventListener('click', (event) => {
    if (!$('#sleep-menu').hidden && !event.target.closest('.sleep')) toggleSleepMenu(false);
  });

  // Reader
  const box = $('#reader');
  box.addEventListener('click', onReaderClick);
  box.addEventListener('wheel', markUserScroll, { passive: true });
  box.addEventListener('touchmove', markUserScroll, { passive: true });
  box.addEventListener('keydown', (event) => {
    if (['PageUp', 'PageDown', 'Home', 'End', 'ArrowUp', 'ArrowDown'].includes(event.key)) markUserScroll();
  });
  box.addEventListener('scroll', () => {
    if (performance.now() > reader.programmaticUntil) markUserScroll();
    if (!$('#sel-bubble').hidden) $('#sel-bubble').hidden = true;
  }, { passive: true });
  document.addEventListener('selectionchange', debounce(updateSelectionBubble, 150));
  const bubble = $('#sel-bubble');
  bubble.addEventListener('mousedown', (event) => event.preventDefault());
  $('#sel-explain').addEventListener('click', () => {
    const selection = readerSelection();
    bubble.hidden = true;
    if (selection) explainText(selection.text);
  });
  $('#sel-read').addEventListener('click', () => {
    const selection = readerSelection();
    bubble.hidden = true;
    if (!selection) return;
    const index = utteranceAtRange(selection.range);
    window.getSelection()?.removeAllRanges();
    if (index !== null && Number.isFinite(index)) seekTo(index, { play: true });
  });

  setupStudy();
  setupDialogs();
  setupSettings();
  setupExport();
  setupMediaSession();
  document.addEventListener('keydown', onKeyDown);

  // One AudioContext, created on the first user gesture.
  const unlock = () => webAudio.ensure();
  window.addEventListener('pointerdown', unlock, { once: true, capture: true });
  window.addEventListener('keydown', unlock, { once: true, capture: true });

  if (hasSpeechSynthesis()) {
    window.speechSynthesis.addEventListener('voiceschanged', () => {
      renderVoicePicker();
      if ($('#settings-dialog').open && $('#set-voice').dataset.engine === 'browser') renderSettingsVoice();
    });
  }
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') saveProgress({ beacon: true });
  });
  window.addEventListener('pagehide', () => saveProgress({ beacon: true }));
}

async function init() {
  hydrateIcons();
  applyAppearance();
  bindUI();
  syncPanels();
  renderSleep();
  updatePlayUI();
  updateStatus();
  renderStudy();
  try {
    const [status, settings, library] = await Promise.all([
      api('/api/status'), api('/api/settings'), api('/api/library'),
    ]);
    state.status = status;
    state.settings = settings;
    state.books = library.books || [];
  } catch (err) {
    if (err.status !== 401) {
      $('#reader-body').replaceChildren(el('div', { class: 'welcome' },
        el('h1', { text: "Can't connect to Athena" }),
        el('p', { class: 'welcome-lead', text: `${errorText(err)} If you closed the Athena window, start it again and then reload this page.` }),
        el('div', { class: 'welcome-actions' },
          el('button', { type: 'button', class: 'btn btn-primary', onclick: () => window.location.reload() }, 'Try again'))));
    }
    return;
  }
  setSpeed(state.settings.speed || 1, { save: false, apply: false });
  applyAppearance();
  renderPrivacyChip();
  await refreshServerVoices();
  renderVoicePicker();
  renderLibrary();
  renderStudy();
  checkVoiceBanner();
  const last = local.get('lastBook', null);
  const target = state.books.find((b) => b.id === last) || state.books.find((b) => b.progress);
  if (target) await openBook(target.id);
  else renderReader();
}

init();
