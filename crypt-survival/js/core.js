// Crypt Survival — константы, канвас и утилиты
'use strict';

// the canvas is 320×192 like pixel-crypt's, scaled by a whole factor; T = tile, HPH = quarters in a heart
const W = 320, H = 192, T = 16, HPH = 4;
// the run: RUN_T seconds to survive; a chunk of the endless field is CH tiles a side
const RUN_T = 15 * 60, CH = 16, CHPX = CH * T;
const cv = document.getElementById('c');
const ctx = cv.getContext('2d');
ctx.imageSmoothingEnabled = false;
const stage = document.getElementById('stage');
const reduceMotion = !!(window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches);
const coarsePointer = window.matchMedia ? matchMedia('(pointer: coarse)') : { matches: false };

/* ================= utils ================= */
const rnd = n => Math.floor(Math.random() * n);
const rr = (a, b) => a + Math.random() * (b - a);
const clamp = (v, a, b) => v < a ? a : v > b ? b : v;
const shuffle = a => { for (let i = a.length - 1; i > 0; i--) { const j = rnd(i + 1); [a[i], a[j]] = [a[j], a[i]]; } return a; };
function mulberry(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const hash1 = n => { const v = Math.sin(n * 127.1 + 311.7) * 43758.5453; return v - Math.floor(v); };
// a whole-number seed of a chunk: the same field every time for the same run seed
const chunkSeed = (cx, cy, s) => (Math.imul(cx, 73856093) ^ Math.imul(cy, 19349663) ^ Math.imul(s, 83492791)) | 0;
// one of the items of a weighted list [[item, weight], ...]
function pickW(list, r = Math.random()) {
  let sum = 0; for (const [, w] of list) sum += w;
  let k = r * sum; for (const [v, w] of list) { if ((k -= w) <= 0) return v; }
  return list[list.length - 1][0];
}
const fmtTime = s => { s = Math.max(0, Math.floor(s)); return String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0'); };
