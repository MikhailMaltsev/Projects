// Crypt Survival — состояние забега, герои, сохранение
'use strict';

/* ================= state ================= */
// state: 'title' | 'play' | 'levelup' | 'pause' | 'dead' | 'win'
let state = 'title', tt = 0, stateT = 0;
let P = null, G = null;
let enemies = [], gems = [], items = [], props = [], swings = [], projs = [], parts = [], floats = [];
let ox = 0, oy = 0, shake = 0, hurtFlash = 0, freeze = 0;
// the level-up cards on screen and the one under the cursor
let choices = [], sel = 0;

/* ================= heroes =================
   Stats as in pixel-crypt (HEROES there); weapon = the starting weapon (weapons.js). The prototype has the three heroes
   whose base attack is already a weapon here; the rest come with their own weapons and E (GDD, M2) */
const HEROES = {
  knight: { name: 'РЫЦАРЬ', hp: 3, spd: 78, weapon: 'wave' },
  archer: { name: 'ЛУЧНИК', hp: 3, spd: 82, weapon: 'arrows' },
  necro: { name: 'НЕКРОМАНТ', hp: 2, spd: 78, weapon: 'souls' }
};
const HERO_IDS = Object.keys(HEROES);
let hero = 'knight';

// pickup radius in px and how far the magnet item pulls from
const MAGNET_R = 32;
function makePlayer(id) {
  const h = HEROES[id];
  return {
    hero: id, x: CHPX / 2, y: CHPX / 2, w: 10, h: 10,
    hp: h.hp * HPH, maxhp: h.hp * HPH, spd: h.spd, inv: 0, kx: 0, ky: 0,
    fx: 1, fy: 0, face: 1, anim: 0, moving: false, swingT: 0,
    // multipliers the passives change (weapons.js, PASSIVES)
    might: 1, area: 1, cdMul: 1, spdMul: 1, magnet: MAGNET_R,
    weapons: [], passives: []
  };
}
function makeRun(seed) {
  return { time: 0, kills: 0, gold: 0, xp: 0, lvl: 1, pending: 0, seed, spawnT: 0, events: new Set(), bestNew: false };
}

/* ================= saves =================
   Best survival time per hero and the gold wallet between runs; localStorage only for now (GDD: meta comes in M3) */
const SAVE_KEY = 'cryptsurvival_save';
let wallet = 0, best = {};
function loadSave() {
  try {
    const s = JSON.parse(localStorage.getItem(SAVE_KEY) || '{}');
    wallet = Math.max(0, Math.floor(+s.wallet || 0));
    best = s.best && typeof s.best === 'object' ? s.best : {};
    if (HEROES[s.hero]) hero = s.hero;
  } catch (e) { /* storage unavailable */ }
}
function saveGame() {
  try { localStorage.setItem(SAVE_KEY, JSON.stringify({ wallet, best, hero })); } catch (e) { /* ignore */ }
}
// the run is over (death or dawn): its gold goes to the wallet, the time may be a new best
function finishRun() {
  wallet += G.gold;
  if (G.time > (best[P.hero] || 0)) { best[P.hero] = Math.floor(G.time); G.bestNew = true; }
  saveGame();
}
