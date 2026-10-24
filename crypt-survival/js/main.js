// Crypt Survival — забег, обновление кадра, масштаб и игровой цикл (подключать последним)
'use strict';

/* ================= the run ================= */
// the gold for seeing the dawn (the whole run survived)
const DAWN_GOLD = 50;
function startRun() {
  G = makeRun((Math.random() * 1e9) | 0); P = makePlayer(hero);
  enemies = []; gems = []; items = []; swings = []; projs = []; parts = []; floats = []; choices = [];
  resetWorld(); updChunks(); addWeapon(HEROES[hero].weapon);
  saveGame();
  state = 'play'; stateT = 0; Snd.sfx('open'); Music.start('dun');
}
function toTitle() { state = 'title'; stateT = 0; Music.start('title'); }
function update(dt) {
  stateT += dt;
  if (state !== 'play') { if (state === 'dead' || state === 'win') updFx(dt); return; }
  readInput();
  G.time += dt;
  updPlayer(dt); updChunks(); updSpawn(dt); updEnemies(dt); updWeapons(dt); updGems(dt); updItems(dt); updFx(dt);
  if (state !== 'play') return; // died this frame
  if (G.time >= RUN_T) {
    G.gold += DAWN_GOLD; state = 'win'; stateT = 0; Music.stop(); Snd.sfx('win'); finishRun();
  } else if (G.pending > 0) { state = 'levelup'; stateT = 0; sel = 0; choices = rollChoices(); Snd.sfx('chest'); }
}

/* ================= sizing & loop ================= */
function resize() {
  const w = stage.clientWidth - 4, h = stage.clientHeight - 4; if (w <= 0 || h <= 0) return;
  let s = Math.min(w / W, h / H); if (s >= 2) s = Math.floor(s);
  cv.style.width = Math.floor(W * s) + 'px'; cv.style.height = Math.floor(H * s) + 'px';
}
if (window.ResizeObserver) new ResizeObserver(resize).observe(stage);
addEventListener('resize', resize); resize();
try { if (document.fonts && document.fonts.load) document.fonts.load('8px "Press Start 2P"', 'АБВгде0123'); } catch (e) { /* ignore */ }

let last = performance.now(), lastErr = null;
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.05, Math.max(0, (now - last) / 1000)); last = now; tt += dt;
  // one broken frame must not stop the loop
  try { update(dt); } catch (e) { if (lastErr !== String(e)) { lastErr = String(e); console.error('кадр упал:', e); } }
  try { draw(); } catch (e) { if (lastErr !== String(e)) { lastErr = String(e); console.error('кадр упал:', e); } }
}
loadSave();
Snd.init();
for (const ev of ['pointerdown', 'keydown', 'touchend']) addEventListener(ev, () => { const was = !!Snd.ctx; Snd.init(); if (!was && Snd.ctx && state === 'title') Music.start('title'); }, { capture: true, passive: true });
if (Snd.ctx) Music.start('title');
requestAnimationFrame(frame);
