// Crypt Survival — враги, волны по минутам, толпа, урон и смерть
'use strict';

/* ================= enemies =================
   The old crypt's roster with pixel-crypt's sprites. hp/xp grow from EN of pixel-crypt (xp = ceil(score / 10));
   the speeds are lower than there: a crowd must stay slower than the hero. fly = goes over the pieces on the field */
const EN = {
  slime: { hp: 3, spd: 30, dmg: 1, xp: 1, w: 12, h: 10, col: '#4cc36a', img: () => S_SLIME, hop: true },
  bat: { hp: 2, spd: 50, dmg: 1, xp: 2, w: 10, h: 8, col: '#a77bd6', img: () => S_BAT, fly: true, dy: 4 },
  skel: { hp: 5, spd: 28, dmg: 1, xp: 3, w: 10, h: 12, col: '#e8e4d0', img: t => S_SKEL[Math.floor(t * 6) % 2] },
  ghost: { hp: 7, spd: 24, dmg: 1, xp: 3, w: 10, h: 12, col: '#bfe0ff', img: t => S_GHOST[Math.floor(t * 4) % 2], fly: true, alpha: 0.8 },
  spider: { hp: 4, spd: 42, dmg: 1, xp: 3, w: 12, h: 8, col: '#8a6aaa', img: t => S_SPIDER[Math.floor(t * 8) % 2], dy: -2 }
};
/* waves (GDD §3): who comes in each minute of the run; alive at least N(m) = min(15 + 8m, CROWD_MAX).
   Swarms of bats close in from all sides at SWARM_AT; a giant (ELITE) walks in at ELITE_AT */
const CROWD_MAX = 120;
const WAVES = [
  [['slime', 1]],
  [['slime', 2], ['bat', 1]],
  [['bat', 2], ['skel', 1]],
  [['skel', 2], ['slime', 1]],
  [['skel', 1], ['ghost', 1]],
  [['skel', 1], ['ghost', 1], ['spider', 1]],
  [['spider', 2], ['bat', 1]],
  [['ghost', 2], ['skel', 1]],
  [['skel', 2], ['spider', 1], ['slime', 1]],
  [['ghost', 1], ['spider', 1], ['bat', 1]],
  [['skel', 2], ['ghost', 2]],
  [['spider', 2], ['skel', 1]],
  [['ghost', 2], ['spider', 1], ['bat', 1]],
  [['skel', 2], ['ghost', 1], ['spider', 1]],
  [['skel', 1], ['ghost', 1], ['spider', 1], ['bat', 1]]
];
const SWARM_AT = [3, 8, 13], ELITE_AT = { 5: 'skel', 10: 'ghost' };
const crowdMin = m => Math.min(10 + 8 * m, CROWD_MAX);
// how far off the screen's middle they appear, and past what distance a straggler is brought round in front again
const SPAWN_R = Math.hypot(W / 2, H / 2) + 16, FAR_R = SPAWN_R * 1.3;
// health grows by HP_RAMP of the base every minute (GDD §3)
const HP_RAMP = 0.2;
const hpAt = (base, m) => Math.max(1, Math.round(base * (1 + HP_RAMP * m)));
function mkEnemy(type, x, y, elite = false) {
  const b = EN[type], m = Math.floor(G.time / 60), hp = hpAt(b.hp, m) * (elite ? 20 : 1);
  return { type, x, y, w: b.w * (elite ? 2 : 1), h: b.h * (elite ? 2 : 1), hp, maxhp: hp, spd: b.spd * (elite ? 0.8 : 1), dmg: b.dmg * (elite ? 2 : 1),
    xp: b.xp, elite, t: Math.random() * 3, seed: Math.random() * 6, kx: 0, ky: 0, flash: 0, face: 1, rise: 0 };
}
function spawnAt(type, a, elite = false) {
  if (enemies.length >= CROWD_MAX && !elite) return null;
  const e = mkEnemy(type, P.x + Math.cos(a) * SPAWN_R, P.y + Math.sin(a) * SPAWN_R, elite);
  enemies.push(e); return e;
}
// the spawner, as in VS: every SPAWN_GAP s a trickle of 0.5 + m/4 newcomers whatever the crowd (the fractions carry over),
// and the crowd topped up to its minimum; then the minute's events, once each
const SPAWN_GAP = 1;
const trickle = m => 0.5 + m / 4;
function updSpawn(dt) {
  const m = Math.min(WAVES.length - 1, Math.floor(G.time / 60));
  if ((G.spawnT -= dt) <= 0) {
    G.spawnT = SPAWN_GAP;
    G.trickle = (G.trickle || 0) + trickle(m);
    const tr = Math.floor(G.trickle); G.trickle -= tr;
    const n = Math.max(tr, Math.min(8, crowdMin(m) - enemies.length));
    for (let k = 0; k < n; k++) spawnAt(pickW(WAVES[m]), Math.random() * 6.283);
  }
  const mm = Math.floor(G.time / 60);
  if (SWARM_AT.includes(mm) && !G.events.has('swarm' + mm)) {
    G.events.add('swarm' + mm);
    for (let k = 0; k < 24; k++) { const e = spawnAt('bat', k / 24 * 6.283); if (e) e.swarm = true; }
    banner('СТАЯ!', '#a77bd6');
  }
  if (ELITE_AT[mm] && !G.events.has('elite' + mm)) {
    G.events.add('elite' + mm);
    spawnAt(ELITE_AT[mm], Math.random() * 6.283, true);
    banner('ГИГАНТ!', '#ff5a4a'); Snd.sfx('roar');
  }
}

/* ---------- the crowd: a grid of CELL px for the neighbours ---------- */
const CELL = 32, grid = new Map();
const gkey = (x, y) => (Math.floor(x / CELL) & 0xffff) << 16 | (Math.floor(y / CELL) & 0xffff);
function buildGrid() {
  grid.clear();
  for (const e of enemies) { const k = gkey(e.x, e.y); let c = grid.get(k); if (!c) grid.set(k, c = []); c.push(e); }
}
// every enemy within r of (x, y) (roughly: by cells, the caller checks the exact shape)
function nearEnemies(x, y, r, fn) {
  const x0 = Math.floor((x - r) / CELL), x1 = Math.floor((x + r) / CELL), y0 = Math.floor((y - r) / CELL), y1 = Math.floor((y + r) / CELL);
  for (let cy = y0; cy <= y1; cy++) for (let cx = x0; cx <= x1; cx++) {
    const c = grid.get((cx & 0xffff) << 16 | (cy & 0xffff)); if (!c) continue;
    for (const e of c) if (!e.dead && fn(e) === false) return;
  }
}
function nearestEnemy(x, y, maxR, skip) {
  let best = null, bd = maxR;
  for (const e of enemies) { if (e.dead || (skip && skip.has(e))) continue; const d = Math.hypot(e.x - x, e.y - y); if (d < bd) { bd = d; best = e; } }
  return best;
}

function updEnemies(dt) {
  enemies = enemies.filter(e => !e.dead);
  buildGrid();
  for (const e of enemies) {
    e.t += dt; e.flash = Math.max(0, e.flash - dt);
    if (e.rise > 0) { e.rise -= dt; continue; } // getting up out of a coffin
    const b = EN[e.type], dx = P.x - e.x, dy = P.y - e.y, d = Math.hypot(dx, dy) || 1;
    // a straggler left far behind is brought round in front of the hero (where he is heading)
    if (d > FAR_R && !e.elite) {
      const a = P.moving ? Math.atan2(P.fy, P.fx) + rr(-0.9, 0.9) : Math.random() * 6.283;
      e.x = P.x + Math.cos(a) * SPAWN_R; e.y = P.y + Math.sin(a) * SPAWN_R; continue;
    }
    let sp = e.spd * (b.hop ? 0.35 + Math.abs(Math.sin(e.t * 4 + e.seed)) * 1.1 : 1);
    let vx = dx / d * sp, vy = dy / d * sp;
    if (b.fly && e.type === 'bat') { vx += Math.cos(e.t * 5 + e.seed) * 20; vy += Math.sin(e.t * 6 + e.seed) * 20; }
    // keep apart from the neighbours: a crowd, not one blob
    let px = 0, py = 0;
    nearEnemies(e.x, e.y, 14, o => {
      if (o === e) return;
      const ax = e.x - o.x, ay = e.y - o.y, ad = Math.hypot(ax, ay), want = (e.w + o.w) * 0.42;
      if (ad > 0 && ad < want) { px += ax / ad * (want - ad); py += ay / ad * (want - ad); }
    });
    vx += px * 8; vy += py * 8;
    if (Math.abs(dx) > 1) e.face = dx < 0 ? -1 : 1;
    const mx = (vx + e.kx) * dt, my = (vy + e.ky) * dt;
    if (b.fly) { e.x += mx; e.y += my; } else moveEnt(e, mx, my);
    e.kx *= Math.pow(0.0005, dt); e.ky *= Math.pow(0.0005, dt);
    // touching the hero
    if (Math.abs(dx) < e.w / 2 + P.w / 2 - 1 && Math.abs(dy) < e.h / 2 + P.h / 2 - 1) hurtPlayer(e.dmg, e.x, e.y);
  }
}

/* ---------- damage ---------- */
// kb = how hard it is thrown back (giants barely move)
function damageEnemy(e, dmg, fx, fy, kb = 120) {
  if (e.dead || e.rise > 0) return;
  dmg *= P.might;
  e.hp -= dmg; e.flash = 0.1;
  const a = Math.atan2(e.y - fy, e.x - fx), k = e.elite ? kb * 0.15 : kb;
  e.kx = Math.cos(a) * k; e.ky = Math.sin(a) * k;
  burst(e.x, e.y, e.col, 3, 60);
  if (e.hp <= 0) killEnemy(e); else Snd.sfx('hit');
}
function killEnemy(e) {
  if (e.dead) return;
  e.dead = true; G.kills++;
  burst(e.x, e.y, e.col, e.elite ? 30 : 10, e.elite ? 110 : 80);
  Snd.sfx('kill'); addShake(e.elite ? 6 : 0.8);
  if (e.elite) {
    dropGem(e.x, e.y, 25); dropGem(e.x + 6, e.y + 4, 25); dropItem('heart', { x: e.x - 6, y: e.y + 2 });
  } else {
    dropGem(e.x, e.y, e.xp);
    if (Math.random() < 0.03) dropItem('coin', { x: e.x + 4, y: e.y + 2 });
  }
}

/* ---------- drawing ---------- */
const flipCache = new Map();
function flipOf(img) {
  let f = flipCache.get(img);
  if (!f) { f = document.createElement('canvas'); f.width = img.width; f.height = img.height; const g = f.getContext('2d'); g.translate(img.width, 0); g.scale(-1, 1); g.drawImage(img, 0, 0); flipCache.set(img, f); }
  return f;
}
function drawEnemy(e) {
  const b = EN[e.type], sc = e.elite ? 2 : 1;
  let img = b.img(e.t); if (e.face < 0) img = flipOf(img);
  const w = img.width * sc, h = img.height * sc, x = Math.round(e.x - ox - w / 2);
  let y = Math.round(e.y - oy - h / 2 - 2 + (b.dy || 0) * sc);
  if (b.hop) y -= Math.round(Math.abs(Math.sin(e.t * 4 + e.seed)) * 2);
  if (e.rise > 0) { ctx.globalAlpha = 1 - e.rise / 0.6; y += Math.round(e.rise * 10); }
  else if (b.alpha) ctx.globalAlpha = b.alpha + Math.sin(e.t * 3) * 0.1;
  if (e.elite) { // a red rim round the giant
    const rim = tintOf(img, '#ff3a3a');
    for (const [a, c] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) ctx.drawImage(rim, x + a, y + c, w, h);
  }
  ctx.drawImage(e.flash > 0 ? flashOf(img) : img, x, y, w, h);
  ctx.globalAlpha = 1;
}
function enemySortList(list) {
  for (const e of enemies) {
    if (!onScreen(e.x, e.y, 30)) continue;
    if (!EN[e.type].fly) shadowAt(e.x, e.y + e.h / 2 - 2, e.w);
    list.push({ y: e.y + e.h / 2 + (EN[e.type].fly ? 8 : 0), f: () => drawEnemy(e) });
  }
}
