// Crypt Survival — оружие (атаки героев pixel-crypt), пассивки, снаряды
'use strict';

/* ================= weapons =================
   Every weapon fires by itself every cd s (times the hero's cdMul). lv = 1..WEAPON_MAX; the arrays are per level.
   desc[n] = what level n+1 adds (the card's line). GDD §5 */
const WEAPON_MAX = 8, PASSIVE_MAX = 5, SLOTS = 4;
const WEAPONS = {
  // the knight's blow: a crescent wave at the nearest enemy in reach, else where he looks (pixel-crypt player.js, startSwing / waveR)
  wave: {
    name: 'ВОЛНА КЛИНКА', desc: ['Волна в ближайшего врага', '+1 урон', '+15% площадь', 'Вторая волна назад', '+1 урон', '+15% площадь', '+1 урон', '-20% перезарядка'],
    dmg: [3, 4, 4, 4, 5, 5, 6, 6], area: [1, 1, 1.15, 1.15, 1.15, 1.3, 1.3, 1.3], back: [0, 0, 0, 1, 1, 1, 1, 1], cd: [1, 1, 1, 1, 1, 1, 1, 0.8],
    fire(lv) {
      const t = nearestEnemy(P.x, P.y, WAVE_R * 1.6 * this.area[lv - 1] * P.area), a = t ? Math.atan2(t.y - P.y, t.x - P.x) : Math.atan2(P.fy, P.fx);
      swingAt(a, this.dmg[lv - 1], this.area[lv - 1], 0);
      if (this.back[lv - 1]) swingAt(a + Math.PI, this.dmg[lv - 1], this.area[lv - 1], 0.12);
      Snd.sfx('swing');
    }
  },
  // the archer's fan of arrows at the nearest enemy (pixel-crypt shootFan)
  arrows: {
    name: 'ВЕЕР СТРЕЛ', desc: ['Стрелы в ближайшего врага', '+1 урон', '+1 стрела', '+1 пробой', '+1 стрела', '+1 урон', '+1 стрела', '+1 пробой'],
    dmg: [2, 3, 3, 3, 3, 4, 4, 4], n: [3, 3, 4, 4, 5, 5, 6, 6], pierce: [1, 1, 1, 2, 2, 2, 2, 3], cd: [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2],
    fire(lv) {
      const t = nearestEnemy(P.x, P.y, 170), base = t ? Math.atan2(t.y - P.y, t.x - P.x) : Math.atan2(P.fy, P.fx), n = this.n[lv - 1];
      for (let i = 0; i < n; i++) {
        const a = base + (i - (n - 1) / 2) * 0.16;
        projs.push({ kind: 'arrow', x: P.x + Math.cos(a) * 6, y: P.y - 2 + Math.sin(a) * 6, vx: Math.cos(a) * 230, vy: Math.sin(a) * 230, life: 0.75,
          dmg: this.dmg[lv - 1], pierce: this.pierce[lv - 1], hit: new Set() });
      }
      Snd.sfx('bow');
    }
  },
  // the necromancer's souls: each picks its own enemy and turns after it (pixel-crypt soulFan, made homing)
  souls: {
    name: 'ДУШИ', desc: ['Души сами ищут врага', '+1 душа', '+1 урон', '+1 душа', '+1 урон', '+1 душа', '+1 урон', '+1 душа'],
    dmg: [2, 2, 3, 3, 4, 4, 5, 5], n: [2, 3, 3, 4, 4, 5, 5, 6], cd: [1, 1, 1, 1, 1, 1, 1, 1],
    fire(lv) {
      const taken = new Set();
      for (let i = 0; i < this.n[lv - 1]; i++) {
        const t = nearestEnemy(P.x, P.y, 200, taken); if (t) taken.add(t);
        const a = t ? Math.atan2(t.y - P.y, t.x - P.x) : i / this.n[lv - 1] * 6.283;
        projs.push({ kind: 'soul', x: P.x, y: P.y - 4, vx: Math.cos(a) * 60, vy: Math.sin(a) * 60, life: 1.8, dmg: this.dmg[lv - 1], pierce: 1, hit: new Set(), target: t, seed: Math.random() * 9 });
      }
      Snd.sfx('souls');
    }
  }
};
const WEAPON_IDS = Object.keys(WEAPONS);

/* ================= passives =================
   Pixel-crypt's upgrades (state.js, UPGRADES) made into levels 1..PASSIVE_MAX. GDD §6 */
const PASSIVES = {
  heart: { name: 'КРЕПКОЕ СЕРДЦЕ', desc: '+1 сердце', fn: () => { P.maxhp += HPH; P.hp += HPH; } },
  might: { name: 'ОСТРЫЙ КЛИНОК', desc: '+10% урона', fn: () => { P.might += 0.1; } },
  boots: { name: 'ЛЁГКИЕ САПОГИ', desc: '+8% скорости', fn: () => { P.spdMul += 0.08; } },
  rate: { name: 'БЫСТРАЯ РУКА', desc: '-8% перезарядки', fn: () => { P.cdMul *= 0.92; } },
  area: { name: 'ДАЛЬНИЙ УДАР', desc: '+10% площади', fn: () => { P.area += 0.1; } },
  magnet: { name: 'АМУЛЕТ ДУШ', desc: '+25% радиус сбора', fn: () => { P.magnet += MAGNET_R * 0.25; } }
};
const PASSIVE_IDS = Object.keys(PASSIVES);

function addWeapon(id) { P.weapons.push({ id, lv: 1, cd: 0.3 }); }
function updWeapons(dt) {
  for (const w of P.weapons) {
    if ((w.cd -= dt) > 0) continue;
    const d = WEAPONS[w.id]; d.fire(w.lv); w.cd = d.cd[w.lv - 1] * P.cdMul;
  }
  updSwings(dt); updProjs(dt);
}

/* ---------- the blade wave ----------
   Like pixel-crypt's: fast at first, slowing down, from WAVE_FROM to the reach; hits each enemy and each piece once */
const SWING_T = 0.22, WAVE_FROM = 8, WAVE_R = 40, WAVE_SPAN = 1;
function swingAt(a, dmg, area, delay) { swings.push({ a, dmg, R: WAVE_R * area * P.area, t: -delay, hit: new Set() }); P.swingT = SWING_T; }
const waveRad = s => { const p = clamp(s.t / SWING_T, 0, 1); return WAVE_FROM + (s.R - WAVE_FROM) * (1 - (1 - p) * (1 - p)); };
function inArc(s, x, y, pad) {
  const vx = x - P.x, vy = y - P.y, d = Math.hypot(vx, vy);
  if (d < 12) return true; // right next to the hero the swing itself hits
  let da = Math.atan2(vy, vx) - s.a; while (da > Math.PI) da -= 6.2832; while (da < -Math.PI) da += 6.2832;
  return Math.abs(da) <= WAVE_SPAN + 0.15 && d - pad <= waveRad(s);
}
function updSwings(dt) {
  for (const s of swings) {
    s.t += dt; if (s.t < 0) continue;
    nearEnemies(P.x, P.y, s.R + 16, e => {
      if (!s.hit.has(e) && inArc(s, e.x, e.y, Math.max(e.w, e.h) / 2)) { s.hit.add(e); damageEnemy(e, s.dmg, P.x, P.y, 150); }
    });
    for (const p of [...props]) if (p.hp && inArc(s, p.x, p.y, p.w / 2)) hitProp(p, s.hit);
  }
  swings = swings.filter(s => s.t < SWING_T);
}

/* ---------- shots ---------- */
function updProjs(dt) {
  for (const r of projs) {
    if (r.dead) continue;
    if (r.kind === 'soul') { // turn after the target, take another when it is gone
      if (!r.target || r.target.dead) r.target = nearestEnemy(r.x, r.y, 160, r.hit);
      const sp = Math.min(150, Math.hypot(r.vx, r.vy) + 260 * dt);
      if (r.target) {
        const a = Math.atan2(r.target.y - r.y, r.target.x - r.x), cur = Math.atan2(r.vy, r.vx);
        let da = a - cur; while (da > Math.PI) da -= 6.2832; while (da < -Math.PI) da += 6.2832;
        const na = cur + clamp(da, -7 * dt, 7 * dt); r.vx = Math.cos(na) * sp; r.vy = Math.sin(na) * sp;
      } else { const k = sp / (Math.hypot(r.vx, r.vy) || 1); r.vx *= k; r.vy *= k; }
      if (Math.random() < 0.5) parts.push({ x: r.x, y: r.y, vx: rr(-6, 6), vy: rr(-10, 0), life: 0.25, max: 0.25, col: Math.random() < 0.5 ? '#9a6aff' : '#c49bff', sz: 1 });
    }
    r.x += r.vx * dt; r.y += r.vy * dt;
    if ((r.life -= dt) <= 0) { r.dead = true; if (r.kind === 'soul') burst(r.x, r.y, '#9a6aff', 4, 30); continue; }
    nearEnemies(r.x, r.y, 16, e => {
      if (r.hit.has(e) || Math.abs(e.x - r.x) > e.w / 2 + 2 || Math.abs(e.y - r.y) > e.h / 2 + 2) return;
      r.hit.add(e); damageEnemy(e, r.dmg, r.x - r.vx * 0.05, r.y - r.vy * 0.05, 90);
      if (--r.pierce <= 0) { r.dead = true; return false; }
    });
    if (r.dead || r.kind === 'soul') continue; // souls fly over the pieces
    for (const p of props) {
      if (Math.abs(p.x - r.x) > p.w / 2 || Math.abs(p.y - r.y) > p.h / 2 + 4) continue;
      if (p.hp) hitProp(p, r.hit);
      if (p.solid) { r.dead = true; burst(r.x, r.y, '#c8a070', 3, 30); }
      break;
    }
  }
  projs = projs.filter(r => !r.dead);
}

/* ---------- drawing ---------- */
function drawSwings() {
  for (const s of swings) {
    if (s.t < 0) continue;
    const p = s.t / SWING_T, r = waveRad(s), n = Math.max(12, Math.round(WAVE_SPAN * 2 * r)), fade = 1 - p * 0.6, alpha = [0.6, 0.38, 0.2];
    const wx = P.x - ox, wy = P.y - oy - 2;
    ctx.fillStyle = '#ffffff';
    for (let i = 0; i <= n; i++) {
      const a = s.a - WAVE_SPAN + 2 * WAVE_SPAN * i / n, edge = Math.min(i, n - i) / n, th = edge < 0.08 ? 1 : edge < 0.2 ? 2 : 3, c = Math.cos(a), sn = Math.sin(a);
      for (let k = 0; k < th; k++) { ctx.globalAlpha = alpha[k] * fade; ctx.fillRect(Math.round(wx + c * (r - k * 1.5)), Math.round(wy + sn * (r - k * 1.5)), 1, 1); }
    }
    ctx.globalAlpha = 1;
    // the sword sweeps close round the hero, two fading copies behind it
    for (let k = 2; k >= 0; k--) {
      const a = s.a + (Math.max(0, p - k * 0.12) * 2 - 1) * WAVE_SPAN * 0.9;
      ctx.save(); ctx.translate(Math.round(P.x - ox), Math.round(P.y - oy - 3)); ctx.rotate(a + Math.PI / 2);
      ctx.globalAlpha = k ? 0.3 / k : 1; ctx.drawImage(S_SWORD, -2, -3 - S_SWORD.height); ctx.restore();
    }
    ctx.globalAlpha = 1;
  }
}
function drawProjs() {
  for (const r of projs) {
    const x = r.x - ox, y = r.y - oy;
    if (r.kind === 'soul') { // pixel-crypt render.js, drawArrows: a pale core, a violet body and a wavy tail
      const a = Math.atan2(r.vy, r.vx), c = Math.cos(a), s = Math.sin(a);
      for (let i = 1; i <= 7; i++) {
        const w = Math.sin(tt * 24 + r.seed + i * 0.9) * i * 0.25;
        ctx.globalAlpha = (1 - i / 8) * 0.85; ctx.fillStyle = i < 3 ? '#c49bff' : '#9a6aff';
        ctx.fillRect(Math.round(x - c * i * 1.4 - s * w), Math.round(y - s * i * 1.4 + c * w), 1, 1);
      }
      ctx.globalAlpha = 1;
      const X = Math.round(x), Y = Math.round(y);
      ctx.fillStyle = 'rgba(180,140,255,.28)'; ctx.fillRect(X - 3, Y - 3, 7, 7);
      ctx.fillStyle = '#9a6aff'; ctx.fillRect(X - 2, Y - 1, 5, 3); ctx.fillRect(X - 1, Y - 2, 3, 5);
      ctx.fillStyle = '#e8dcff'; ctx.fillRect(X - 1, Y - 1, 2, 2);
      continue;
    }
    const a = Math.atan2(r.vy, r.vx), c = Math.cos(a), s = Math.sin(a);
    for (let i = -5; i <= 3; i++) { ctx.fillStyle = i >= 2 ? '#ffffff' : i <= -4 ? '#e0453a' : '#c8a070'; ctx.fillRect(Math.round(x + c * i), Math.round(y + s * i), 1, 1); }
    ctx.fillStyle = '#e0453a';
    ctx.fillRect(Math.round(x - c * 5 - s), Math.round(y - s * 5 + c), 1, 1); ctx.fillRect(Math.round(x - c * 5 + s), Math.round(y - s * 5 - c), 1, 1);
  }
}
// slot icons (HUD and the level-up cards): 12×12 px at (x, y)
function drawSlotIcon(kind, id, x, y) {
  ctx.save(); ctx.translate(x, y);
  if (kind === 'w') {
    if (id === 'wave') { ctx.translate(6, 6); ctx.rotate(Math.PI / 4); ctx.drawImage(S_SWORD, -2, -4); }
    else if (id === 'arrows') { for (const [dx, dy] of [[0, 0], [3, -3], [-3, 3]]) for (let i = -4; i <= 4; i++) { ctx.fillStyle = i >= 2 ? '#fff' : i <= -3 ? '#e0453a' : '#c8a070'; ctx.fillRect(6 + i + dx, 6 - i + dy, 1, 1); } }
    else if (id === 'souls') { ctx.fillStyle = '#9a6aff'; ctx.fillRect(3, 4, 7, 5); ctx.fillRect(4, 3, 5, 7); ctx.fillStyle = '#e8dcff'; ctx.fillRect(5, 5, 3, 3); ctx.fillStyle = '#c49bff'; ctx.fillRect(1, 7, 2, 1); ctx.fillRect(0, 9, 2, 1); }
  } else {
    const img = { heart: S_HEART_ITEM, might: S_SWORD, boots: TR_BOOT, rate: TR_CLAW, area: TR_ARC, magnet: null }[id];
    if (id === 'magnet') { ctx.fillStyle = '#e0453a'; ctx.fillRect(2, 2, 3, 7); ctx.fillRect(7, 2, 3, 7); ctx.fillRect(2, 8, 8, 2); ctx.fillStyle = '#c3cad8'; ctx.fillRect(2, 1, 3, 2); ctx.fillRect(7, 1, 3, 2); }
    else if (img) ctx.drawImage(img, Math.round(6 - img.width / 2), Math.round(6 - img.height / 2));
  }
  ctx.restore();
}
