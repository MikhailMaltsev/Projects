// Crypt Survival — герой: движение, урон, эффекты
'use strict';

/* ================= the hero ================= */
// after a blow the hero can't be hurt for INV_T s (shorter than pixel-crypt's 1.1: here the crowd is the danger)
const INV_T = 0.6;
function updPlayer(dt) {
  P.inv = Math.max(0, P.inv - dt); P.swingT = Math.max(0, P.swingT - dt);
  let mx = inp.mx, my = inp.my; const n = Math.hypot(mx, my);
  if (n > 1) { mx /= n; my /= n; }
  P.moving = n > 0.1;
  if (P.moving) { const un = Math.hypot(mx, my); P.fx = mx / un; P.fy = my / un; if (Math.abs(P.fx) > 0.2) P.face = P.fx < 0 ? -1 : 1; P.anim += dt * 9; }
  const sp = P.spd * P.spdMul;
  moveEnt(P, (mx * sp + P.kx) * dt, (my * sp + P.ky) * dt);
  P.kx *= Math.pow(0.0005, dt); P.ky *= Math.pow(0.0005, dt);
}
function hurtPlayer(n, fx, fy) {
  if (P.inv > 0 || P.hp <= 0 || state !== 'play') return;
  P.hp -= n; P.inv = INV_T; hurtFlash = 0.15; addShake(3);
  const a = Math.atan2(P.y - fy, P.x - fx); P.kx = Math.cos(a) * 90; P.ky = Math.sin(a) * 90;
  burst(P.x, P.y, '#e0453a', 8); Snd.sfx('hurt');
  if (P.hp <= 0) {
    P.hp = 0; state = 'dead'; stateT = 0; Music.stop(); Snd.sfx('die'); burst(P.x, P.y, '#c3cad8', 24, 90);
    finishRun();
  }
}
function drawPlayer() {
  const img = S_HERO[P.hero][P.moving ? Math.floor(P.anim) % 2 : 0];
  if (P.inv > 0 && Math.floor(tt * 20) % 2) return;
  const x = Math.round(P.x - ox), y = Math.round(P.y - oy + 6);
  ctx.save(); ctx.translate(x, y); ctx.scale(P.face, 1);
  ctx.drawImage(img, -8, -16);
  // what the hero holds (pixel-crypt render.js, drawHeroAt); the knight's sword is in the swing while he swings
  if (P.hero === 'archer') ctx.drawImage(S_BOW, 5, -13);
  else if (P.hero === 'necro') {
    ctx.drawImage(S_NSTAFF, 4, -14);
    const f = Math.floor(tt * 10) % 2;
    ctx.fillStyle = 'rgba(154,106,255,.35)'; ctx.fillRect(4, -18 - f, 5, 3 + f);
    ctx.fillStyle = '#c49bff'; ctx.fillRect(5, -17 - f, 3, 2 + f);
  } else if (P.swingT <= 0) { ctx.fillStyle = '#e8eef8'; ctx.fillRect(6, -13, 2, 7); ctx.fillStyle = '#7d869c'; ctx.fillRect(6, -13, 1, 7); ctx.fillStyle = '#e8b84a'; ctx.fillRect(5, -6, 4, 1); }
  ctx.restore();
}

/* ================= particles and floating text ================= */
function burst(x, y, col, n = 8, sp = 60) {
  if (parts.length > 600) n = Math.min(n, 2);
  for (let i = 0; i < n; i++) {
    const a = Math.random() * 6.283, s = rr(sp * 0.3, sp);
    parts.push({ x, y, vx: Math.cos(a) * s, vy: Math.sin(a) * s, life: rr(0.25, 0.6), max: 0.6, col, sz: Math.random() < 0.5 ? 1 : 2 });
  }
}
function floatText(x, y, text, col = '#fff') { floats.push({ x, y, text, col, t: 0.9 }); }
const addShake = v => { if (!reduceMotion) shake = Math.max(shake, v); };
let bannerT = 0, bannerText = '', bannerCol = '#fff';
function banner(text, col) { bannerText = text; bannerCol = col; bannerT = 2; }
function updFx(dt) {
  for (const p of parts) { p.x += p.vx * dt; p.y += p.vy * dt; p.vx *= 0.9; p.vy *= 0.9; p.life -= dt; }
  parts = parts.filter(p => p.life > 0);
  for (const f of floats) { f.t -= dt; f.y -= 18 * dt; }
  floats = floats.filter(f => f.t > 0);
  for (const p of props) p.flash = Math.max(0, p.flash - dt);
  shake = Math.max(0, shake - dt * 20); hurtFlash = Math.max(0, hurtFlash - dt); bannerT = Math.max(0, bannerT - dt);
}
