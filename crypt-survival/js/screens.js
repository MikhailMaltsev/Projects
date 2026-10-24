// Crypt Survival — текст, отрисовка кадра, HUD и экраны
'use strict';

/* ================= text =================
   As in pixel-crypt (render.js, txtMask): the font is drawn 8× bigger and sampled at the centre of every font pixel
   into a hard mask, so the letters stay on the pixel grid; cached per string + size + colour */
const txtBig = document.createElement('canvas').getContext('2d', { willReadFrequently: true });
const txtCache = new Map();
let txtFontReady = false;
function txtMask(s, size, col) {
  const key = size + '|' + col + '|' + s;
  let m = txtCache.get(key);
  if (m) return m;
  const big = '64px "Press Start 2P", ui-monospace, monospace';
  txtBig.font = big;
  const bw = Math.ceil(txtBig.measureText(s).width), bh = 64, k = size / 8, w = Math.max(1, Math.ceil(bw / 8 * k)), h = Math.round(8 * k);
  if (txtBig.canvas.width < bw + 8 || txtBig.canvas.height < bh + 8) { txtBig.canvas.width = Math.max(txtBig.canvas.width, bw + 8); txtBig.canvas.height = bh + 8; txtBig.font = big; }
  txtBig.clearRect(0, 0, txtBig.canvas.width, txtBig.canvas.height);
  txtBig.textBaseline = 'top'; txtBig.textAlign = 'left'; txtBig.fillStyle = '#fff'; txtBig.fillText(s, 0, 0);
  const sw = bw || 1, src = txtBig.getImageData(0, 0, sw, bh).data;
  m = document.createElement('canvas'); m.width = w; m.height = h;
  const g = m.getContext('2d'), out = g.createImageData(w, h), od = out.data;
  for (let py = 0; py < h; py++) {
    const sy = Math.min(bh - 1, Math.floor(py / k) * 8 + 4);
    for (let px = 0; px < w; px++) { const sx = Math.min(sw - 1, Math.floor(px / k) * 8 + 4); if (src[(sy * sw + sx) * 4 + 3] > 127) od[(py * w + px) * 4 + 3] = 255; }
  }
  g.putImageData(out, 0, 0);
  g.globalCompositeOperation = 'source-in'; g.fillStyle = col; g.fillRect(0, 0, w, h);
  if (!txtFontReady) txtFontReady = !document.fonts || document.fonts.check('8px "Press Start 2P"');
  if (txtFontReady) { if (txtCache.size > 600) txtCache.clear(); txtCache.set(key, m); }
  return m;
}
function txt(s, x, y, col = '#fff', size = 8, align = 'left', shadow = true) {
  s = String(s);
  const m = txtMask(s, size, col), dx = Math.round(align === 'center' ? x - m.width / 2 : align === 'right' ? x - m.width : x), dy = Math.round(y);
  if (shadow) ctx.drawImage(txtMask(s, size, '#000'), dx + 1, dy + 1);
  ctx.drawImage(m, dx, dy);
}
function box(x, y, w, h, on, fill = '#1b1527') {
  ctx.fillStyle = '#000'; ctx.fillRect(x - 1, y - 1, w + 2, h + 2);
  ctx.fillStyle = on ? '#f0a93a' : '#33294a'; ctx.fillRect(x, y, w, h);
  ctx.fillStyle = fill; ctx.fillRect(x + 1, y + 1, w - 2, h - 2);
}
// pad = extra px round the box that still count (a small button under a finger)
const hitBox = (b, x, y, pad = 0) => x >= b[0] - pad && y >= b[1] - pad && x < b[0] + b[2] + pad && y < b[1] + b[3] + pad;

/* ================= the frame ================= */
const VIGNETTE = (() => { const g = ctx.createRadialGradient(W / 2, H / 2, 50, W / 2, H / 2, 190); g.addColorStop(0, 'rgba(6,4,12,0)'); g.addColorStop(1, 'rgba(6,4,12,.7)'); return g; })();
function draw() {
  ctx.fillStyle = '#0d0a14'; ctx.fillRect(0, 0, W, H);
  if (state === 'title') { drawTitle(); return; }
  const sx = shake ? rr(-shake, shake) : 0, sy = shake ? rr(-shake, shake) : 0;
  ox = Math.round(P.x - W / 2 + sx); oy = Math.round(P.y - H / 2 + sy);
  drawFloor(); drawGround(); drawGemsItems();
  const list = [];
  propSortList(list); enemySortList(list);
  shadowAt(P.x, P.y + 4, 10);
  list.push({ y: P.y + P.h / 2, f: drawPlayer });
  list.sort((a, b) => a.y - b.y);
  for (const o of list) o.f();
  drawSwings(); drawProjs();
  for (const p of parts) { ctx.globalAlpha = clamp(p.life / p.max * 1.5, 0, 1); ctx.fillStyle = p.col; ctx.fillRect(Math.round(p.x - ox), Math.round(p.y - oy), p.sz, p.sz); }
  ctx.globalAlpha = 1;
  ctx.fillStyle = VIGNETTE; ctx.fillRect(0, 0, W, H);
  drawLights();
  for (const f of floats) { ctx.globalAlpha = clamp(f.t * 2, 0, 1); txt(f.text, f.x - ox, f.y - oy - 4, f.col, 8, 'center'); }
  ctx.globalAlpha = 1;
  if (hurtFlash > 0) { ctx.fillStyle = 'rgba(224,69,58,' + hurtFlash * 1.5 + ')'; ctx.fillRect(0, 0, W, H); }
  drawHud(); drawStick();
  if (bannerT > 0) { ctx.globalAlpha = Math.min(1, bannerT * 2); txt(bannerText, W / 2, 46, bannerCol, 16, 'center'); ctx.globalAlpha = 1; }
  if (state === 'levelup') drawLevelUp();
  else if (state === 'pause') drawPause();
  else if (state === 'dead' || state === 'win') drawEnd();
}

/* ================= HUD ================= */
const PAUSE_BTN = [W - 14, 12, 11, 11];
function drawHud() {
  // the shards bar across the top
  const need = xpNeed(G.lvl + G.pending), k = clamp(G.xp / need, 0, 1);
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, 5); ctx.fillStyle = '#1a2a4a'; ctx.fillRect(1, 1, W - 2, 3);
  ctx.fillStyle = '#5ab8ff'; ctx.fillRect(1, 1, Math.round((W - 2) * k), 3); ctx.fillStyle = '#e0f6ff'; ctx.fillRect(1, 1, Math.round((W - 2) * k), 1);
  txt('УР ' + (G.lvl + G.pending), W - 18, 7, '#9fe8ff', 8, 'right');
  txt(fmtTime(RUN_T - G.time), W / 2, 8, G.time > RUN_T - 60 ? '#ff5a4a' : '#ece4d2', 8, 'center');
  const hearts = Math.ceil(P.maxhp / HPH);
  for (let i = 0; i < hearts; i++) ctx.drawImage(S_HEART[clamp(P.hp - i * HPH, 0, HPH)], 4 + i * 9, 8);
  ctx.drawImage(S_COIN, 4, 18); txt(G.gold, 4 + S_COIN.width + 3, 19, '#ffd23f');
  ctx.drawImage(S_SKULL_MINI, 4, 30); txt(G.kills, 4 + S_SKULL_MINI.width + 3, 29, '#c3cad8');
  // pause
  box(PAUSE_BTN[0], PAUSE_BTN[1], PAUSE_BTN[2], PAUSE_BTN[3], false);
  ctx.fillStyle = '#ece4d2'; ctx.fillRect(PAUSE_BTN[0] + 3, PAUSE_BTN[1] + 3, 2, 5); ctx.fillRect(PAUSE_BTN[0] + 6, PAUSE_BTN[1] + 3, 2, 5);
  // the slots: weapons over passives, level pips at the bottom of each
  for (let row = 0; row < 2; row++) for (let i = 0; i < SLOTS; i++) {
    const it = (row ? P.passives : P.weapons)[i], x = 4 + i * 18, y = H - 38 + row * 19;
    box(x, y, 16, 17, false, it ? '#1b1527' : '#0d0a14');
    if (!it) continue;
    drawSlotIcon(row ? 'p' : 'w', it.id, x + 2, y + 1);
    const max = row ? PASSIVE_MAX : WEAPON_MAX, step = row ? 3 : 2;
    for (let p = 0; p < max; p++) { ctx.fillStyle = p < it.lv ? '#ffd23f' : '#3c2a4e'; ctx.fillRect(x + 1 + p * step + (row ? 1 : 0), y + 14, 1, 2); }
  }
}

/* ================= screens ================= */
const CARD = { x: 24, y: 34, w: 272, h: 40, gap: 6 };
const cardBox = i => [CARD.x, CARD.y + i * (CARD.h + CARD.gap), CARD.w, CARD.h];
function drawLevelUp() {
  ctx.fillStyle = 'rgba(8,6,14,.72)'; ctx.fillRect(0, 0, W, H);
  txt('НОВЫЙ УРОВЕНЬ!', W / 2, 14, '#ffd23f', 8, 'center');
  choices.forEach((c, i) => {
    const [x, y, w, h] = cardBox(i), on = i === sel;
    box(x, y, w, h, on, on ? '#2a2040' : '#1b1527');
    box(x + 6, y + 10, 20, 20, false, '#0d0a14');
    if (c.kind === 'w' || c.kind === 'p') drawSlotIcon(c.kind, c.id, x + 10, y + 14);
    else if (c.kind === 'gold') drawBag(x + 16, y + 21); else ctx.drawImage(S_HEART_ITEM, x + 12, y + 16);
    const [name, desc] = choiceText(c);
    txt(name, x + 34, y + 8, on ? '#ffd23f' : '#ece4d2');
    if (c.lv) txt(c.lv === 1 ? 'НОВОЕ' : 'УР ' + c.lv, x + w - 8, y + 8, c.lv === 1 ? '#5ae07a' : '#9fe8ff', 8, 'right');
    txt(desc, x + 34, y + 24, '#9186a6');
  });
}
const PAUSE_GO = [W / 2 - 70, 88, 140, 20], PAUSE_MENU = [W / 2 - 70, 116, 140, 20];
function drawPause() {
  ctx.fillStyle = 'rgba(8,6,14,.72)'; ctx.fillRect(0, 0, W, H);
  txt('ПАУЗА', W / 2, 52, '#ffd23f', 16, 'center');
  box(...PAUSE_GO, true); txt('ДАЛЬШЕ', W / 2, PAUSE_GO[1] + 6, '#ece4d2', 8, 'center');
  box(...PAUSE_MENU, false); txt('В МЕНЮ', W / 2, PAUSE_MENU[1] + 6, '#ece4d2', 8, 'center');
}
function drawEnd() {
  const k = Math.min(1, stateT / 0.6);
  ctx.fillStyle = 'rgba(8,6,14,' + 0.8 * k + ')'; ctx.fillRect(0, 0, W, H);
  if (k < 1) return;
  const win = state === 'win';
  txt(win ? 'РАССВЕТ' : 'ТЫ ПАЛ', W / 2, 30, win ? '#ffd23f' : '#e0453a', 16, 'center');
  txt(win ? 'Склеп выпустил тебя' : 'Склеп забрал ещё одного', W / 2, 54, '#9186a6', 8, 'center');
  const rows = [['ВРЕМЯ', fmtTime(G.time)], ['УРОВЕНЬ', G.lvl + G.pending], ['УБИТО', G.kills], ['ЗОЛОТО', '+' + G.gold]];
  rows.forEach(([a, b], i) => { txt(a, 80, 74 + i * 14, '#c3cad8'); txt(b, 240, 74 + i * 14, i === 0 && G.bestNew ? '#ffd23f' : '#ece4d2', 8, 'right'); });
  if (G.bestNew) txt('НОВЫЙ РЕКОРД!', W / 2, 132, '#ffd23f', 8, 'center');
  if (stateT > 1.2 && Math.floor(tt * 2) % 2) txt(coarsePointer.matches ? 'ТАП — В МЕНЮ' : 'ENTER — В МЕНЮ', W / 2, 150, '#f0a93a', 8, 'center');
}
const heroBox = i => [W / 2 - 138 + i * 94, 60, 88, 66];
const START_BTN = [W / 2 - 60, 150, 120, 20];
function drawTitle() {
  // the field under the title: the floor tiles drifting slowly
  ox = Math.round(tt * 8); oy = Math.round(tt * 5);
  drawFloor(); ctx.fillStyle = VIGNETTE; ctx.fillRect(0, 0, W, H); ctx.fillStyle = 'rgba(8,6,14,.45)'; ctx.fillRect(0, 0, W, H);
  txt('CRYPT SURVIVAL', W / 2, 14, '#f0a93a', 16, 'center');
  txt('Продержись 15 минут', W / 2, 38, '#9186a6', 8, 'center');
  HERO_IDS.forEach((id, i) => {
    const [x, y, w, h] = heroBox(i), on = id === hero, img = S_HERO[id][Math.floor(tt * 3) % 2];
    box(x, y, w, h, on, on ? '#2a2040' : '#1b1527');
    ctx.drawImage(img, x + w / 2 - 16, y + 4, 32, 32);
    txt(HEROES[id].name, x + w / 2, y + 40, on ? '#ffd23f' : '#ece4d2', 8, 'center');
    drawSlotIcon('w', HEROES[id].weapon, x + w / 2 - 6, y + 51);
  });
  txt('РЕКОРД ' + fmtTime(best[hero] || 0), W / 2 - 4, 134, '#c3cad8', 8, 'right');
  ctx.drawImage(S_COIN, W / 2 + 8, 133); txt(wallet, W / 2 + 12 + S_COIN.width, 134, '#ffd23f');
  box(...START_BTN, true); txt('СТАРТ', W / 2, START_BTN[1] + 6, '#ece4d2', 8, 'center');
  txt(coarsePointer.matches ? 'Палец на поле — ходьба' : 'WASD — ходьба, ESC — пауза', W / 2, 178, '#5d5278', 8, 'center');
}

/* ================= keys and taps for the screens ================= */
function onKey(code) {
  const ok = code === 'Enter' || code === 'Space';
  if (state === 'title') {
    const i = HERO_IDS.indexOf(hero);
    if (code === 'ArrowLeft' || code === 'KeyA') hero = HERO_IDS[(i + HERO_IDS.length - 1) % HERO_IDS.length];
    else if (code === 'ArrowRight' || code === 'KeyD') hero = HERO_IDS[(i + 1) % HERO_IDS.length];
    else if (ok) startRun();
  } else if (state === 'play') { if (code === 'Escape' || code === 'KeyP') state = 'pause'; }
  else if (state === 'pause') { if (code === 'Escape' || code === 'KeyP' || ok) state = 'play'; else if (code === 'KeyQ') toTitle(); }
  else if (state === 'levelup') {
    if (code === 'ArrowUp' || code === 'KeyW') sel = (sel + choices.length - 1) % choices.length;
    else if (code === 'ArrowDown' || code === 'KeyS') sel = (sel + 1) % choices.length;
    else if (stateT > 0.3 && /^Digit[1-3]$/.test(code) && choices[+code[5] - 1]) takeChoice(choices[+code[5] - 1]);
    else if (stateT > 0.3 && ok) takeChoice(choices[sel]);
  } else if ((state === 'dead' || state === 'win') && stateT > 1.2 && ok) toTitle();
}
function onTap(x, y) {
  if (state === 'title') {
    HERO_IDS.forEach((id, i) => { if (hitBox(heroBox(i), x, y)) hero = id; });
    if (hitBox(START_BTN, x, y)) startRun();
  } else if (state === 'pause') { if (hitBox(PAUSE_GO, x, y)) state = 'play'; else if (hitBox(PAUSE_MENU, x, y)) toTitle(); }
  else if (state === 'levelup') { choices.forEach((c, i) => { if (stateT > 0.3 && hitBox(cardBox(i), x, y)) takeChoice(c); }); }
  else if ((state === 'dead' || state === 'win') && stateT > 1.2) toTitle();
}
