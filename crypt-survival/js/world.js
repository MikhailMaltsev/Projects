// Crypt Survival — бесконечное поле: чанки, пол, декор, разрушаемые предметы
'use strict';

/* ================= chunks =================
   The field is cut into CH×CH-tile chunks, each made from the run seed and its own coordinates, so a chunk left behind
   and met again looks the same. Chunks within LOAD_R of the hero's are kept, the ones past DROP_R are dropped;
   what was broken stays broken for the whole run (broken: chunk key -> prop indices). */
const LOAD_R = 2, DROP_R = 3;
const chunks = new Map(), broken = new Map();
const ckey = (cx, cy) => cx + ',' + cy;
// the stone of the old crypt (pixel-crypt crypt.js, CZ1)
const CZ = { st: '#4a3f5e', stL: '#5e5274', stD: '#2e2540', gap: '#0d0914', bone: '#cfc8b0', boneD: '#8f8670', oss: '#9a927c', ossD: '#625c4e' };

/* ---------- what lies on the field ----------
   tw = tiles wide (the picture is drawn from the cell's top-left), w/h = the solid box at the foot of the picture,
   hp = blows to break it (0 = can't be broken), big = a big piece: better loot, and it may hold a surprise (rise) */
const PROPS = {
  candle: { tw: 1, w: 12, h: 5, hp: 1, solid: true, light: true },
  bonepile: { tw: 1, w: 12, h: 6, hp: 1, solid: false },
  adv: { tw: 1, w: 14, h: 7, hp: 3, solid: true, big: true },
  coffin: { tw: 2, w: 26, h: 9, hp: 4, solid: true, big: true, rise: 'skel' },
  sarc: { tw: 2, w: 30, h: 9, hp: 5, solid: true, big: true, rise: 'skel' },
  altar: { tw: 2, w: 26, h: 9, hp: 5, solid: true, big: true },
  rubble: { tw: 2, w: 28, h: 9, hp: 0, solid: true }
};
// the clear spot round the start: nothing solid within START_CLEAR px of it
const START_CLEAR = 64;
function genChunk(cx, cy) {
  const r = mulberry(chunkSeed(cx, cy, G.seed)), x0 = cx * CHPX, y0 = cy * CHPX;
  const ground = [], list = [], used = new Set();
  const free = (tx, ty, tw) => { for (let k = 0; k < tw; k++) if (used.has((ty) * CH + tx + k)) return false; return true; };
  const take = (tx, ty, tw) => { for (let k = 0; k < tw; k++) used.add(ty * CH + tx + k); };
  // the floor's own decor: walked over, never hit
  const ng = 4 + (r() * 6 | 0);
  for (let k = 0; k < ng; k++) {
    const kind = pickW([['slab', 3], ['grate', 2], ['pebble', 5], ['bones', 2], ['corpse', 1]], r());
    const tw = kind === 'slab' ? 2 : 1, tx = r() * (CH - tw) | 0, ty = r() * CH | 0;
    if (!free(tx, ty, tw)) continue;
    take(tx, ty, tw); ground.push({ kind, x: x0 + tx * T, y: y0 + ty * T, s: tx * 7.3 + ty * 3.1 + cx * 13 + cy * 17 });
  }
  // a heap of rubble now and then, and 1–3 pieces to break
  const pieces = [];
  if (r() < 0.35) pieces.push('rubble');
  for (let k = 1 + (r() * 3 | 0); k > 0; k--) pieces.push(pickW([['candle', 4], ['bonepile', 3], ['coffin', 2], ['sarc', 2], ['altar', 1], ['adv', 1]], r()));
  const gone = broken.get(ckey(cx, cy));
  pieces.forEach((kind, idx) => {
    const d = PROPS[kind];
    for (let tries = 0; tries < 6; tries++) {
      const tx = 1 + (r() * (CH - d.tw - 2) | 0), ty = 1 + (r() * (CH - 2) | 0);
      const px = x0 + tx * T, py = y0 + ty * T, x = px + d.tw * 8, y = py + 14 - d.h / 2;
      if (!free(tx, ty, d.tw) || (d.solid && Math.hypot(x - CHPX / 2, y - CHPX / 2) < START_CLEAR)) continue;
      take(tx, ty, d.tw);
      if (!(gone && gone.has(idx))) list.push({ kind, idx, cx, cy, px, py, x, y, w: d.w, h: d.h, hp: d.hp, solid: d.solid, flash: 0, s: tx * 7.3 + ty * 3.1 });
      break;
    }
  });
  return { cx, cy, ground, props: list };
}
// keep the chunks round the hero: returns true when the set changed (the props list is then rebuilt)
let chunkAt = '';
function updChunks() {
  const pcx = Math.floor(P.x / CHPX), pcy = Math.floor(P.y / CHPX), here = ckey(pcx, pcy);
  if (here === chunkAt) return;
  chunkAt = here;
  for (let y = pcy - LOAD_R; y <= pcy + LOAD_R; y++) for (let x = pcx - LOAD_R; x <= pcx + LOAD_R; x++) {
    const k = ckey(x, y); if (!chunks.has(k)) chunks.set(k, genChunk(x, y));
  }
  for (const [k, c] of chunks) if (Math.abs(c.cx - pcx) > DROP_R || Math.abs(c.cy - pcy) > DROP_R) chunks.delete(k);
  props = []; for (const c of chunks.values()) for (const p of c.props) props.push(p);
}
function resetWorld() { chunks.clear(); broken.clear(); chunkAt = ''; props = []; }

/* ---------- collisions with the solid pieces ---------- */
function solidHit(x, y, hw, hh) {
  for (const p of props) if (p.solid && Math.abs(x - p.x) < hw + p.w / 2 && Math.abs(y - p.y) < hh + p.h / 2) return p;
  return null;
}
// move by (dx, dy), sliding along whatever is solid
function moveEnt(e, dx, dy) {
  const hw = e.w / 2, hh = e.h / 4; // only the feet collide: a body may stand in front of a sarcophagus
  if (dx && !solidHit(e.x + dx, e.y, hw, hh)) e.x += dx;
  if (dy && !solidHit(e.x, e.y + dy, hw, hh)) e.y += dy;
}

/* ---------- breaking ---------- */
// one blow on a piece; hit = the Set of a swing or a shot, so one blow counts once
function hitProp(p, hit) {
  if (!p.hp || (hit && hit.has(p))) return false;
  if (hit) hit.add(p);
  p.hp--; p.flash = 0.1;
  burst(p.x, p.y - 4, p.kind === 'candle' ? '#d8cca0' : CZ.stL, 5, 50); Snd.sfx('crack');
  if (p.hp <= 0) breakProp(p);
  return true;
}
function breakProp(p) {
  const i = props.indexOf(p); if (i >= 0) props.splice(i, 1);
  const c = chunks.get(ckey(p.cx, p.cy)); if (c) c.props = c.props.filter(q => q !== p);
  const k = ckey(p.cx, p.cy); if (!broken.has(k)) broken.set(k, new Set()); broken.get(k).add(p.idx);
  const d = PROPS[p.kind];
  burst(p.x, p.y - 4, CZ.stD, d.big ? 18 : 8, 80); burst(p.x, p.y - 4, CZ.bone, d.big ? 8 : 4, 60);
  if (d.light) burst(p.x, p.y - 6, '#ff9a30', 6, 40);
  addShake(d.big ? 2.5 : 1);
  propLoot(p, d);
}
/* the loot (GDD §9): a small piece 55% gold, 25% a heart, 10% the magnet, 5% the bomb; a big one always one of heart / gold bag,
   plus 15% the magnet, 10% the bomb; one big piece in RISE_P lets out who was lying in it */
const RISE_P = 0.1;
function propLoot(p, d) {
  const at = (dx, dy) => ({ x: p.x + dx, y: p.y - 2 + dy });
  if (d.rise && Math.random() < RISE_P) {
    const e = mkEnemy(d.rise, p.x, p.y + 2); e.rise = 0.6; enemies.push(e);
    floatText(p.x, p.y - 16, 'ВСТАЛ!', '#e8e4d0'); Snd.sfx('raise');
    return;
  }
  if (!d.big) {
    const k = pickW([['coin', 55], ['heart', 25], ['magnet', 10], ['bomb', 5], [null, 5]]);
    if (k) dropItem(k, at(0, 0));
    return;
  }
  dropItem(Math.random() < 0.5 ? 'heart' : 'bag', at(-5, 0));
  if (Math.random() < 0.15) dropItem('magnet', at(5, 2));
  if (Math.random() < 0.10) dropItem('bomb', at(0, 5));
}

/* ================= drawing =================
   The pictures are pixel-crypt's (crypt.js: drawCryptDecor, drawCSlab, drawCryptProp) with the old crypt's stone */
const cR = (c, x, y, w = 1, h = 1) => { ctx.fillStyle = c; ctx.fillRect(x, y, w, h); };
function shadowAt(x, y, w) {
  ctx.fillStyle = 'rgba(0,0,0,.35)';
  const sx = Math.round(x - ox), sy = Math.round(y - oy);
  ctx.fillRect(sx - w / 2 + 1, sy, w - 2, 1); ctx.fillRect(sx - w / 2, sy + 1, w, 2); ctx.fillRect(sx - w / 2 + 1, sy + 3, w - 2, 1);
}
function drawFloor() {
  const x0 = Math.floor(ox / T), y0 = Math.floor(oy / T);
  for (let y = y0; y <= y0 + Math.ceil(H / T); y++) for (let x = x0; x <= x0 + Math.ceil(W / T); x++) {
    const h = hash1(x * 3.7 + y * 11.3), v = h < 0.7 ? 0 : h < 0.82 ? 1 : h < 0.94 ? 2 : 3;
    ctx.drawImage(TS_OLD.floor[v], x * T - ox, y * T - oy);
  }
}
const onScreen = (x, y, m = 40) => x - ox > -m && y - oy > -m && x - ox < W + m && y - oy < H + m;
function drawGround() {
  for (const c of chunks.values()) for (const g of c.ground) {
    if (!onScreen(g.x, g.y)) continue;
    const px = Math.round(g.x - ox), py = Math.round(g.y - oy);
    if (g.kind === 'slab') drawSlab(px, py);
    else if (g.kind === 'grate') drawGrate(px, py, g.s);
    else if (g.kind === 'pebble') drawPebble(px, py);
    else if (g.kind === 'bones') drawBonePile(px, py, g.s);
    else if (g.kind === 'corpse') drawCorpse(px, py, g.s);
  }
  // the flat pieces lie under everyone
  for (const p of props) if (!p.solid && onScreen(p.x, p.y)) drawProp(p);
}
// the tall pieces go into the list sorted with the creatures by their foot
function propSortList(list) {
  for (const p of props) if (p.solid && onScreen(p.x, p.y)) list.push({ y: p.y + p.h / 2, f: () => drawProp(p) });
}
function drawProp(p) {
  const px = Math.round(p.px - ox), py = Math.round(p.py - oy);
  if (p.flash > 0) ctx.globalAlpha = 0.6;
  const k = p.kind;
  if (k === 'candle') drawCandles(px, py, p.s);
  else if (k === 'bonepile') drawBonePile(px, py, p.s);
  else if (k === 'sarc') drawSarc(px, py);
  else if (k === 'altar') drawAltar(px, py);
  else if (k === 'coffin') drawCoffin(px, py);
  else if (k === 'rubble') drawRubble(px, py);
  else if (k === 'adv') drawAdventurer(px, py);
  ctx.globalAlpha = 1;
}
function drawSlab(px, py) {
  const z = CZ;
  cR(z.stD, px + 1, py + 2, 30, 12); cR(z.st, px + 2, py + 3, 28, 10); cR(z.stL, px + 2, py + 3, 28, 1);
  cR(z.stD, px + 6, py + 6, 20, 1); cR(z.stD, px + 8, py + 8, 16, 1); cR(z.stD, px + 11, py + 10, 10, 1);
  [[21, 3], [21, 4], [22, 5], [22, 6], [23, 7], [24, 8], [24, 9], [25, 10], [26, 11], [26, 12]].forEach(([a, b]) => cR(z.gap, px + a, py + b));
  cR('rgba(0,0,0,.25)', px + 2, py + 13, 28, 1);
}
function drawGrate(px, py, s) {
  cR(CZ.stD, px + 2, py + 2, 12, 12); cR('#06040a', px + 3, py + 3, 10, 10);
  for (const a of [5, 7, 9, 11]) cR('#4a4458', px + a, py + 3, 1, 10);
  cR('#4a4458', px + 3, py + 7, 10, 1); cR('#6a6478', px + 3, py + 3, 10, 1);
  if (Math.sin(tt * 0.8 + s) > 0.6) cR('#5a7a8a', px + 8, py + 9 + ((tt * 20) % 3 | 0));
}
function drawPebble(px, py) {
  const z = CZ;
  for (const [a, b, w, h, c] of [[3, 9, 3, 2, z.st], [4, 8, 2, 1, z.stL], [9, 11, 2, 2, z.stD], [11, 6, 2, 1, z.st], [7, 13, 1, 1, z.stL]]) cR(c, px + a, py + b, w, h);
}
function drawCandles(px, py, s) {
  cR('#9a8e70', px + 2, py + 12, 12, 2);
  for (const [a, h, o] of [[3, 4, 0], [7, 6, 1.3], [11, 3, 2.1]]) {
    const top = py + 12 - h; cR('#d8cca0', px + a, top, 2, h); cR('#a89870', px + a + 1, top + 1, 1, h - 1);
    const fl = Math.sin(tt * 13 + o + s) > 0 ? 1 : 0; cR('#ff9a30', px + a, top - 2 - fl, 1, 2); cR('#fff0b0', px + a, top - 1);
  }
}
function drawBonePile(px, py, s) {
  const z = CZ;
  cR('rgba(0,0,0,.3)', px + 1, py + 13, 14, 2);
  cR(z.ossD, px + 2, py + 10, 12, 3); cR(z.ossD, px + 4, py + 8, 8, 2);
  for (let i = 0; i < 6; i++) { const h = hash1(s + i * 2.7), a = px + 3 + (h * 9 | 0), b = py + 9 + (hash1(h * 7) * 3 | 0); cR(z.oss, a, b, h < 0.5 ? 3 : 1, h < 0.5 ? 1 : 2); }
  cR(z.oss, px + 6, py + 6, 4, 3); cR(z.gap, px + 7, py + 7); cR(z.gap, px + 9, py + 7);
}
function drawCorpse(px, py, s) {
  const x = px + 8, y = py + 8, flip = hash1(s) < 0.5;
  shadowAt(x + ox, y + 3 + oy, 14);
  ctx.save(); ctx.translate(x, y + 5); ctx.scale(flip ? -1 : 1, 1); ctx.drawImage(S_CORPSE, -8, -9); ctx.restore();
}
function drawSarc(px, py) {
  const z = CZ;
  cR('rgba(0,0,0,.35)', px + 1, py + 14, 30, 2);
  cR(z.stD, px + 1, py + 2, 30, 13); cR(z.st, px + 2, py + 8, 28, 6);
  for (const a of [4, 13, 22]) { cR(z.stD, px + a, py + 9, 6, 4); cR(z.stL, px + a + 1, py + 9, 4, 1); }
  cR(z.stD, px + 1, py - 1, 30, 9); cR(z.stL, px + 2, py, 28, 7); cR(z.st, px + 2, py + 6, 28, 1);
  cR(z.st, px + 5, py + 2, 3, 3); cR(z.st, px + 9, py + 2, 16, 3); cR(z.stD, px + 12, py + 3, 6, 1);
  cR(z.st, px + 26, py + 2, 2, 1); cR(z.st, px + 26, py + 4, 2, 1);
}
function drawAltar(px, py) {
  const z = CZ;
  cR('rgba(0,0,0,.35)', px + 2, py + 14, 28, 2);
  cR(z.stD, px + 3, py + 4, 26, 11); cR(z.st, px + 4, py + 7, 24, 7);
  cR(z.stL, px + 2, py + 2, 28, 3); cR(z.stD, px + 2, py + 5, 28, 1); cR('#9a8e70', px + 4, py + 2, 24, 1);
  cR('#4a1622', px + 10, py + 5, 12, 8); cR('#2e0c16', px + 10, py + 5, 12, 1); cR('#6a2a34', px + 12, py + 8, 8, 1);
  for (let i = 0; i < 6; i++) cR('#4a1622', px + 10 + i * 2, py + 13);
  cR(z.bone, px + 14, py - 2, 5, 4); cR(z.gap, px + 15, py - 1); cR(z.gap, px + 17, py - 1); cR(z.bone, px + 15, py + 2, 3, 1);
  for (const [a, h, o] of [[5, 5, 0], [8, 3, 1], [23, 4, 2], [26, 6, 3]]) {
    const top = py + 2 - h; cR('#d8cca0', px + a, top, 1, h);
    const fl = Math.sin(tt * 12 + o) > 0 ? 1 : 0; cR('#ff9a30', px + a, top - 2 - fl, 1, 2); cR('#fff0b0', px + a, top - 1);
  }
}
function drawCoffin(px, py) {
  cR('rgba(0,0,0,.35)', px + 1, py + 14, 30, 2);
  const coffin = (x, y, w, open) => {
    cR('#1a100c', x, y + 1, w, 6); cR('#1a100c', x + 1, y, w - 3, 8);
    cR('#4a3226', x + 1, y + 1, w - 3, 5); cR('#6a4a34', x + 1, y + 1, w - 3, 1); cR('#3a261c', x + 2, y + 4, w - 5, 1); cR('#2a1a12', x + (w >> 1), y + 1, 1, 5);
    if (open) { cR('#0a0608', x + 2, y + 1, w - 6, 2); cR('#1a100c', x + 3, y - 3, w - 5, 3); cR('#6a4a34', x + 4, y - 3, w - 7, 2); }
  };
  coffin(px + 1, py + 7, 22, false); coffin(px + 3, py, 20, true);
  for (let i = 0; i < 12; i++) cR('#5a4030', px + 28 - (i >> 2), py + i);
  cR('#6a6478', px + 24, py + 11, 3, 4); cR('#8a8498', px + 24, py + 11);
  cR('#8a7a5a', px + 25, py + 13, 6, 1); cR('#6a5a3a', px + 26, py + 14, 5, 1); cR('#8a7a5a', px + 27, py + 15, 3, 1);
}
function drawRubble(px, py) {
  const z = CZ;
  cR('rgba(0,0,0,.35)', px + 1, py + 14, 30, 2);
  cR('#3a2a20', px + 1, py + 4, 14, 2); cR('#5a4030', px + 1, py + 4, 14, 1);
  [[3, 8, 8, 6], [9, 4, 9, 9], [17, 6, 8, 8], [24, 9, 6, 5], [12, 1, 6, 4], [6, 11, 5, 3], [20, 11, 6, 3]].forEach(([a, b, w, h], i) => {
    cR(z.stD, px + a, py + b, w, h); cR(i % 2 ? z.st : z.stL, px + a + 1, py + b, w - 2, h - 2); cR(z.stL, px + a + 1, py + b, w - 3, 1);
  });
}
function drawAdventurer(px, py) {
  const z = CZ;
  cR('rgba(0,0,0,.35)', px - 2, py + 13, 20, 3);
  cR('#4a3a26', px + 1, py - 2, 8, 9); cR('#2e2418', px + 1, py - 2, 8, 1); cR('#6a5236', px + 2, py - 1, 6, 1);
  cR('#2e2a3a', px + 4, py + 2, 8, 8); cR('#3e3a4e', px + 5, py + 3, 6, 6);
  cR(z.bone, px + 10, py + 3, 4, 3); cR(z.gap, px + 11, py + 4); cR(z.gap, px + 13, py + 4);
  cR('#7d869c', px + 9, py - 1, 6, 4); cR('#c3cad8', px + 10, py - 1, 4, 1);
  cR('#2e2a38', px + 8, py + 10, 8, 2); cR('#2a1f1a', px + 15, py + 9, 2, 3); cR('#2e2a38', px + 6, py + 12, 7, 2); cR('#2a1f1a', px + 13, py + 12, 2, 3);
  cR('#8a92a6', px + 2, py + 15, 12, 1); cR('#5a4030', px + 14, py + 15, 3, 1);
  cR('#6b5a48', px - 3, py + 7, 5, 1); cR('#3a3428', px - 3, py + 8, 5, 5); cR('#1a1810', px - 2, py + 9, 3, 3); cR('#6b5a48', px - 3, py + 13, 5, 1);
}
// a corpse in rags (pixel-crypt crypt.js, CRAWL_A)
const S_CORPSE = build(['................', '................', '...........kkk..', '..........khhhk.', '...kkkkkkkhsSek.', '..kRRrRRRrRksssk', '.kRRRRrRRRRRkSk.', 'krRrkRRRRRrkSSsk', '.k.k.kkkkkkk.kkk'],
  { k: '#140e14', R: '#6e5a5a', r: '#4a3a3c', h: '#3a2e34', s: '#aab896', S: '#76825e', e: '#4a4038' });
// the warm light of the candles over the dark of the field
function drawLights() {
  ctx.globalCompositeOperation = 'lighter';
  for (const p of props) {
    if (!PROPS[p.kind].light && p.kind !== 'altar') continue;
    if (!onScreen(p.x, p.y, 60)) continue;
    const r = 30 + Math.sin(tt * 11 + p.s) * 1.5, x = p.x - ox, y = p.y - 6 - oy;
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, 'rgba(255,170,80,.22)'); g.addColorStop(1, 'rgba(255,170,80,0)');
    ctx.fillStyle = g; ctx.fillRect(x - r, y - r, r * 2, r * 2);
  }
  ctx.globalCompositeOperation = 'source-over';
}
