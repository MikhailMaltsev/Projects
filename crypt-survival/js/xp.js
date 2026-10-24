// Crypt Survival — осколки душ, предметы, уровни и выбор улучшений
'use strict';

/* ================= soul shards (GDD §4) =================
   Blue 1, green 5, red 25; a bigger value is a bigger shard. A shard within the hero's magnet radius flies to him;
   past GEM_MAX on the field the far ones melt into one */
const GEM_MAX = 150, GEM_PULL = 260;
const GEM_PAL = [
  { k: '#0c1830', w: '#e0f6ff', c: '#5ab8ff', C: '#2a6ad8' },
  { k: '#0a2012', w: '#e0ffe0', c: '#5ae07a', C: '#2a9a4a' },
  { k: '#2a0810', w: '#ffe0e0', c: '#ff5a5a', C: '#b81a2a' }
];
const S_GEM = GEM_PAL.map(p => build(['.k.', 'kwk', 'kck', 'kCk', '.k.'], p));
const S_GEM_BIG = build(['..k..', '.kwk.', 'kwcck', 'kcCCk', 'kcCCk', '.kCk.', '..k..'], GEM_PAL[2]);
const gemTier = v => v >= 25 ? 2 : v >= 5 ? 1 : 0;
function dropGem(x, y, v) { gems.push({ x, y, v, pull: false, t: Math.random() * 3 }); }
function updGems(dt) {
  for (const g of gems) {
    g.t += dt;
    const dx = P.x - g.x, dy = P.y - 2 - g.y, d = Math.hypot(dx, dy);
    if (!g.pull && d < P.magnet) g.pull = true;
    if (g.pull) {
      const sp = Math.max(60, GEM_PULL - d) * dt * (g.all ? 1.6 : 1);
      g.x += dx / (d || 1) * Math.min(d, sp); g.y += dy / (d || 1) * Math.min(d, sp);
      if (d < 6) { g.dead = true; gainXp(g.v); }
    }
  }
  gems = gems.filter(g => !g.dead);
  if (gems.length > GEM_MAX) { // the far ones melt into one red shard
    let sum = 0, at = null;
    for (const g of gems) if (!g.pull && Math.hypot(g.x - P.x, g.y - P.y) > 200) { sum += g.v; g.dead = true; at = at || g; }
    gems = gems.filter(g => !g.dead);
    if (at) dropGem(at.x, at.y, sum);
  }
}
let xpSnd = 0;
function gainXp(v) {
  G.xp += v;
  if (tt - xpSnd > 0.06) { xpSnd = tt; Snd.sfx('coin'); }
  while (G.xp >= xpNeed(G.lvl + G.pending)) { G.xp -= xpNeed(G.lvl + G.pending); G.pending++; }
}
// xp from level L to L+1 (GDD §4): XP_STEP[0] for the first, then +XP_STEP[1] a level up to 20, +[2] up to 40, +[3] after
const XP_STEP = [5, 10, 13, 16];
function xpNeed(L) {
  if (L < 20) return XP_STEP[0] + XP_STEP[1] * (L - 1);
  if (L < 40) return xpNeed(19) + XP_STEP[2] * (L - 19);
  return xpNeed(39) + XP_STEP[3] * (L - 39);
}

/* ================= items =================
   coin = a little gold, bag = a lot, heart = one heart back, magnet = every shard on the field flies to the hero,
   bomb = a skull that blows up every enemy on screen (giants take BOMB_DMG) */
const BOMB_DMG = 40;
function dropItem(kind, at) { items.push({ kind, x: at.x, y: at.y, t: Math.random() * 3 }); }
function updItems(dt) {
  for (const it of items) {
    it.t += dt;
    const d = Math.hypot(P.x - it.x, P.y - it.y);
    if (d > 12) { if (d < P.magnet * 0.6 && (it.kind === 'coin' || it.kind === 'bag')) { it.x += (P.x - it.x) / d * 120 * dt; it.y += (P.y - it.y) / d * 120 * dt; } continue; }
    it.dead = true; pickItem(it);
  }
  items = items.filter(it => !it.dead);
}
function pickItem(it) {
  const k = it.kind;
  if (k === 'coin') { const n = 1 + rnd(3); G.gold += n; floatText(it.x, it.y - 8, '+' + n, '#ffd23f'); Snd.sfx('coin'); }
  else if (k === 'bag') { G.gold += 10; floatText(it.x, it.y - 8, '+10', '#ffd23f'); Snd.sfx('chest'); }
  else if (k === 'heart') { P.hp = Math.min(P.maxhp, P.hp + HPH); floatText(it.x, it.y - 8, '+HP', '#ff8aa8'); Snd.sfx('heart'); burst(P.x, P.y, '#ff8aa8', 10, 50); }
  else if (k === 'magnet') { for (const g of gems) { g.pull = true; g.all = true; } floatText(it.x, it.y - 8, 'МАГНИТ', '#9fe8ff'); Snd.sfx('tele'); }
  else if (k === 'bomb') {
    hurtFlash = 0.25; addShake(6); Snd.sfx('boom');
    for (const e of [...enemies]) if (onScreen(e.x, e.y, 0)) { burst(e.x, e.y, '#ff7a1a', 6, 70); if (e.elite) damageEnemy(e, BOMB_DMG / P.might, it.x, it.y); else killEnemy(e); }
  }
}
function drawGemsItems() {
  for (const g of gems) {
    if (!onScreen(g.x, g.y, 8)) continue;
    const img = g.v > 25 ? S_GEM_BIG : S_GEM[gemTier(g.v)], bob = g.pull ? 0 : Math.round(Math.sin(g.t * 3) * 1);
    ctx.drawImage(img, Math.round(g.x - ox - img.width / 2), Math.round(g.y - oy - img.height / 2 + bob));
  }
  for (const it of items) {
    if (!onScreen(it.x, it.y, 10)) continue;
    const x = Math.round(it.x - ox), y = Math.round(it.y - oy + Math.sin(it.t * 3) * 1.2);
    shadowAt(it.x, it.y + 4, 8);
    if (it.kind === 'coin') ctx.drawImage(S_COIN, x - (S_COIN.width >> 1), y - (S_COIN.height >> 1));
    else if (it.kind === 'heart') ctx.drawImage(S_HEART_ITEM, x - (S_HEART_ITEM.width >> 1), y - (S_HEART_ITEM.height >> 1));
    else if (it.kind === 'bag') drawBag(x, y);
    else if (it.kind === 'magnet') drawSlotIcon('p', 'magnet', x - 6, y - 6);
    else if (it.kind === 'bomb') drawBomb(x, y);
  }
}
function drawBag(x, y) {
  cR('#1a1008', x - 4, y - 3, 9, 8); cR('#8a5a2a', x - 3, y - 2, 7, 6); cR('#c9922a', x - 2, y - 1, 3, 2);
  cR('#1a1008', x - 2, y - 5, 5, 2); cR('#ffd23f', x - 1, y, 3, 2);
}
function drawBomb(x, y) {
  cR('#1a1426', x - 4, y - 3, 9, 8); cR('#e8e4d0', x - 3, y - 2, 7, 6); cR('#1a1426', x - 2, y, 2, 2); cR('#1a1426', x + 1, y, 2, 2);
  cR('#8a5a2a', x, y - 6, 1, 3); cR(Math.floor(tt * 12) % 2 ? '#ffd23f' : '#ff7a1a', x, y - 7, 1, 1);
}

/* ================= level-up (GDD §4) =================
   3 cards: new weapons and passives while there are free slots, the next level of what the hero has (twice as likely).
   Nothing left to take → gold or a full heal */
function rollChoices() {
  const pool = [];
  for (const w of P.weapons) if (w.lv < WEAPON_MAX) pool.push({ kind: 'w', id: w.id, lv: w.lv + 1, wt: 2 });
  if (P.weapons.length < SLOTS) for (const id of WEAPON_IDS) if (!P.weapons.some(w => w.id === id)) pool.push({ kind: 'w', id, lv: 1, wt: 1 });
  for (const p of P.passives) if (p.lv < PASSIVE_MAX) pool.push({ kind: 'p', id: p.id, lv: p.lv + 1, wt: 2 });
  if (P.passives.length < SLOTS) for (const id of PASSIVE_IDS) if (!P.passives.some(p => p.id === id)) pool.push({ kind: 'p', id, lv: 1, wt: 1 });
  const out = [];
  while (out.length < 3 && pool.length) {
    const c = pickW(pool.map(c => [c, c.wt])); out.push(c); pool.splice(pool.indexOf(c), 1);
  }
  if (!out.length) out.push({ kind: 'gold', wt: 1 }, { kind: 'elixir', wt: 1 });
  return out;
}
function choiceText(c) {
  if (c.kind === 'w') return [WEAPONS[c.id].name, WEAPONS[c.id].desc[c.lv - 1]];
  if (c.kind === 'p') return [PASSIVES[c.id].name, PASSIVES[c.id].desc];
  if (c.kind === 'gold') return ['МЕШОК ЗОЛОТА', '+25 золота'];
  return ['ЭЛИКСИР', 'Полное лечение'];
}
function takeChoice(c) {
  if (c.kind === 'w') { const w = P.weapons.find(w => w.id === c.id); if (w) w.lv++; else addWeapon(c.id); }
  else if (c.kind === 'p') { const p = P.passives.find(p => p.id === c.id); if (p) p.lv++; else P.passives.push({ id: c.id, lv: 1 }); PASSIVES[c.id].fn(); }
  else if (c.kind === 'gold') G.gold += 25;
  else P.hp = P.maxhp;
  Snd.sfx('pick');
  G.pending--; G.lvl++;
  if (G.pending > 0) { choices = rollChoices(); sel = 0; stateT = 0; } else state = 'play';
}
