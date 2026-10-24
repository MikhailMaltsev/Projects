// Crypt Survival — пиксельные спрайты и тайлсеты (перенесено из pixel-crypt js/sprites.js без изменений)
'use strict';

/* ================= sprites ================= */
function build(rows, pal) {
  const w = Math.max(...rows.map(r => r.length)), h = rows.length;
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const g = c.getContext('2d');
  rows.forEach((row, y) => {
    if (row.length !== w) console.warn('sprite row length', y, row.length, w);
    for (let x = 0; x < w; x++) { const ch = row[x]; if (ch && ch !== '.' && pal[ch]) { g.fillStyle = pal[ch]; g.fillRect(x, y, 1, 1); } }
  });
  return c;
}
const flashCache = new Map();
function flashOf(img) {
  let f = flashCache.get(img);
  if (!f) {
    f = document.createElement('canvas'); f.width = img.width; f.height = img.height;
    const g = f.getContext('2d'); g.drawImage(img, 0, 0);
    g.globalCompositeOperation = 'source-in'; g.fillStyle = '#fff'; g.fillRect(0, 0, f.width, f.height);
    flashCache.set(img, f);
  }
  return f;
}

// the picture in one flat colour (the rogue's cloak: a dark body and a cold rim)
const tintCache = new Map();
function tintOf(img, col) {
  let m = tintCache.get(img); if (!m) { m = new Map(); tintCache.set(img, m); }
  let f = m.get(col);
  if (!f) {
    f = document.createElement('canvas'); f.width = img.width; f.height = img.height;
    const g = f.getContext('2d'); g.drawImage(img, 0, 0);
    g.globalCompositeOperation = 'source-in'; g.fillStyle = col; g.fillRect(0, 0, f.width, f.height);
    m.set(col, f);
  }
  return f;
}

const PAL_P = { k: '#1a1426', g: '#c3cad8', G: '#7d869c', r: '#e0453a', s: '#f0c090', b: '#3f72d8', B: '#2a4a94', y: '#e8b84a', w: '#ffffff' };
const PLAYER_A = [
  '.......rr.......',
  '.....kkrrkk.....',
  '....kggggggk....',
  '...kggggggggk...',
  '...kgkkkkkkgk...',
  '...kGggggggGk...',
  '....kkggggkk....',
  '..kggbbbbbbggk..',
  '..kgkbbbbbbkgk..',
  '..kskbbyybbksk..',
  '...kBbbbbbbBk...',
  '...kBBBBBBBBk...',
  '....kBBkkBBk....',
  '....kGGk.kGGk...',
  '....kGGk.kGGk...',
  '....kkkk.kkkk...'
];
const PLAYER_B = PLAYER_A.slice(0, 12).concat([
  '....kBBkkBBk....',
  '...kGGk...kGGk..',
  '...kGGk...kGGk..',
  '...kkkk...kkkk..'
]);
const S_PLAYER = [build(PLAYER_A, PAL_P), build(PLAYER_B, PAL_P)];

const PAL_A = { k: '#1a1426', h: '#2f7a40', H: '#5cb85c', s: '#f0c090', t: '#4a8a3a', T: '#2e5a2a', y: '#c9922a', b: '#8a5a2a', G: '#5a4a3a' };
const ARCHER_A = [
  '.......kk.......',
  '......khhk......',
  '.....khHhhk.....',
  '....khHhhhhk....',
  '...khhssssshk...',
  '...khsksskshk...',
  '...khhssssshk...',
  '....kkhhhhkk....',
  '..khttttttthhk..',
  '..khkttttttkhk..',
  '..kskttyyttksk..',
  '...kTttttttTk...',
  '...kTTTTTTTTk...',
  '....kGGkkGGk....',
  '....kbbk.kbbk...',
  '....kkkk.kkkk...'
];
const ARCHER_B = ARCHER_A.slice(0, 13).concat([
  '....kGGkkGGk....',
  '...kbbk...kbbk..',
  '...kkkk...kkkk..'
]);
const S_ARCHER = [build(ARCHER_A, PAL_A), build(ARCHER_B, PAL_A)];
const S_BOW = build([
  '.bb..',
  'w.bb.',
  'w..b.',
  'w...b',
  'w...b',
  'w...b',
  'w...b',
  'w..b.',
  'w.bb.',
  '.bb..'
], { b: '#a8743a', w: '#e8e4d0' });
// paladin as a Teutonic knight: flat-topped bucket helm with an eye slit crossed by a gold strip,
// mail arms, white surcoat with a black cross, white mantle hanging behind
const PALADIN_A = [
  '...kkkkkkkkkk...',
  '...kwggggggGk...',
  '...kgggyyggGk...',
  '...kkkkyykkkk...',
  '...kgggyyggGk...',
  '...kgggyyggGk...',
  '.kccGGGGGGGGcck.',
  'kcggwwwwwwwwggck',
  'kcgGwwwkkwwwGgck',
  'kcgGwkkkkkkwGgck',
  'kGGGwwwkkwwwGGGk',
  'kcckbbbyybbbkcck',
  'kcckWwwwwwwWkcck',
  'kccCkGGkkGGkCcck',
  '.kkkkggk.kggkkk.',
  '....kkkk.kkkk...'
];
const PALADIN_B = PALADIN_A.slice(0, 14).concat(['.kkkggk...kggkk.', '...kkkk...kkkk..']);
const PAL_PAL = { k: '#1a1426', g: '#d8dce8', G: '#8a92a8', y: '#f0c850', w: '#f4f0e4', W: '#bdb4a4', c: '#b4aa96', C: '#746c5e', b: '#5a3a22' };
// dwarf: short and wide, horned helm, orange beard down to the belt
const DWARF_A = [
  '................',
  '.k............k.',
  'khk..kkkkkk..khk',
  'khhkkggggggkkhhk',
  '.khhgggggggghhk.',
  '..kkGGGGGGGGkk..',
  '...kskssssksk...',
  '...kossssssok...',
  '..kooooOOooook..',
  '.krkooooooookrk.',
  '.kskoOooooOoksk.',
  '..krrkooookrrk..',
  '..kbbbbyybbbbk..',
  '..kRRRRkkRRRRk..',
  '..kBBBk..kBBBk..',
  '..kkkkk..kkkkk..'
];
const DWARF_B = DWARF_A.slice(0, 14).concat(['.kBBBk....kBBBk.', '.kkkkk....kkkkk.']);
const PAL_DWF = { k: '#1a1426', h: '#ece4d2', g: '#c3cad8', G: '#7d869c', s: '#f0c090', o: '#e8832a', O: '#a8501a', r: '#9a3a2a', R: '#5e4a3e', b: '#4a2e18', y: '#ffd23f', B: '#33241a' };
// wizard: crooked pointy hat with a star, white beard, blue robe
const WIZARD_A = [
  '..........k.....',
  '.........kbk....',
  '........kbbk....',
  '.......kbbbk....',
  '......kbbybk....',
  '.....kbbbbbbk...',
  '..kBBBBBBBBBBk..',
  '....kskssksk....',
  '....kwsssswk....',
  '...kbwwwwwwbk...',
  '..kbbkwwwwkbbk..',
  '..ksbbwwwwbbsk..',
  '...kbbbwwbbbk...',
  '...kbbybbbbBk...',
  '..kBbbbbbbbbBk..',
  '..kkkkkkkkkkkk..'
];
const WIZARD_B = WIZARD_A.slice(0, 14).concat(['..kbBbbbbbbBbk..', '...kkkkkkkkkkk..']);
const PAL_WIZ = { k: '#1a1426', b: '#4a5ae0', B: '#2c348a', y: '#ffd23f', w: '#ece4d2', s: '#f0c090' };
// necromancer: violet hooded robe, the face lost in the hood but for two glowing eyes, bone collar and belt
const NECRO_A = [
  '......kkkk......',
  '.....kppppk.....',
  '....kpPPPPpk....',
  '...kpPPPPPPpk...',
  '...kpPddddPpk...',
  '...kpdeddedpk...',
  '...kpddffddpk...',
  '....kpPddPpk....',
  '..kppPPwwPPppk..',
  '..kpkPPwwPPkpk..',
  '..khkPpwwpPkhk..',
  '...kpPPPPPPpk...',
  '...kpPPvvPPpk...',
  '..kpPppppppPpk..',
  '..kPPpPPPPpPPk..',
  '..kkk.kkkk.kkk..'
];
const NECRO_B = NECRO_A.slice(0, 14).concat(['..kPpPPPPPPpPk..', '...kk.kkkk.kk...']);
// rogue: dark slate hood and cloak, eyes over a red scarf, leather jerkin
const ROGUE_A = [
  '......kkkk......',
  '.....kcccck.....',
  '....kcCCCCck....',
  '...kcCCCCCCck...',
  '...kcsessesck...',
  '...kcrrrrrrck...',
  '....kcrRRrck....',
  '...kkcrrrrckk...',
  '..kccLLLLLLcck..',
  '..kckLlLLlLkck..',
  '..kskLLbbLLksk..',
  '...kcLLyyLLck...',
  '...kcLLLLLLck...',
  '....kGGkkGGk....',
  '....kBBk.kBBk...',
  '....kkkk.kkkk...'
];
const ROGUE_B = ROGUE_A.slice(0, 13).concat(['....kGGkkGGk....', '...kBBk...kBBk..', '...kkkk...kkkk..']);
const PAL_RG = { k: '#1a1426', c: '#2c3340', C: '#45505f', s: '#f0c090', e: '#2a1a10', r: '#a83246', R: '#d9506a', L: '#5a3a2a', l: '#7a5238', b: '#2a1a10', y: '#c9922a', G: '#2a2e38', B: '#3a2a20' };
const PAL_NEC = { k: '#1a1426', p: '#2e2048', P: '#4a3478', d: '#0c0814', e: '#c49bff', f: '#a898b8', w: '#e8e4d0', h: '#d8d0c8', v: '#9a6aff' };
// werewolf (see nightkin.js), man: shaggy hair, amber eyes, a snarl, bare chest; the arms are already grey fur
// with white claws; torn trousers. The wolf: pointed ears, yellow eyes, a long muzzle with fangs, fur all over.
// On his last heart the eyes turn red (the _R sets)
const PAL_WW = { k: '#1a1426', h: '#3a2a22', H: '#5a4232', s: '#e0a878', S: '#b07850', e: '#d8a030', f: '#5e5866', F: '#8e8898', w: '#f4f0e4', c: '#f4f0e4', p: '#4a4a62', P: '#33334a', b: '#6b4a2a' };
const PAL_WW_RED = { ...PAL_WW, e: '#ff3a3a' };
const WWH_A = [
  '....kkkkkkkk....',
  '...khHhhhhHhk...',
  '..khhhhhhhhhhk..',
  '..khsssssssshk..',
  '..khsesssseshk..',
  '..khsskwwksshk..',
  '...kksssssskk...',
  '..kfkSssssSkfk..',
  '.kffksssssskffk.',
  '.kfFkSssssSkFfk.',
  '.kffkssSSsskffk.',
  '.kfFkpbbbbpkFfk.',
  'kckckppppppkckck',
  '.c.ckpPppPpkc.c.',
  '....kpPk.kPpk...',
  '....kkkk.kkkk...'
];
const WWW_A = [
  '...kk......kk...',
  '..kfFk....kFfk..',
  '..kffFkkkkFffk..',
  '..kffffffffffk..',
  '..kffeffffeffk..',
  '..kfffFFFFfffk..',
  '...kfFFkkFFfk...',
  '...kFwFFFFwFk...',
  '..kfkkwkkwkkfk..',
  '.kffkfFFFFfkffk.',
  '.kfFkfFFFFfkFfk.',
  '.kffkpbbbbpkffk.',
  'kckckppppppkckck',
  '.c.ckpPppPpkc.c.',
  '....kfFk.kFfk...',
  '....kkkk.kkkk...'
];
const WWH_B = WWH_A.slice(0, 14).concat(['...kpPk...kPpk..', '...kkkk...kkkk..']);
const WWW_B = WWW_A.slice(0, 14).concat(['...kfFk...kFfk..', '...kkkk...kkkk..']);
const S_WOLF = [build(WWW_A, PAL_WW), build(WWW_B, PAL_WW)];
const S_WW_R = [build(WWH_A, PAL_WW_RED), build(WWH_B, PAL_WW_RED)], S_WOLF_R = [build(WWW_A, PAL_WW_RED), build(WWW_B, PAL_WW_RED)];
// during a claw blow the striking (front) arm is off the body: the hand is drawn sweeping instead (nightkin.js)
const armless = (rows, y0) => rows.map((r, y) => y >= y0 && y <= 13 ? r.slice(0, 12) + '....' : r);
const armSet = (A, B, y0, pal) => [build(armless(A, y0), pal), build(armless(B, y0), pal)];
const S_WW_SW = armSet(WWH_A, WWH_B, 7, PAL_WW), S_WW_SW_R = armSet(WWH_A, WWH_B, 7, PAL_WW_RED);
const S_WOLF_SW = armSet(WWW_A, WWW_B, 8, PAL_WW), S_WOLF_SW_R = armSet(WWW_A, WWW_B, 8, PAL_WW_RED);
// warlock: hood with two bone horns, the face lost in shadow but for red eyes, crimson robe with a gold sigil, grey hands
const WARLOCK_A = [
  '.k............k.',
  'kmk..kkkkkk..kmk',
  '.kmkkoOOOOokkmk.',
  '..kMooOOOOooMk..',
  '...kooddddook...',
  '...kodessedok...',
  '...kodssssdok...',
  '....koddddok....',
  '..kooRRyyRRook..',
  '..korRRYYRRrok..',
  '.kskrRRyyRRrksk.',
  '..kkrRRRRRRrkk..',
  '...krrRyyRrrk...',
  '...krRRRRRRrk...',
  '..krrRrrrrRrrk..',
  '..kkkkkkkkkkkk..'
];
const WARLOCK_B = WARLOCK_A.slice(0, 14).concat(['.krrRrrrrrrRrk..', '.kkkkkkkkkkkkk..']);
const PAL_WL = { k: '#1a1426', m: '#d8ccb8', M: '#8a7e6e', o: '#3a1220', O: '#5a1a2c', d: '#0c0608', e: '#ff4a3a', s: '#9a8c94', r: '#5a1424', R: '#8a2232', y: '#c9922a', Y: '#ffd23f' };
// vampire: slicked black hair with a widow's peak, pale face, red eyes and fangs, a high collar lined with red,
// white jabot with a gold brooch, black cape over a dark coat, pale clawed hands
const VAMP_A = [
  '.....kkkkkk.....',
  '....khhhhhhk....',
  '....khhsshhk....',
  '.k..khsssshk..k.',
  '.kr.ksessesk.rk.',
  '.kr.kssssssk.rk.',
  '.krk.kswwsk.krk.',
  '.krrkkwwwwkkrrk.',
  'kcrrkCwyywCkrrck',
  'kcrkCCwwwwCCkrck',
  'kcksCCCwwCCCskck',
  'kcknkCCCCCCknkck',
  '.kcckCCCCCCkcck.',
  '.kcckppppppkcck.',
  '.kckkpk..kpkkck.',
  '.kk.kkk..kkk.kk.'
];
const VAMP_B = VAMP_A.slice(0, 14).concat(['.kckpk....kpkck.', '.kkkkk....kkkkk.']);
const PAL_VP = { k: '#1a1426', h: '#231a30', s: '#d8d4e0', e: '#ff3a3a', w: '#ece8f0', c: '#2e2440', C: '#3e3252', r: '#9a1a30', y: '#c9922a', n: '#f4f0ff', p: '#2a2236' };
// the vampire's striking hand hidden under the cape while it rakes (nightkin.js)
const vampArmless = rows => rows.map((r, y) => y === 10 || y === 11 ? r.slice(0, 12) + 'c' + r.slice(13) : r);
const S_VAMP_SW = [build(vampArmless(VAMP_A), PAL_VP), build(vampArmless(VAMP_B), PAL_VP)];
const S_HERO = {
  knight: S_PLAYER, archer: S_ARCHER,
  paladin: [build(PALADIN_A, PAL_PAL), build(PALADIN_B, PAL_PAL)],
  dwarf: [build(DWARF_A, PAL_DWF), build(DWARF_B, PAL_DWF)],
  wizard: [build(WIZARD_A, PAL_WIZ), build(WIZARD_B, PAL_WIZ)],
  rogue: [build(ROGUE_A, PAL_RG), build(ROGUE_B, PAL_RG)],
  necro: [build(NECRO_A, PAL_NEC), build(NECRO_B, PAL_NEC)],
  werewolf: [build(WWH_A, PAL_WW), build(WWH_B, PAL_WW)],
  warlock: [build(WARLOCK_A, PAL_WL), build(WARLOCK_B, PAL_WL)],
  vampire: [build(VAMP_A, PAL_VP), build(VAMP_B, PAL_VP)]
};
// weapons held in the facing hand (the knight's sword is drawn with rects in render.js)
const S_SWORD = build(['.Gw.', '.Gw.', '.Gw.', '.Gw.', '.Gw.', '.Gw.', '.Gw.', 'yyyy'], { G: '#7d869c', w: '#e8eef8', y: '#e8b84a' });
// all-steel morning star: spiked ball, steel haft, leather grip
const S_MACE = build(['...w...', '.wkkkw.', '.kwgGk.', 'wkggGkw', '.kgGGk.', '.wkkkw.', '...G...', '...G...', '...G...', '...b...', '...G...'], { k: '#2a2a38', g: '#c3cad8', G: '#7d869c', w: '#ffffff', b: '#5a3a22' });
// icy-blue axe held on the slant: broad blade over the haft, tip of the haft poking out on top, white honed edge
const PAL_AXE = { D: '#5a9fd0', m: '#94cdee', l: '#cdeefb', w: '#ffffff', b: '#e0904a', B: '#b0682e' };
const S_AXE = outlined(['....DD.b', '...Dmmll', '....Dmlw', '....bllw', '...b.ww.', '..b.....', '.b......', 'B.......'], PAL_AXE);
const S_STAFF = build(['.kbk.', '..b..', '..b..', '..b..', '..b..', '..b..', '..b..', '..b..', '..b..', '..b..', '..k..'], { k: '#1a1426', b: '#8a5a2a' });
// the rogue's dagger
const S_DAGGER = build(['.w.', '.w.', 'Gw.', 'Gw.', 'yyy', '.b.'], { w: '#e8eef8', G: '#7d869c', y: '#c9922a', b: '#5a3a2a' });
// the hand that rakes with the claws (nightkin.js), claws up: the werewolf's grey paw, the vampire's pale hand
// with long nails out of a dark sleeve
const S_PAW = outlined(['w.w.w', 'w.w.w', 'fFfFf', 'fFFFf', '.fFf.', '.fff.'], { w: '#f4f0e4', f: '#5e5866', F: '#8e8898' });
const S_VHAND = outlined(['n.n.n', 'n.n.n', 'sSsSs', 'sssss', '.sss.', '.CCC.', '.CCC.'], { n: '#f4f0ff', s: '#d8d4e0', S: '#a8a0b8', C: '#3e3252' });
// the necromancer's staff: a small skull with violet eyes on top
const S_NSTAFF = build(['.www.', 'wewew', '.www.', '.w.w.', '..b..', '..b..', '..b..', '..b..', '..b..', '..b..', '..b..', '..b..', '..b..', '..k..'], { w: '#e8e4d0', e: '#c49bff', b: '#5a3a22', k: '#1a1426' });
// 10×10 ability icons for the HUD and the touch button
const S_ICON = {
  paladin: build(['.kkkkkkkk.', 'kyyyyyyyyk', 'kyyywwyyyk', 'kywwwwwwyk', 'kyyywwyyyk', 'kyyywwyyyk', '.kyywwyyk.', '.kyyyyyyk.', '..kyyyyk..', '...kkkk...'], { k: '#5a3a10', y: '#f0c850', w: '#fffbe0' }),
  dwarf: S_AXE,
  wizard: build(['....w.....', '.w..w..w..', '..w.w.w...', '...www....', 'wwwwWwwwww', '...www....', '..w.w.w...', '.w..w..w..', '....w.....', '..........'], { w: '#9fe8ff', W: '#ffffff' }),
  rogue: build(['...kkkk...', '..kcCCck..', '.kcCCCCck.', 'kcCkkkkCck', 'kckeddekck', 'kckddddkck', 'kcCkkkkCck', '.kcCCCCck.', '..kccccck.', '...kkkkk..'], { k: '#0c0a14', c: '#2c3340', C: '#5a6678', d: '#05040a', e: '#e8f4ff' }),
  necro: build(['..kkkkkk..', '.kwwwwwwk.', 'kwwwwwwwwk', 'kwddwwddwk', 'kwdewwdewk', 'kwwwkkwwwk', '.kwwwwwwk.', '..kwkkwk..', '...kkkk...', '..........'], { k: '#2a1a40', w: '#e8e4d0', d: '#1a1426', e: '#c49bff' }),
  // a wolf's head, a ring of fire, a drop of blood
  werewolf: build(['k........k', 'kfk....kfk', 'kffkkkkffk', 'kffffffffk', 'kfeffffefk', 'kffFFFFffk', '.kfFkkFfk.', '.kfwFFwfk.', '..kfFFfk..', '...kkkk...'],
    { k: '#140f1c', f: '#6e6878', F: '#a8a2b0', e: '#ff3a3a', w: '#ffffff' }),
  warlock: build(['...rRRr...', '..rYrrYr..', '.rY....Yr.', 'rY......Yr', 'rR......Rr', 'rR......Rr', 'rY......Yr', '.rY....Yr.', '..rYrrYr..', '...rRRr...'],
    { r: '#8a0a28', R: '#e02a4a', Y: '#ff9a3a' }),
  vampire: build(['....kk....', '...krrk...', '...krrk...', '..krrrrk..', '..krRrrk..', '.krRrrrrk.', '.krRrrrrk.', '.krrrrrrk.', '..krrrrk..', '...kkkk...'],
    { k: '#2a0a12', r: '#b0122e', R: '#ff6a7a' })
};
// HUD: the floor counter's icon, a stone crypt arch with a pale keystone and a dark doorway
const S_ARCH = build([
  '...kkkkk...',
  '.kkgggggkk.',
  'kgggwwwgggk',
  'kgggkkkgggk',
  'kggkdddkggk',
  'kGGkdddkGGk',
  'kggkdddkggk',
  'kggkdddkggk',
  'kGGkdddkGGk',
  'kkkkkkkkkkk'
], { k: '#1a1426', g: '#c9bfd9', G: '#6b5e86', w: '#ffe9a8', d: '#0a0710' });
// a locked hero's slot in the shop: a small gold padlock with a steel shackle
const S_LOCK = build([
  '..ggg..',
  '.g...g.',
  '.g...g.',
  '.g...g.',
  'yyyyyyy',
  'yYYkYYy',
  'yYYkYYy',
  'yYYYYYy',
  'yyyyyyy'
], { g: '#c3cad8', y: '#ffd23f', Y: '#c9922a', k: '#3a2a10' });
// gold shop button on the title: a chest with its lid ajar, light pouring out of the seam (glow drawn in drawTitle)
const S_SHOPCHEST = build([
  'g.....G.....g',
  '.g.kkkkkkk.g.',
  '..kbbbbbbbk..',
  '.GkYYYYYYYkG.',
  'GwwwwwwwwwwwG',
  'kwwLLLLLLLwwk',
  'kYYYYYYYYYYYk',
  'kbbbbkykbbbbk',
  'kBbbbkkkbbbBk',
  'kbbbbbbbbbbbk',
  'kYYYYYYYYYYYk',
  '.kkkkkkkkkkk.'
], { k: '#1a1008', b: '#8a5a2a', B: '#5a3a1a', Y: '#c9922a', w: '#fff5b8', L: '#ffffff', g: 'rgba(255,224,102,.45)', G: 'rgba(255,224,102,.8)' });

const S_SLIME = build([
  '................',
  '................',
  '................',
  '......kkkk......',
  '....kkGGGGkk....',
  '...kGGGGGGGGk...',
  '..kGGgGGGGGGGk..',
  '..kGgggGGGGGGk..',
  '.kGGGGGGGGGGGGk.',
  '.kGGwkGGGGwkGGk.',
  '.kGGGGGGGGGGGGk.',
  '.kGGGGGGkkGGGGk.',
  '..kGGGGGGGGGGk..',
  '...kkGGGGGGkk...',
  '.....kkkkkk.....',
  '................'
], { k: '#10281a', G: '#4cc36a', g: '#9af0a8', w: '#ffffff' });

const S_BAT = build([
  '................',
  '.kk..........kk.',
  '.kpk........kpk.',
  '.kppk..kk..kppk.',
  '.kpppkkppkkpppk.',
  'kpppppkrrkpppppk',
  'kpPpppkPPkpppPpk',
  '.kpppkkwwkkpppk.',
  '..kpk..kkkk..kpk',
  '...k....kk....k.',
  '................',
  '................',
  '................',
  '................',
  '................',
  '................'
], { k: '#1d1030', p: '#7b4fa8', P: '#a77bd6', r: '#ff5a4a', w: '#ffffff' });

const PAL_SK = { k: '#1a1426', w: '#e8e4d0', r: '#ff4a4a' };
const SKEL_A = [
  '.....kkkkkk.....',
  '....kwwwwwwk....',
  '...kwwwwwwwwk...',
  '...kwkkwwkkwk...',
  '...kwkrwwkrwk...',
  '....kwwwwwwk....',
  '.....kwkwkwk....',
  '......kkkk......',
  '....kwwwwwwk....',
  '...kwkwwwwkwk...',
  '...kwkkwwkkwk...',
  '...kwkwwwwkwk...',
  '....kwkkkkwk....',
  '....kwk..kwk....',
  '....kwk..kwk....',
  '...kkwk..kwkk...'
];
const SKEL_B = SKEL_A.slice(0, 13).concat([
  '...kwk....kwk...',
  '...kwk....kwk...',
  '..kkwk....kwkk..'
]);
const S_SKEL = [build(SKEL_A, PAL_SK), build(SKEL_B, PAL_SK)];
// the necromancer's raised skeleton: the same bones with violet eyes (it also glows violet, see necro.js)
const PAL_SKA = { k: '#1a1426', w: '#e8e4d0', r: '#c49bff' };
const S_SKEL_ALLY = [build(SKEL_A, PAL_SKA), build(SKEL_B, PAL_SKA)];
// HUD: a tiny skull in front of the skeleton's hearts (dim when there is no skeleton)
const SKULL_MINI = ['.kkkkk.', 'kwwwwwk', 'kwewewk', 'kwwwwwk', '.kwkwk.', '..kkk..'];
const S_SKULL_MINI = build(SKULL_MINI, { k: '#1a1426', w: '#e8e4d0', e: '#9a6aff' });
const S_SKULL_MINI_OFF = build(SKULL_MINI, { k: '#221a30', w: '#3c3350', e: '#221a30' });

const S_BOSS = build([
  '.h............h.',
  '.hh..........hh.',
  '..hh........hh..',
  '..khhkkkkkkhhk..',
  '...kRRRRRRRRk...',
  '..kRRRRRRRRRRk..',
  '..kRyykRRkyyRk..',
  '..kRRRRRRRRRRk..',
  '..kRRkwkwkwRRk..',
  '...kRRRkkkRRk...',
  '..kDDDRRRRDDDk..',
  '.kDDDDDDDDDDDDk.',
  '.kDDkDDDDDDkDDk.',
  '.kDDkDDDDDDkDDk.',
  '..kkkDDkkDDkkk..',
  '..kkk.kk.kk.kkk.'
], { k: '#12081a', R: '#b3202a', y: '#ffd23f', w: '#f4ead0', h: '#e8dcc0', D: '#5c2d63' });

const HEART_ROWS = [
  '.kk..kk.',
  'kRRkkRRk',
  'kRwRRRRk',
  'kRRRRRRk',
  '.kRRRRk.',
  '..kRRk..',
  '...kk...'
];
// q = filled quarters (0..4), filled from the left
// R, w = the fill and its glint (the necromancer's skeleton has violet hearts)
function heartSprite(q, R = '#e0453a', w = '#ff9a8a') {
  const c = document.createElement('canvas'); c.width = 8; c.height = 7;
  const g = c.getContext('2d');
  HEART_ROWS.forEach((row, y) => {
    for (let x = 0; x < 8; x++) {
      const ch = row[x]; if (ch === '.') continue;
      const on = x < q * 2;
      let col = '#1a1426';
      if (ch === 'R') col = on ? R : '#3c2a4e';
      else if (ch === 'w') col = on ? w : '#3c2a4e';
      else if (ch === 'k') col = q === 0 ? '#2c2040' : '#1a1426';
      g.fillStyle = col; g.fillRect(x, y, 1, 1);
    }
  });
  return c;
}
const S_HEART = [0, 1, 2, 3, 4].map(q => heartSprite(q));
const S_HEART_SK = [0, 1, 2, 3, 4].map(q => heartSprite(q, '#9a6aff', '#d8c8ff'));
const S_HEART_ITEM = build(HEART_ROWS, { k: '#1a1426', R: '#e0453a', w: '#ff9a8a' });
const S_POTION = build([
  '..kkkk..',
  '..kcCk..',
  '..kggk..',
  '.kgwggk.',
  'kgwrRRRk',
  'kwrRRRDk',
  'kRRRRRDk',
  'kRRRRDDk',
  '.kDDDDk.',
  '..kkkk..'
], { k: '#1a1426', c: '#b07a40', C: '#7a5028', g: '#8fa8c8', w: '#e8f4ff', r: '#ff8aa8', R: '#e0365e', D: '#8e1a44' });
const S_POTION_EMPTY = build([
  '..kkkk..',
  '..k..k..',
  '..k..k..',
  '.k....k.',
  'k......k',
  'k......k',
  'k......k',
  'k......k',
  '.k....k.',
  '..kkkk..'
], { k: '#3c2a4e' });

const PAL_GH = { k: '#2a3a5a', w: '#d6ecff', W: '#8fb0d8', b: '#7fe3ff' };
const GHOST_TOP = [
  '......kkkk......',
  '....kkwwwwkk....',
  '...kwwwwwwwwk...',
  '..kwwwwwwwwwwk..',
  '..kwkkwwwwkkwk..',
  '..kwkbwwwwkbwk..',
  '..kwwwwwwwwwwk..',
  '..kwwwwkkwwwwk..',
  '..kwwwkkkkwwwk..',
  '..kwwwwkkwwwwk..',
  '..kWwwwwwwwwwk..',
  '..kWWwwwwwwWWk..',
  '..kWWWWWWWWWWk..'
];
const S_GHOST = [
  build(GHOST_TOP.concat(['..kWWkWWWWkWWk..', '..kWk.kWWk.kWk..', '...k...kk...k...']), PAL_GH),
  build(GHOST_TOP.concat(['..kWWWkWWWWkWk..', '...kWk.kWWk.kk..', '....k...kk......']), PAL_GH)
];

const PAL_SP = { k: '#120c18', b: '#3a2a4a', B: '#6a5088', l: '#7a6494', r: '#ff3a3a' };
const S_SPIDER = [
  build(['................', '................', '................', '................', '................', '................',
    '.l....kkkk....l.', 'l.l..kBBBBk..l.l', 'l..llkBbbBkll..l', '.ll.kbbbbbbk.ll.', 'l..llkrbbrkll..l', '.ll.lkkbbkkl.ll.', 'l..l..kkkk..l..l',
    '................', '................', '................'], PAL_SP),
  build(['................', '................', '................', '................', '................', '................',
    '..l...kkkk...l..', '.l.l.kBBBBk.l.l.', '.l.llkBbbBkll.l.', 'l.ll.kbbbbk.ll.l', '.l.llkrbbrkll.l.', 'l.ll.lkbbkl.ll.l', '..l...kkkk...l..',
    '................', '................', '................'], PAL_SP)
];

const PAL_CU = { k: '#14081c', h: '#7a1e3a', H: '#b8405e', d: '#3d0e22', e: '#ffcc33', F: '#ffe066', f: '#ff7a1a', s: '#e0a080' };
const CULT_A = [
  '......kkkk......',
  '.....khhhhk.....',
  '....khHhhhhk....',
  '...khHhhhhhhk...',
  '...khhkkkkhhk...',
  '...khkekkekhk...',
  '...khkkkkkkhk...',
  '....khkkkkhk....',
  '...khhhhhhhhk...',
  '..khhhhhhhhhhk..',
  '.kFkdhhhhhhdkhk.',
  'kFfFkdhhhhdkshk.',
  '.kFkkdhhhhhdkhk.',
  '..k.kddhhhhddk..',
  '....kdddddddk...',
  '....kkkkkkkkk...'
];
const CULT_B = CULT_A.slice(0, 10).concat([
  '.kfkdhhhhhhdkhk.',
  'kfFfkdhhhhdkshk.',
  '.kfkkdhhhhhdkhk.'
]).concat(CULT_A.slice(13));
const S_CULT = [build(CULT_A, PAL_CU), build(CULT_B, PAL_CU)];

const S_LICH = build([
  '...y...yy...y...',
  '..kyk.kyyk.kyk..',
  '..kyyyyrryyyyk..',
  '..kwwwwwwwwwwk..',
  '..kwkkwwwwkkwk..',
  '..kwkgwwwwkgwk..',
  '..kWwwwkkwwwWk..',
  '...kwkwkwkwwk...',
  '..PPkkkkkkkkPP..',
  '.kPPPpppppppPPk.',
  'kPPppppggppppPPk',
  'kPpppcpttpcpppPk',
  'kPppcpppppppcpPk',
  '.kppcpppppppcpk.',
  '..kcccpppppcck..',
  '..kk.kkk.kkk.kk.'
], { k: '#0a0612', y: '#ffd23f', r: '#5dff9a', w: '#e8e4d0', W: '#a8a290', g: '#5dff9a', p: '#3a2a6a', P: '#6a54b8', c: '#1e1640', t: '#2ab868' });
const S_COIN = build([
  '..kkkk..',
  '.kyyyyk.',
  'kyywyyYk',
  'kyywyyYk',
  'kyyyyyYk',
  'kyyyyYYk',
  '.kYYYYk.',
  '..kkkk..'
], { k: '#3a2a10', y: '#ffd23f', Y: '#c9922a', w: '#fff5b8' });

/* ================= hero training icons (shop) ================= */
// drawn without an outline: outlined() pads the picture by 1 px and rings every pixel with a dark one
function outlined(rows, pal) {
  const w = rows[0].length + 2, pad = '.'.repeat(w), grid = [pad, ...rows.map(r => '.' + r + '.'), pad];
  const on = (x, y) => y >= 0 && y < grid.length && x >= 0 && x < w && grid[y][x] !== '.';
  return build(grid.map((r, y) => [...r].map((ch, x) => ch !== '.' ? ch : on(x - 1, y) || on(x + 1, y) || on(x, y - 1) || on(x, y + 1) ? 'k' : '.').join('')), { k: '#1a1426', ...pal });
}
// the hero card's training button: a gold arrow in a gold frame
const S_UP = build([
  '.kkkkkkkkkkkk.',
  'kFFFFFFFFFFFFk',
  'kFddddddddddfk',
  'kFddddwyddddfk',
  'kFdddwwyYdddfk',
  'kFddwwyyYYddfk',
  'kFdwwyyyyYYdfk',
  'kFddkwyyYkddfk',
  'kFdddwyyYdddfk',
  'kFdddwyyYdddfk',
  'kFdddwyyYdddfk',
  'kFddddddddddfk',
  'kffffffffffffk',
  '.kkkkkkkkkkkk.'
], { k: '#1a1426', F: '#ffd23f', f: '#a8741a', d: '#2a1d10', w: '#fff5b8', y: '#ffd23f', Y: '#c9922a' });
const PAL_TR = { w: '#ffffff', g: '#c3cad8', G: '#7d869c', y: '#ffd23f', Y: '#c9922a', b: '#a8743a', B: '#6b4420', r: '#e0453a', R: '#ff9a8a', o: '#ff7a1a', c: '#7fe3ff', C: '#2f8fd8' };
// per hero: [damage, hearts, range]
const TR_HEART = outlined([
  '.rr...rr.',
  'rRrr.rrrr',
  'rRrrrrrrr',
  'rrrrrrrrr',
  '.rrrrrrr.',
  '..rrrrr..',
  '...rrr...',
  '....r....'
], PAL_TR);
const TR_ARC = outlined([
  '.wwwww...',
  'wccccwww.',
  '....cccw.',
  '.....ccww',
  '......ccw',
  '.......cw',
  '.......cw',
  '.......cw',
  '.......w.'
], PAL_TR);
// claws (the werewolf's white, the vampire's red), the warlock's fireball and its flight
const TR_CLAW = outlined(['w..w..w..', '.g..g..g.', '.w..w..w.', '..g..g..g', '..w..w..w', '...G..G..'], PAL_TR);
const TR_CLAW_R = outlined(['R..R..R..', '.r..r..r.', '.R..R..R.', '..r..r..r', '..R..R..R', '...r..r..'], PAL_TR);
const PAL_HELL = { ...PAL_TR, r: '#8a0a28', R: '#e02a4a', o: '#ff9a3a', y: '#fff0c0' };
const TR_HELL = outlined(['...rrr...', '..rRRRr..', '.rRRoRRr.', '.rRoyoRr.', '.rRRoRRr.', '..rRRRr..', '...rrr...'], PAL_HELL);
const TR_COMET = outlined(['.......rRr', '......rRoR', '.....rRoyo', '...r.rRoRr', '..r.r.rRr.', '.r.r......', 'r.r.......', '.r........'], PAL_HELL);
// speed and armor: the same for every hero (a leather boot, a steel cuirass), added to each hero's three below
const TR_BOOT = outlined([
  'yyyy....',
  'YYYY....',
  'bbbB....',
  'bbbB....',
  'bbbB....',
  'bbbbB...',
  'bbbbbbB.',
  'Bbbbbbbb',
  'GGGGGGGG'
], PAL_TR);
const TR_CUIRASS = outlined([
  '.ggG...gGG.',
  'gwgGgggGgGG',
  'gwggggggGGG',
  '.gwggyggGG.',
  '.gwgyYygGG.',
  '.gwggyggGG.',
  '.gwgggggGG.',
  '..gwgggGG..',
  '...gGGGG...'
], PAL_TR);
// per hero: [damage, hearts, range], then TR_BOOT and TR_CUIRASS
const S_TRAIN = {
  knight: [outlined([
    '.........w',
    '........wG',
    '.......wG.',
    '.y....wG..',
    '..y..wG...',
    '...ywG....',
    '...by.....',
    '..b..y....',
    '.b....y...',
    'Y.........'
  ], PAL_TR), TR_HEART, TR_ARC],
  archer: [outlined([
    '.Bb....',
    'w.Bb...',
    'w..Bb..',
    'w...Bb.',
    'w...Bb.',
    'w...Yy.',
    'w...Yy.',
    'w...Bb.',
    'w...Bb.',
    'w..Bb..',
    'w.Bb...',
    '.Bb....'
  ], PAL_TR), TR_HEART, outlined([
    '.........gw',
    '........gwg',
    '........Gg.',
    '.......B...',
    '......B....',
    '.....b.....',
    '....b......',
    '...b.......',
    '.pp........',
    'pPp........',
    '.p.........'
  ], { ...PAL_TR, w: '#e4e8f4', g: '#b9bdd8', G: '#9a9ec0', b: '#d9925e', B: '#b0623e', p: '#f8c99c', P: '#d98e62' })],
  paladin: [outlined([
    '.......w....',
    '....w..w..w.',
    '.....wgggw..',
    '.....gwggG..',
    '...wwgggGGww',
    '.....ggGGG..',
    '.....GGGGw..',
    '....G..w..w.',
    '...G...w....',
    '..b.........',
    '.b..........',
    'B...........'
  ], PAL_TR), TR_HEART, TR_ARC],
  dwarf: [outlined([
    '......DDD...',
    '.....DDmD.b.',
    '.....Dmmmlll',
    '......Dmllll',
    '......Dmlllw',
    '......bmllww',
    '.....b..lww.',
    '....b.......',
    '...b........',
    '..b.........',
    '.b..........',
    'B...........'
  ], { ...PAL_TR, ...PAL_AXE }), TR_HEART, TR_ARC],
  wizard: [outlined([
    '...cccc...',
    '..cwwccc..',
    'y.cwcccC.y',
    'y.ccccCC.y',
    'Yy.cCCC.yY',
    '.Yy....yY.',
    '..YyyyyY..',
    '....bB....',
    '....bB....',
    '....bB....',
    '....bB....'
  ], PAL_TR), TR_HEART, outlined([
    '.....ooo..',
    '....oyyyo.',
    '...oywwyyo',
    '..ooywyyyo',
    '..ooyyyyyo',
    '.o.ooyyyo.',
    '..o.oooo..',
    '.o.o.o....',
    'o..o......'
  ], PAL_TR)],
  rogue: [outlined([
    '........w',
    '.......wG',
    '......wG.',
    '.....wG..',
    '....wG...',
    '.y.wG....',
    '..yy.....',
    '..by.....',
    '.b..y....',
    'B........'
  ], PAL_TR), TR_HEART, TR_ARC],
  necro: [outlined([
    '...vVVv...',
    '..vwwwwv..',
    '..wewwew..',
    '..wwwwww..',
    '...wkkw...',
    '...wwww...',
    '....bB....',
    '....bB....',
    '....bB....',
    '....bB....',
    '....bB....'
  ], { ...PAL_TR, v: '#9a6aff', V: '#e8dcff', e: '#c49bff', w: '#e8e4d0', k: '#1a1426' }), TR_HEART, outlined([
    '......vVVv.',
    '.....vVwwVv',
    '.....vVwwVv',
    '......vVVv.',
    '....v.vv...',
    '...v.v.....',
    '..v.v......',
    '.v.........',
    'v..........'
  ], { ...PAL_TR, v: '#9a6aff', V: '#e8dcff' })],
  werewolf: [TR_CLAW, TR_HEART, TR_ARC],
  warlock: [TR_HELL, TR_HEART, TR_COMET],
  vampire: [TR_CLAW_R, TR_HEART, TR_ARC]
};
for (const id in S_TRAIN) S_TRAIN[id].push(TR_BOOT, TR_CUIRASS);

/* ================= tiles ================= */
const WALL_CRACK = [[1, 2], [2, 3], [2, 4], [3, 5], [3, 6], [9, 9], [10, 10], [10, 11], [11, 12], [11, 13], [14, 3], [14, 4], [13, 5]];
function tile(fn) { const c = document.createElement('canvas'); c.width = c.height = T; fn(c.getContext('2d')); return c; }
function tileset(pal, seed, moss, ash) {
  const rngT = mulberry(seed);
  const floor = [0, 1, 2, 3].map(v => tile(g => {
    g.fillStyle = pal.f; g.fillRect(0, 0, T, T);
    g.fillStyle = pal.fl; g.fillRect(0, 0, T, 1); g.fillRect(0, 0, 1, T);
    g.fillStyle = pal.fs1; for (let i = 0; i < 9; i++) g.fillRect(rngT() * T | 0, rngT() * T | 0, 1, 1);
    g.fillStyle = pal.fs2; for (let i = 0; i < 5; i++) g.fillRect(rngT() * T | 0, rngT() * T | 0, 1, 1);
    if (v === 1) { g.fillStyle = pal.fs2; g.fillRect(3, 9, 5, 1); g.fillRect(7, 10, 3, 1); g.fillRect(9, 11, 2, 1); }
    if (v === 2) { g.fillStyle = pal.fs3; g.fillRect(8, 3, 2, 1); g.fillRect(9, 4, 1, 1); }
    if (v === 3 && ash) { g.fillStyle = '#3a2e30'; g.fillRect(4, 6, 2, 1); g.fillRect(11, 12, 1, 1); g.fillStyle = '#5a2a1a'; g.fillRect(5, 6, 1, 1); }
  }));
  const face = tile(g => {
    g.fillStyle = pal.w; g.fillRect(0, 0, T, T);
    g.fillStyle = pal.wm; g.fillRect(0, 7, T, 1); g.fillRect(0, 15, T, 1); g.fillRect(4, 0, 1, 7); g.fillRect(12, 8, 1, 7);
    g.fillStyle = pal.wh; g.fillRect(0, 0, T, 1); g.fillRect(5, 8, 7, 1);
    g.fillStyle = pal.ws; for (let i = 0; i < 14; i++) g.fillRect(rngT() * T | 0, rngT() * T | 0, 1, 1);
    if (moss) { g.fillStyle = '#4a7a3a'; g.fillRect(1, 6, 3, 1); g.fillRect(2, 5, 1, 1); g.fillRect(13, 14, 2, 1); g.fillRect(8, 6, 2, 1); g.fillStyle = '#6aa04a'; g.fillRect(2, 6, 1, 1); g.fillRect(9, 6, 1, 1); }
    if (ash) { g.fillStyle = '#1a0c0a'; WALL_CRACK.forEach(([x, y]) => g.fillRect(x, y, 1, 1)); }
    g.fillStyle = pal.wb; g.fillRect(0, 14, T, 1);
  });
  const top = tile(g => {
    g.fillStyle = pal.t; g.fillRect(0, 0, T, T);
    g.fillStyle = pal.t2; for (let i = 0; i < 10; i++) g.fillRect(rngT() * T | 0, rngT() * T | 0, 2, 1);
  });
  return { floor, face, top, pal }; // pal: the walls' stone also colours the way down (drawGrateExit)
}
const TS_OLD = tileset({ f: '#2b2539', fl: '#241f31', fs1: '#332c45', fs2: '#211c2d', fs3: '#383250', w: '#54476b', wm: '#35294a', wh: '#6b5e86', ws: '#463a5c', wb: '#2a2038', t: '#16111f', t2: '#1d1729' }, 1337, false);
const TS_NEW = tileset({ f: '#1f2a2c', fl: '#192224', fs1: '#283638', fs2: '#162022', fs3: '#2e4042', w: '#3d5856', wm: '#22383a', wh: '#557a74', ws: '#33504c', wb: '#1a2a2a', t: '#0b1213', t2: '#111c1d' }, 4242, true);
// ash zone (11-15): obsidian walls with smouldering cracks (the glow itself is drawn per frame in drawTiles)
const TS_ASH = tileset({ f: '#241b1f', fl: '#1d161a', fs1: '#2e2328', fs2: '#1a1317', fs3: '#3a2a2a', w: '#4a3532', wm: '#2c1d1c', wh: '#684840', ws: '#3c2a28', wb: '#201414', t: '#110b0d', t2: '#1a1013' }, 777, false, true);
let TS = TS_OLD;

/* ================= ash zone (floors 11-15) ================= */
const mirror = rows => rows.map(r => r + [...r].reverse().join(''));
const S_MAGMA = build([
  '................',
  '................',
  '................',
  '......kkkk......',
  '....kkGGGGkk....',
  '...kGGoGGGGGk...',
  '..kGGooGGGGGGk..',
  '..kGoyoGGGoGGk..',
  '.kGGGoGGGGooGGk.',
  '.kGGwkGGGGwkGGk.',
  '.kGGGGoGGGGGGGk.',
  '.kGGGGooGkoGGGk.',
  '..kGGGGoGGGGGk..',
  '...kkGGGGGGkk...',
  '.....kkkkkk.....',
  '................'
], { k: '#1a0604', G: '#7a3018', o: '#ff8a2a', y: '#ffe066', w: '#ffffff' });
// salamander (the ash lizard): a crust of cooled lava over a body of fire, a ridge of spines, a wide jaw; side view,
// facing right, the frames differ in the legs. S_SAL_HOT is the same with the cracks white-hot (laid over it to pulse)
const SAL_TOP = ['...................kkk..', '..................kCCCk.', '..........k.k.k..kcCCCCk', '.........kCkCkCk.kccceCk', '........kccMcccMcckcccck', '.....kkkccmcccmcccckrrrk', '..kkkccMcccmcccMccckttk.', '.kmccccccccccccccccrrk..', 'kmk.kccmcccccmcccckk....', 'kk...kcccccccccccck.....'];
const SAL_LEGS = [['.....kCk.....kCk........', '....kCk.....kCk.........', '....kk......kk..........'], ['......kCk.....kCk.......', '.......kCk.....kCk......', '.......kk......kk.......']];
const PAL_SAL = { k: '#0a0404', c: '#2a2224', C: '#45393a', m: '#7a2410', M: '#c8401a', r: '#3a0806', t: '#d8c8a8', e: '#ffb030' };
const PAL_SAL_HOT = Object.assign({}, PAL_SAL, { m: '#ff5a1a', M: '#ffd23f', e: '#fff3b0', r: '#ff7a1a' });
const S_SAL = SAL_LEGS.map(l => build([...SAL_TOP, ...l], PAL_SAL)), S_SAL_HOT = SAL_LEGS.map(l => build([...SAL_TOP, ...l], PAL_SAL_HOT));
const S_SKULL = build([
  '...kkkkkk...',
  '..kwwwwwwk..',
  '.kwwwwwwwwk.',
  'kwwwwwwwwwwk',
  'kwkkwwwwkkwk',
  'kwkekwwkekwk',
  'kWwkkwwkkwWk',
  '.kWwwkkwwWk.',
  '..kwkwkwkk..',
  '...kkkkkk...'
], { k: '#2a0c06', w: '#f4ead0', W: '#b8ac90', e: '#ffe066' });
const PAL_GU = { k: '#1a1426', w: '#d8ccb8', r: '#ff7a1a', H: '#5a5060', h: '#8a8098', a: '#4a4250', A: '#6a6070' };
const GUARD_TOP = [
  '....kkkkkkkk....',
  '...kHHHHHHHHk...',
  '...kHhHHHHHHk...',
  '...kkkkkkkkkk...',
  '...kwkrwwkrwk...',
  '....kwwwwwwk....',
  '.....kwkwkwk....',
  '......kkkk......',
  '....kaaaaaak....',
  '...kwkaAAakwk...',
  '...kwkaaaakwk...',
  '...kwkaAAakwk...',
  '....kkkkkkkk....'
];
const S_GUARD = [
  build(GUARD_TOP.concat(['....kwk..kwk....', '....kwk..kwk....', '...kkwk..kwkk...']), PAL_GU),
  build(GUARD_TOP.concat(['...kwk....kwk...', '...kwk....kwk...', '..kkwk....kwkk..']), PAL_GU)
];
const S_SHIELD = build([
  '.kkkkk.',
  'kSSSSSk',
  'kSsesSk',
  'kSeeeSk',
  'kSsesSk',
  'kSsesSk',
  'kSsssSk',
  'kSsssSk',
  'kSsssSk',
  '.kSsSk.',
  '..kSk..',
  '...k...'
], { k: '#1a1426', S: '#8a8098', s: '#5a5060', e: '#ff7a1a' });
const S_WORM = build(mirror([
  '.....kkk',
  '...kkBBB',
  '..kBBbBB',
  '.kBBktkt',
  '.kBktooo',
  '.kBkoRRR',
  '.kBktooo',
  '.kBBktkt',
  '..kBBBBB',
  '..kbbbbb',
  '..kBBBBB',
  '..kbbbbb',
  '..kBBBBB',
  '..kbbbbb',
  '..kBBBBB',
  '..kbbbbb'
]), { k: '#1a0c0c', B: '#8a6a52', b: '#6a4a3a', t: '#e8dcc0', o: '#ff7a1a', R: '#2a0606' });
const PAL_CH = { k: '#1a1008', b: '#8a5a2a', B: '#5a3a1a', Y: '#c9922a', y: '#ffd23f', t: '#f4ead0', R: '#4a0a16', r: '#e0365e', e: '#ffe066' };
const CHEST_ROWS = [
  '..kkkkkkkkkk..',
  '.kbbbbbbbbbbk.',
  'kbBbbbbbbbbBbk',
  'kYYYYYYYYYYYYk',
  'kbbbbkyykbbbbk',
  'kBbbbkYYkbbbBk',
  'kbbbbbbbbbbbbk',
  'kBbbbbbbbbbbBk',
  'kYYYYYYYYYYYYk',
  '.kkkkkkkkkkkk.'
];
const S_CHEST = build(CHEST_ROWS, PAL_CH);
const S_MIMIC = build([
  '..kkkkkkkkkk..',
  '.kbbebbbbebbk.',
  'kbBbbbbbbbbBbk',
  'kYYYYYYYYYYYYk',
  'kttttttttttttk',
  'kRtRRtRRtRRtRk',
  'kRRRRrrrrRRRRk',
  'kRRRrrrrrrRRRk',
  'kRtRRtRRtRRtRk',
  'kttttttttttttk',
  'kYYYYYYYYYYYYk',
  'kbbbbkyykbbbbk',
  'kBbbbbbbbbbbBk',
  '.kkkkkkkkkkkk.'
], PAL_CH);
const S_LORD = build([
  '.h............h.',
  '.hH..........Hh.',
  '..hH.kkkkkk.Hh..',
  '...hkrrrrrrkh...',
  '...krRRRRRRrk...',
  '..krRyyRRyyRrk..',
  '..krRRRRRRRRrk..',
  '..krRkoooookrk..',
  '.kAAkrrRRrrkAAk.',
  'kAAAkRoRRoRkAAAk',
  'kAaAkRRooRRkAaAk',
  'kAaAkaaaaaakAaAk',
  '.kkkaaAaaAaakkk.',
  '...kaaAaaAaak...',
  '...kRRk..kRRk...',
  '...kkkk..kkkk...'
], { k: '#12060a', h: '#2e2228', H: '#5a4450', r: '#5a1a14', R: '#8a2a1a', y: '#ffd23f', o: '#ff7a1a', A: '#4a4250', a: '#2e2830' });
