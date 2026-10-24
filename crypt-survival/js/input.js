// Crypt Survival — клавиатура и сенсорный джойстик
'use strict';

/* ================= input =================
   WASD / arrows to walk; on a touch screen the stick appears where the finger lands on the field.
   Screens with choices: arrows + Enter / Space, the digits 1–3, or a tap on the card */
const keys = new Set();
const inp = { mx: 0, my: 0, sx: 0, sy: 0 };
let stickOn = null; // { id, x0, y0, x, y } in canvas px
const STICK_R = 18;
addEventListener('keydown', e => {
  if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(e.code)) e.preventDefault();
  if (e.repeat) { keys.add(e.code); return; }
  keys.add(e.code); Snd.init();
  onKey(e.code);
});
addEventListener('keyup', e => keys.delete(e.code));
addEventListener('blur', () => { keys.clear(); if (state === 'play') state = 'pause'; });
function readInput() {
  let mx = 0, my = 0;
  if (keys.has('KeyA') || keys.has('ArrowLeft')) mx -= 1;
  if (keys.has('KeyD') || keys.has('ArrowRight')) mx += 1;
  if (keys.has('KeyW') || keys.has('ArrowUp')) my -= 1;
  if (keys.has('KeyS') || keys.has('ArrowDown')) my += 1;
  inp.mx = clamp(mx + inp.sx, -1, 1); inp.my = clamp(my + inp.sy, -1, 1);
}
const toCanvas = e => { const r = cv.getBoundingClientRect(); return { x: (e.clientX - r.left) / r.width * W, y: (e.clientY - r.top) / r.height * H }; };
cv.addEventListener('pointerdown', e => {
  e.preventDefault(); Snd.init();
  const p = toCanvas(e);
  if (state === 'play' && hitBox(PAUSE_BTN, p.x, p.y, 6)) { state = 'pause'; return; }
  if (state === 'play' && !stickOn) { stickOn = { id: e.pointerId, x0: p.x, y0: p.y, x: p.x, y: p.y }; cv.setPointerCapture(e.pointerId); return; }
  onTap(p.x, p.y);
});
cv.addEventListener('pointermove', e => {
  if (!stickOn || e.pointerId !== stickOn.id) return;
  const p = toCanvas(e); stickOn.x = p.x; stickOn.y = p.y;
  const dx = p.x - stickOn.x0, dy = p.y - stickOn.y0, d = Math.hypot(dx, dy);
  if (d > STICK_R) { stickOn.x0 = p.x - dx / d * STICK_R; stickOn.y0 = p.y - dy / d * STICK_R; } // the stick follows a finger gone too far
  const k = Math.min(1, d / STICK_R) / (d || 1);
  inp.sx = d > 2 ? dx * k : 0; inp.sy = d > 2 ? dy * k : 0;
});
const stickOff = e => { if (stickOn && e.pointerId === stickOn.id) { stickOn = null; inp.sx = inp.sy = 0; } };
cv.addEventListener('pointerup', stickOff); cv.addEventListener('pointercancel', stickOff);
function drawStick() {
  if (!stickOn) return;
  ctx.globalAlpha = 0.35; ctx.strokeStyle = '#ece4d2'; ctx.beginPath(); ctx.arc(stickOn.x0, stickOn.y0, STICK_R, 0, 6.283); ctx.stroke();
  ctx.fillStyle = '#f0a93a'; ctx.beginPath(); ctx.arc(stickOn.x0 + inp.sx * STICK_R, stickOn.y0 + inp.sy * STICK_R, 6, 0, 6.283); ctx.fill();
  ctx.globalAlpha = 1;
}
