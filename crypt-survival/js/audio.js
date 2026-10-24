// Crypt Survival — звуки (WebAudio) и музыка (перенесено из pixel-crypt js/audio.js без изменений)
'use strict';

/* ================= audio ================= */
// a pack of enemies dying at once must not be a pack of sounds stacked into one loud one: a sound plays at most once per SFX_GAP s
const SFX_GAP = 0.05;
const Snd = {
  ctx: null, master: null, mus: null, nbuf: null, on: true, last: {},
  init() {
    if (this.ctx) { if (this.on && !this.held && this.ctx.state === 'suspended') this.ctx.resume(); return; }
    try {
      const AC = window.AudioContext || window.webkitAudioContext; if (!AC) return;
      this.ctx = new AC();
      this.master = this.ctx.createGain(); this.master.gain.value = this.on ? 0.55 : 0; this.master.connect(this.ctx.destination);
      this.mus = this.ctx.createGain(); this.mus.gain.value = 0.5; this.mus.connect(this.master);
      const len = Math.floor(this.ctx.sampleRate * 0.5), b = this.ctx.createBuffer(1, len, this.ctx.sampleRate), d = b.getChannelData(0);
      for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
      this.nbuf = b;
      if (!this.on) this.ctx.suspend();
    } catch (e) { this.ctx = null; }
  },
  // sound off = silence everything, music included: zero the master and suspend the whole audio engine
  // (some mobile browsers ignore a gain change on the fly, a suspended context is silent everywhere)
  setOn(v) {
    this.on = v;
    if (this.master) this.master.gain.value = v ? 0.55 : 0;
    if (this.ctx) { try { if (v && !this.held) this.ctx.resume(); else this.ctx.suspend(); } catch (e) { /* ignore */ } }
  },
  // an ad on screen: the whole engine waits, whatever the sound switch says
  held: false,
  hold(v) {
    this.held = v;
    if (this.ctx) { try { if (v || !this.on) this.ctx.suspend(); else this.ctx.resume(); } catch (e) { /* ignore */ } }
  },
  tone(f, dur, type = 'square', vol = 0.15, f2 = null, delay = 0, dest = null) {
    const c = this.ctx; if (!c) return; const t = c.currentTime + Math.max(0, delay);
    const o = c.createOscillator(), g = c.createGain();
    o.type = type; o.frequency.setValueAtTime(f, t);
    if (f2) o.frequency.exponentialRampToValueAtTime(Math.max(20, f2), t + dur);
    g.gain.setValueAtTime(vol, t); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g); g.connect(dest || this.master); o.start(t); o.stop(t + dur + 0.02);
  },
  noise(dur, vol = 0.15, freq = 2000, type = 'bandpass', delay = 0, dest = null) {
    const c = this.ctx; if (!c || !this.nbuf) return; const t = c.currentTime + Math.max(0, delay);
    const s = c.createBufferSource(); s.buffer = this.nbuf;
    const f = c.createBiquadFilter(); f.type = type; f.frequency.value = freq;
    const g = c.createGain(); g.gain.setValueAtTime(vol, t); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    s.connect(f); f.connect(g); g.connect(dest || this.master); s.start(t); s.stop(t + dur + 0.02);
  },
  // a big creature talking in no tongue at all: low chiptune syllables. Each one is a square tone with a saw an octave
  // under it, stepping (not gliding) in pitch, a soft 28 Hz growl on its loudness, a filter moving from one vowel to
  // another, and now and then a breath of a consonant at its start. No hard edges anywhere (they crackle): the growl is
  // a sine on its own gain stage after the envelope, so it dies with the syllable; every start and end is ramped
  voiceSyl(t, dur, f, f2, vol) {
    const c = this.ctx, F = [300, 420, 560, 760, 900], va = F[Math.floor(Math.random() * 5)], vb = F[Math.floor(Math.random() * 5)];
    const env = c.createGain(), trem = c.createGain(), flt = c.createBiquadFilter(), lfo = c.createOscillator(), lg = c.createGain();
    flt.type = 'lowpass'; flt.Q.value = 2.5; flt.frequency.setValueAtTime(va, t); flt.frequency.linearRampToValueAtTime(vb, t + dur);
    env.gain.setValueAtTime(0.0001, t); env.gain.exponentialRampToValueAtTime(vol, t + 0.04);
    env.gain.setValueAtTime(vol, t + dur * 0.65); env.gain.exponentialRampToValueAtTime(0.0001, t + dur + 0.04);
    trem.gain.value = 0.75; lfo.type = 'sine'; lfo.frequency.value = 28; lg.gain.value = 0.25; lfo.connect(lg); lg.connect(trem.gain);
    for (const [type, k, v] of [['square', 1, 0.8], ['sawtooth', 0.5, 0.8], ['triangle', 1.012, 0.6]]) {
      const o = c.createOscillator(), og = c.createGain(); o.type = type; og.gain.value = v;
      for (let s = 0; s < 4; s++) o.frequency.setValueAtTime((f + (f2 - f) * s / 3) * k, t + dur * s / 4);
      o.connect(og); og.connect(flt); o.start(t); o.stop(t + dur + 0.08);
    }
    flt.connect(env); env.connect(trem); trem.connect(this.master); lfo.start(t); lfo.stop(t + dur + 0.08);
    if (Math.random() < 0.55) { // the consonant: a short breath of noise, faded in and out
      const s = c.createBufferSource(), nf = c.createBiquadFilter(), ng = c.createGain();
      s.buffer = this.nbuf; nf.type = 'bandpass'; nf.frequency.value = 1200 + Math.random() * 1200; nf.Q.value = 1.5;
      ng.gain.setValueAtTime(0.0001, t); ng.gain.exponentialRampToValueAtTime(vol * 0.3, t + 0.012); ng.gain.exponentialRampToValueAtTime(0.0001, t + 0.07);
      s.connect(nf); nf.connect(ng); ng.connect(this.master); s.start(t); s.stop(t + 0.09);
    }
  },
  // the Eye's words (void.js): two phrases matching the two lines on screen, the first rising at its end,
  // the second ending in a long falling growl
  voice() {
    if (!this.ctx || !this.on) return;
    const t0 = this.ctx.currentTime + 0.02, j = () => 0.94 + Math.random() * 0.12;
    const SYL = [[0, 0.16, 82, 78], [0.2, 0.14, 92, 88], [0.38, 0.22, 73, 82], [0.66, 0.14, 98, 92], [0.84, 0.3, 87, 110],
      [1.1, 0.14, 73, 70], [1.28, 0.18, 82, 78], [1.5, 0.14, 65, 62], [1.68, 0.22, 73, 82], [1.98, 0.14, 92, 87], [2.16, 0.16, 82, 73], [2.38, 0.55, 62, 41]];
    for (const [s, d, f, f2] of SYL) { const k = j(); this.voiceSyl(t0 + s, d, f * k, f2 * k, s > 2.3 ? 0.135 : 0.11); } // mixed ~2-3 dB over the roar
  },
  sfx(n) {
    if (!this.ctx || !this.on) return;
    const now = this.ctx.currentTime;
    if (now - (this.last[n] ?? -1) < SFX_GAP) return;
    this.last[n] = now;
    switch (n) {
      case 'swing': this.noise(0.09, 0.12, 1800); this.tone(420, 0.08, 'sawtooth', 0.05, 200); break;
      case 'hit': this.tone(240, 0.1, 'square', 0.16, 80); this.noise(0.06, 0.12, 900); break;
      case 'kill': this.tone(320, 0.2, 'square', 0.16, 50); this.noise(0.14, 0.14, 700); break;
      case 'hurt': this.tone(190, 0.28, 'sawtooth', 0.22, 55); this.noise(0.12, 0.12, 400, 'lowpass'); break;
      case 'coin': this.tone(880, 0.06, 'square', 0.09); this.tone(1320, 0.12, 'square', 0.09, null, 0.06); break;
      case 'heart': [523, 659, 784].forEach((f, i) => this.tone(f, 0.12, 'triangle', 0.18, null, i * 0.07)); break;
      case 'dash': this.noise(0.13, 0.1, 3500, 'highpass'); this.tone(300, 0.12, 'sine', 0.1, 700); break;
      case 'bow': this.noise(0.06, 0.08, 2500); this.tone(700, 0.08, 'triangle', 0.08, 350); break;
      case 'buy': [523, 659, 784, 1047].forEach((f, i) => this.tone(f, 0.14, 'square', 0.1, null, i * 0.07)); break;
      case 'shoot': this.tone(520, 0.1, 'square', 0.07, 240); break;
      case 'open': [392, 523, 659, 784].forEach((f, i) => this.tone(f, 0.16, 'square', 0.09, null, i * 0.09)); break;
      case 'stairs': this.tone(500, 0.5, 'triangle', 0.2, 90); break;
      case 'roar': this.tone(90, 0.7, 'sawtooth', 0.25, 38); this.noise(0.5, 0.15, 300, 'lowpass'); break;
      case 'pick': this.tone(660, 0.08, 'square', 0.1); this.tone(990, 0.14, 'square', 0.1, null, 0.08); break;
      case 'potion': [392, 494, 587, 784].forEach((f, i) => this.tone(f, 0.1, 'triangle', 0.16, null, i * 0.05)); this.noise(0.18, 0.06, 1200); break;
      case 'nope': this.tone(150, 0.12, 'square', 0.08, 110); break;
      case 'tele': this.tone(900, 0.25, 'sine', 0.12, 200); this.noise(0.2, 0.06, 5000, 'highpass'); break;
      case 'fire': this.noise(0.18, 0.1, 900); this.tone(200, 0.15, 'sawtooth', 0.06, 90); break;
      case 'die': this.tone(260, 0.9, 'sawtooth', 0.22, 40); this.noise(0.6, 0.12, 500, 'lowpass'); break;
      case 'clang': this.tone(1760, 0.32, 'triangle', 0.14, 1700); this.tone(2640, 0.2, 'square', 0.04, 2500); this.noise(0.05, 0.1, 6000, 'highpass'); break;
      case 'boom': this.tone(110, 0.45, 'sawtooth', 0.22, 35); this.noise(0.4, 0.22, 500, 'lowpass'); break;
      case 'fuse': [0, 0.15, 0.3, 0.45].forEach(d => this.tone(1200, 0.05, 'square', 0.06, null, d)); break;
      case 'crack': this.noise(0.12, 0.16, 1400); this.noise(0.08, 0.1, 600, 'bandpass', 0.06); break;
      case 'flame': this.noise(0.6, 0.09, 700, 'lowpass'); this.noise(0.4, 0.05, 2500, 'bandpass', 0.05); break;
      case 'slam': this.tone(70, 0.6, 'sawtooth', 0.28, 30); this.noise(0.5, 0.22, 350, 'lowpass'); break;
      case 'bite': this.noise(0.06, 0.14, 1600); this.tone(260, 0.08, 'square', 0.1, 120); break;
      case 'burrow': this.noise(0.35, 0.12, 250, 'lowpass'); break;
      case 'chest': [523, 784, 659, 1047].forEach((f, i) => this.tone(f, 0.12, 'square', 0.1, null, i * 0.07)); this.noise(0.1, 0.06, 900); break;
      case 'freeze': [1568, 2093, 2637].forEach((f, i) => this.tone(f, 0.18, 'triangle', 0.07, f * 0.7, i * 0.03)); this.noise(0.3, 0.08, 7000, 'highpass'); break;
      case 'icicle': this.noise(0.06, 0.14, 3000); [2400, 3100, 1900].forEach((f, i) => this.tone(f, 0.12, 'triangle', 0.06, null, 0.02 + i * 0.04)); break;
      case 'roll': this.noise(1.2, 0.14, 160, 'lowpass'); this.tone(55, 1.1, 'sine', 0.18, 45); break;
      case 'crash': this.noise(0.45, 0.22, 600, 'lowpass'); this.noise(0.25, 0.12, 3500, 'highpass', 0.03); break;
      case 'shatter': this.noise(0.18, 0.16, 4200, 'highpass'); [2600, 3300].forEach((f, i) => this.tone(f, 0.1, 'triangle', 0.06, null, i * 0.05)); break;
      case 'howl': this.noise(1.0, 0.1, 900, 'bandpass'); this.tone(320, 0.9, 'sine', 0.05, 520); break;
      case 'cast': this.noise(0.12, 0.07, 1200); this.tone(300, 0.14, 'triangle', 0.08, 600); break;
      case 'shield': [523, 784, 1047].forEach((f, i) => this.tone(f, 0.25, 'triangle', 0.12, null, i * 0.05)); this.tone(1568, 0.4, 'sine', 0.06, 1760, 0.1); break;
      case 'whirl': this.noise(0.35, 0.1, 1800); this.tone(240, 0.3, 'sawtooth', 0.05, 480); break;
      case 'frost': this.noise(0.4, 0.12, 5000, 'highpass'); [1319, 1047, 880].forEach((f, i) => this.tone(f, 0.18, 'sine', 0.08, null, i * 0.06)); break;
      case 'fall': this.tone(620, 0.45, 'triangle', 0.12, 90); this.noise(0.4, 0.05, 900, 'lowpass'); break;
      case 'suck': this.tone(520, 0.35, 'sine', 0.1, 70); this.noise(0.3, 0.04, 2500, 'bandpass'); break;
      case 'zap': this.tone(1400, 0.28, 'sawtooth', 0.05, 320); this.noise(0.2, 0.06, 6000, 'highpass'); break;
      // necromancer: a ghostly whoosh of souls, the skeleton rising (a low moan and rattling bones), crumbling apart
      case 'souls': this.tone(520, 0.18, 'sine', 0.07, 780); this.noise(0.14, 0.05, 2200, 'bandpass'); break;
      case 'raise': this.tone(110, 0.5, 'triangle', 0.14, 220); [0, 0.07, 0.14, 0.21].forEach(d => this.noise(0.04, 0.09, 3000, 'bandpass', d + 0.2)); break;
      case 'bones': [0, 0.05, 0.1, 0.16, 0.22].forEach(d => this.noise(0.05, 0.1, 2400, 'bandpass', d)); this.tone(200, 0.25, 'triangle', 0.06, 80); break;
      // plague zone: rats squeal, a puff of corpse gas, a vial shatters, the gas goes up in flames
      case 'squeak': [0, 0.06].forEach(d => this.tone(2200 + Math.random() * 600, 0.05, 'square', 0.04, 1600, d)); break;
      case 'gas': this.noise(0.5, 0.1, 400, 'lowpass'); this.tone(90, 0.4, 'sine', 0.08, 60); break;
      case 'glass': this.noise(0.08, 0.12, 5000, 'highpass'); [3100, 2400].forEach((f, i) => this.tone(f, 0.08, 'triangle', 0.05, null, i * 0.03)); break;
      case 'blast': this.tone(80, 0.6, 'sawtooth', 0.26, 30); this.noise(0.7, 0.26, 700, 'lowpass'); this.noise(0.3, 0.1, 3000, 'bandpass', 0.05); break;
      // cathedral: a cracked bronze bell, a stone that comes alive; pyramid: a plate clicks, darts hiss, sand hisses
      case 'bell': [[196, 1.6, 0.16], [233.08, 1.2, 0.08], [466.16, 0.9, 0.05], [587.33, 0.6, 0.03]].forEach(([f, d, v]) => this.tone(f, d, 'sine', v, f * 0.985)); this.noise(0.12, 0.08, 1800); break;
      case 'stone': this.noise(0.2, 0.14, 500, 'lowpass'); this.tone(120, 0.18, 'square', 0.06, 70); break;
      case 'click': this.tone(1800, 0.03, 'square', 0.08); this.tone(900, 0.04, 'square', 0.06, null, 0.04); break;
      case 'darts': [0, 0.04, 0.08, 0.12].forEach(d => this.noise(0.08, 0.09, 4200, 'highpass', d)); break;
      case 'hiss': this.noise(1.2, 0.08, 2400, 'bandpass'); this.noise(1.0, 0.05, 600, 'lowpass', 0.2); break;
      case 'win': [523, 659, 784, 1047, 784, 1047, 1319].forEach((f, i) => this.tone(f, 0.22, 'square', 0.11, null, i * 0.13)); break;
    }
  }
};

const Music = {
  timer: null, step: 0, next: 0, mode: 'dun',
  bars: {
    dun: [{ r: 110, a: [220, 261.63, 329.63] }, { r: 87.31, a: [174.61, 220, 261.63] }, { r: 98, a: [196, 246.94, 293.66] }, { r: 82.41, a: [164.81, 196, 246.94] }],
    boss: [{ r: 73.42, a: [146.83, 174.61, 220] }, { r: 58.27, a: [116.54, 146.83, 174.61] }, { r: 65.41, a: [130.81, 155.56, 196] }, { r: 55, a: [110, 130.81, 164.81] }],
    // void zone: E minor - C - F (the Neapolitan, cold and strange) - B, slower
    void: [{ r: 82.41, a: [164.81, 196, 246.94] }, { r: 65.41, a: [130.81, 164.81, 196] }, { r: 87.31, a: [174.61, 220, 261.63] }, { r: 61.74, a: [123.47, 155.56, 185] }],
    // plague catacombs: A minor - F minor - E major (the harmonic minor's dread) - Bb, slow
    plague: [{ r: 110, a: [220, 261.63, 329.63] }, { r: 87.31, a: [174.61, 207.65, 261.63] }, { r: 82.41, a: [164.81, 207.65, 246.94] }, { r: 58.27, a: [116.54, 146.83, 174.61] }],
    // dark cathedral: C minor - Ab - F minor - G (a slow cadence that never resolves)
    cath: [{ r: 65.41, a: [130.81, 155.56, 196] }, { r: 51.91, a: [103.83, 130.81, 155.56] }, { r: 43.65, a: [87.31, 103.83, 130.81] }, { r: 49, a: [98, 123.47, 146.83] }],
    // buried pyramid: D phrygian - Eb major - C minor - D, the b2 of the old east
    pyr: [{ r: 73.42, a: [146.83, 174.61, 220] }, { r: 77.78, a: [155.56, 196, 233.08] }, { r: 65.41, a: [130.81, 155.56, 196] }, { r: 73.42, a: [146.83, 185, 220] }],
    // foundry: G minor - Eb - C minor - D, heavy and slow, a hammer on every beat (see play)
    forge: [{ r: 98, a: [196, 233.08, 293.66] }, { r: 77.78, a: [155.56, 196, 233.08] }, { r: 65.41, a: [130.81, 155.56, 196] }, { r: 73.42, a: [146.83, 185, 220] }],
    // petrified city: E minor - C - A minor - B, almost nothing but the drone and a far bell
    stone: [{ r: 82.41, a: [164.81, 196, 246.94] }, { r: 65.41, a: [130.81, 164.81, 196] }, { r: 55, a: [110, 130.81, 164.81] }, { r: 61.74, a: [123.47, 155.56, 185] }],
    // title: slow D minor - Bb - G minor - A (harmonic minor), a melody note every 4 steps with a chromatic or b2 step
    // at the end of each bar for the menace; 0 = rest
    title: [{ r: 73.42, m: [293.66, 349.23, 440, 415.3] }, { r: 58.27, m: [349.23, 293.66, 233.08, 220] },
      { r: 49, m: [196, 233.08, 293.66, 311.13] }, { r: 55, m: [277.18, 329.63, 392, 0] }]
  },
  start(mode) {
    this.stop(); this.mode = mode;
    if (!Snd.ctx) return;
    this.step = 0; this.next = Snd.ctx.currentTime + 0.1;
    this.timer = setInterval(() => this.sched(), 40);
  },
  stop() { if (this.timer) clearInterval(this.timer); this.timer = null; },
  sched() {
    const c = Snd.ctx; if (!c) return;
    // muted or the tab is hidden: no notes are queued at all
    if (!Snd.on || document.hidden) { this.next = c.currentTime + 0.1; return; }
    // the timer fell behind (a background tab throttles it to 1/s or even 1/min, or the page hitched):
    // drop the missed notes instead of firing them all at once as one loud burst
    if (this.next < c.currentTime) this.next = c.currentTime + 0.05;
    const boss = this.mode === 'boss', dur = boss ? 0.107 : this.mode === 'title' ? 0.24 : this.mode === 'void' ? 0.19 : this.mode === 'plague' || this.mode === 'cath' ? 0.2 : this.mode === 'pyr' ? 0.17 : this.mode === 'forge' ? 0.19 : this.mode === 'stone' ? 0.24 : 0.15;
    while (this.next < c.currentTime + 0.25) { this.play(this.step, this.next - c.currentTime, boss); this.next += dur; this.step++; }
  },
  play(step, delay, boss) {
    if (this.mode === 'title') { this.playTitle(step, delay); return; }
    const s = step % 16, bar = this.bars[this.mode][(step >> 4) % 4], d = Snd.mus;
    if (s % 4 === 0) Snd.tone(bar.r, boss ? 0.2 : 0.28, boss ? 'sawtooth' : 'triangle', boss ? 0.16 : 0.22, null, delay, d);
    if (s % 2 === 0) Snd.tone(bar.a[(s >> 1) % 3] * (boss && s % 8 === 6 ? 2 : 1), 0.1, 'square', 0.035, null, delay, d);
    if (s === 0 || s === 8 || (boss && s === 10)) Snd.tone(130, 0.12, 'sine', 0.3, 40, delay, d);
    if (s % 4 === 2) Snd.noise(0.04, 0.05, 7000, 'highpass', delay, d);
    if (this.mode === 'forge' && s % 8 === 4) { Snd.tone(880, 0.06, 'square', 0.03, 600, delay, d); Snd.noise(0.05, 0.05, 2500, 'bandpass', delay, d); } // the hammer
  },
  // the title theme: a low drone per bar, a slow organ-like melody, a funeral bell and a cold wind
  playTitle(step, delay) {
    const s = step % 16, bi = (step >> 4) % 4, bar = this.bars.title[bi], d = Snd.mus, v = 1.7; // v: overall title loudness
    if (s === 0) {
      Snd.tone(bar.r, 3.9, 'triangle', 0.16 * v, null, delay, d); Snd.tone(bar.r / 2, 3.9, 'sine', 0.2 * v, null, delay, d);
      Snd.noise(2.4, 0.025 * v, 500, 'lowpass', delay, d);
    }
    if (s % 4 === 0) {
      const f = bar.m[s >> 2];
      if (f) { Snd.tone(f, 0.95, 'triangle', 0.07 * v, null, delay, d); Snd.tone(f * 1.003, 0.95, 'sawtooth', 0.012 * v, null, delay, d); }
    }
    // the bell tolls on the first beat of bars 1 and 3: a high D with an out-of-tune partial
    if (s === 0 && bi % 2 === 0) { Snd.tone(587.33, 2.6, 'sine', 0.06 * v, null, delay, d); Snd.tone(587.33 * 2.76, 1.4, 'sine', 0.02 * v, null, delay, d); }
    // a slow heartbeat under it all
    if (s === 8 || s === 10) Snd.tone(70, 0.18, 'sine', (s === 8 ? 0.22 : 0.14) * v, 38, delay, d);
  }
};
