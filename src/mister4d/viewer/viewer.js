// MISTER 4D Player: a small WebGL2 renderer for baked dynamic Gaussian scenes.
// No dependencies. Scene format: see mister4d/export.py.
'use strict';

const VERTEX = `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
layout(location = 0) in vec2 aCorner;
layout(location = 1) in uint aIndex;
uniform usampler2D uPos;
uniform sampler2D uRot;
uniform sampler2D uScale;
uniform sampler2D uColor;
uniform int uTexWidth;
uniform vec3 uBoxMin;
uniform vec3 uBoxSize;
uniform vec2 uLogScale;
uniform float uSplatScale;
uniform mat4 uView;     // world -> camera, camera looks along +z, y down
uniform vec2 uFocal;    // pixels
uniform vec2 uCenter;   // principal point, pixels
uniform vec2 uViewport; // pixels
out vec4 vColor;
out vec2 vOffset;

void main() {
  ivec2 uv = ivec2(int(aIndex) % uTexWidth, int(aIndex) / uTexWidth);
  vec3 pos = uBoxMin + vec3(texelFetch(uPos, uv, 0).rgb) / 65535.0 * uBoxSize;
  vec4 cam = uView * vec4(pos, 1.0);
  vec4 color = texelFetch(uColor, uv, 0);
  if (cam.z < 0.05 || color.a < 0.004) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }
  vec4 q = normalize(texelFetch(uRot, uv, 0) * 2.0 - 1.0);  // w x y z
  float w = q.x, x = q.y, y = q.z, z = q.w;
  mat3 R = mat3(
    1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y),
    2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
    2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y));
  vec3 s = uSplatScale * exp(mix(vec3(uLogScale.x), vec3(uLogScale.y), texelFetch(uScale, uv, 0).rgb));
  mat3 M = R * mat3(s.x, 0.0, 0.0, 0.0, s.y, 0.0, 0.0, 0.0, s.z);
  mat3 sigma = M * transpose(M);

  // Project the 3D covariance to the image plane (EWA splatting).
  float iz = 1.0 / cam.z;
  mat3 J = mat3(
    uFocal.x * iz, 0.0, 0.0,
    0.0, uFocal.y * iz, 0.0,
    -uFocal.x * cam.x * iz * iz, -uFocal.y * cam.y * iz * iz, 0.0);
  mat3 T = J * mat3(uView);
  mat3 cov = T * sigma * transpose(T);
  float a = cov[0][0] + 0.3, b = cov[0][1], c = cov[1][1] + 0.3;

  // Principal axes of the 2D Gaussian.
  float mid = 0.5 * (a + c);
  float rad = sqrt(max(mid * mid - (a * c - b * b), 1e-8));
  float l1 = mid + rad, l2 = max(mid - rad, 0.05);
  vec2 v1 = abs(b) > 1e-6 ? normalize(vec2(b, l1 - a)) : (a >= c ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
  vec2 v2 = vec2(-v1.y, v1.x);
  vec2 center = uFocal * cam.xy * iz + uCenter;
  vec2 px = center + aCorner.x * 3.33 * sqrt(l1) * v1 + aCorner.y * 3.33 * sqrt(l2) * v2;
  gl_Position = vec4(px.x / uViewport.x * 2.0 - 1.0, 1.0 - px.y / uViewport.y * 2.0, 0.0, 1.0);
  vOffset = aCorner * 3.33;
  vColor = color;
}`;

const FRAGMENT = `#version 300 es
precision highp float;
in vec4 vColor;
in vec2 vOffset;
out vec4 outColor;
void main() {
  float alpha = min(vColor.a * exp(-0.5 * dot(vOffset, vOffset)), 0.999);
  if (alpha < 1.0 / 255.0) discard;
  outColor = vec4(vColor.rgb * alpha, alpha);
}`;

const $ = (id) => document.getElementById(id);
const canvas = $('gl');
const gl = canvas.getContext('webgl2', { alpha: false, antialias: false });
const wide = window.matchMedia('(min-width: 900px)');

const state = {
  scene: null, base: '', embedded: false, frames: [], loaded: 0,
  frame: 0, shown: -1, clock: 0, playing: true, speed: 1, loop: true,
  az: 0, el: 0.15, dist: 3, target: [0, 0, 0], fovY: 40 * Math.PI / 180, home: null,
  preset: null, dirty: true, light: false, splatScale: 1, lastTime: 0, fps: 0, statsAt: 0,
};
let program, uniforms, textures, indexBuffer, order, depth, keys, counts;

const number = (value, digits = 0) => value.toLocaleString('it-IT',
  { minimumFractionDigits: digits, maximumFractionDigits: digits });

// ------------------------------------------------------------------- GL ---

function compile(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}

function dataTexture(unit, internal, format, type) {
  const { tex_width: w, tex_height: h } = state.scene;
  const tex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, null);
  return { tex, unit, format, type };
}

function updateTexture(t, data) {
  const { tex_width: w, tex_height: h } = state.scene;
  gl.activeTexture(gl.TEXTURE0 + t.unit);
  gl.bindTexture(gl.TEXTURE_2D, t.tex);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h, t.format, t.type, data);
}

function initGL() {
  program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, VERTEX));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, FRAGMENT));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  gl.useProgram(program);
  uniforms = {};
  for (const name of ['uPos', 'uRot', 'uScale', 'uColor', 'uTexWidth', 'uBoxMin', 'uBoxSize',
    'uLogScale', 'uSplatScale', 'uView', 'uFocal', 'uCenter', 'uViewport']) {
    uniforms[name] = gl.getUniformLocation(program, name);
  }
  const quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  indexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_INT, 0, 0);
  gl.vertexAttribDivisor(1, 1);
  gl.disable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);  // premultiplied colours, drawn back to front
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
}

function setupScene() {
  const s = state.scene;
  const n = Math.max(...s.counts);  // buffers sized for the busiest frame
  order = new Uint32Array(n);
  depth = new Float32Array(n);
  keys = new Uint16Array(n);
  counts = new Uint32Array(65536);
  gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, order.byteLength, gl.DYNAMIC_DRAW);
  textures = {
    pos: dataTexture(0, gl.RGB16UI, gl.RGB_INTEGER, gl.UNSIGNED_SHORT),
    rot: dataTexture(1, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE),
    scale: dataTexture(2, gl.RGB8, gl.RGB, gl.UNSIGNED_BYTE),
    color: dataTexture(3, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE),
  };
  gl.uniform1i(uniforms.uPos, 0);
  gl.uniform1i(uniforms.uRot, 1);
  gl.uniform1i(uniforms.uScale, 2);
  gl.uniform1i(uniforms.uColor, 3);
  gl.uniform1i(uniforms.uTexWidth, s.tex_width);
  gl.uniform3fv(uniforms.uBoxMin, s.bbox_min);
  gl.uniform3fv(uniforms.uBoxSize, s.bbox_max.map((v, i) => v - s.bbox_min[i]));
  gl.uniform2f(uniforms.uLogScale, s.log_scale_min, s.log_scale_max);
}

// --------------------------------------------------------------- camera ---

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const normalize = (a) => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

function orbitEye() {
  const { az, el, dist, target } = state;
  return [target[0] + dist * Math.cos(el) * Math.cos(az),
    target[1] + dist * Math.cos(el) * Math.sin(az),
    target[2] + dist * Math.sin(el)];
}

// Part of the canvas (canvas pixels) not covered by the docked panels: the
// subject is framed there, so opening a panel never hides it.
function stage() {
  const k = canvas.width / Math.max(canvas.clientWidth, 1);
  let right = canvas.clientWidth, bottom = canvas.clientHeight;
  if (!document.body.classList.contains('ui-hidden')) {
    if (!$('transport').hidden) bottom = $('transport').getBoundingClientRect().top - 8;
    if (document.body.classList.contains('docked')) right = $('sidebar').getBoundingClientRect().left - 8;
  }
  return { w: Math.max(160, right) * k, h: Math.max(120, bottom) * k };
}

// World->camera matrix (row-major 3x4), focal length and principal point in pixels.
function currentCamera() {
  const { w: W, h: H } = stage();
  if (state.preset) {
    const c = state.preset;
    const k = Math.min(W / c.width, H / c.height);
    return {
      rows: [[...c.R[0], c.t[0]], [...c.R[1], c.t[1]], [...c.R[2], c.t[2]]],
      focal: [c.K[0][0] * k, c.K[1][1] * k],
      center: [W / 2 + (c.K[0][2] - (c.width - 1) / 2) * k, H / 2 + (c.K[1][2] - (c.height - 1) / 2) * k],
    };
  }
  const eye = orbitEye();
  const z = normalize(sub(state.target, eye));
  const x = normalize(cross(z, [0, 0, 1]));
  const y = cross(z, x);
  const f = (H / 2) / Math.tan(state.fovY / 2);
  return {
    rows: [[...x, -dot(x, eye)], [...y, -dot(y, eye)], [...z, -dot(z, eye)]],
    focal: [f, f], center: [W / 2, H / 2],
  };
}

// Leave a rig camera and keep orbiting from where it stands.
function leavePreset() {
  if (!state.preset) return;
  const c = state.preset;
  const eye = [0, 1, 2].map((i) => -(c.R[0][i] * c.t[0] + c.R[1][i] * c.t[1] + c.R[2][i] * c.t[2]));
  const d = sub(eye, state.target);
  state.dist = Math.hypot(...d);
  state.az = Math.atan2(d[1], d[0]);
  state.el = Math.asin(Math.max(-1, Math.min(1, d[2] / state.dist)));
  state.fovY = 2 * Math.atan(c.height / (2 * c.K[1][1]));
  state.preset = null;
  const free = document.querySelector('input[name="camera"][value=""]');
  if (free) free.checked = true;
}

// Free view from the direction of the first training camera (the best
// constrained one), aimed at the subject and far enough to see all of it.
function homeView() {
  const s = state.scene;
  state.target = s.bbox_min.map((v, i) => (v + s.bbox_max[i]) / 2);
  const first = s.cameras.find((c) => !c.holdout) || s.cameras[0];
  if (first) { state.preset = first; leavePreset(); }
  const radius = 0.5 * Math.hypot(...s.bbox_max.map((v, i) => v - s.bbox_min[i]));
  state.dist = Math.max(0.3, 1.05 * radius / Math.sin(state.fovY / 2));
  state.dirty = true;
}

// --------------------------------------------------------------- render ---

function sortBackToFront(pos, n, row) {
  const s = state.scene;
  const size = s.bbox_max.map((v, i) => (v - s.bbox_min[i]) / 65535);
  const ax = row[0] * size[0], ay = row[1] * size[1], az = row[2] * size[2];
  const d0 = row[0] * s.bbox_min[0] + row[1] * s.bbox_min[1] + row[2] * s.bbox_min[2] + row[3];
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) {
    const z = ax * pos[3 * i] + ay * pos[3 * i + 1] + az * pos[3 * i + 2] + d0;
    depth[i] = z;
    if (z < lo) lo = z;
    if (z > hi) hi = z;
  }
  const k = 65535 / Math.max(hi - lo, 1e-9);
  counts.fill(0);
  for (let i = 0; i < n; i++) {
    const key = 65535 - ((depth[i] - lo) * k | 0);  // far splats first
    keys[i] = key;
    counts[key]++;
  }
  let total = 0;
  for (let i = 0; i < 65536; i++) { const c = counts[i]; counts[i] = total; total += c; }
  for (let i = 0; i < n; i++) order[counts[keys[i]]++] = i;
  return order.subarray(0, n);
}

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    state.dirty = true;
  }
}

function draw(now) {
  requestAnimationFrame(draw);
  const s = state.scene;
  const dt = Math.min((now - state.lastTime) / 1000, 0.1);
  state.lastTime = now;
  if (dt > 0) state.fps = state.fps * 0.9 + (1 / dt) * 0.1;

  if (state.playing && s.num_frames > 1) {
    let clock = state.clock + dt * state.speed * s.fps;
    if (clock >= s.num_frames) {
      if (state.loop) clock %= s.num_frames;
      else { clock = s.num_frames - 1; setPlaying(false); }
    }
    if (state.frames[Math.floor(clock)]) {  // otherwise wait: that frame is still loading
      state.clock = clock;
      state.frame = Math.floor(clock);
    }
  }
  const data = state.frames[state.frame];
  if (!data) return;
  resize();
  if (state.shown !== state.frame) {
    updateTexture(textures.pos, data.pos);
    updateTexture(textures.rot, data.rot);
    updateTexture(textures.scale, data.scale);
    updateTexture(textures.color, data.color);
    state.shown = state.frame;
    state.dirty = true;
    syncTimeline();
  }
  const cam = currentCamera();
  const count = s.counts[state.frame];
  if (state.dirty) {
    gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, sortBackToFront(data.pos, count, cam.rows[2]));
    state.dirty = false;
  }
  const r = cam.rows;
  gl.uniformMatrix4fv(uniforms.uView, false, new Float32Array([
    r[0][0], r[1][0], r[2][0], 0, r[0][1], r[1][1], r[2][1], 0,
    r[0][2], r[1][2], r[2][2], 0, r[0][3], r[1][3], r[2][3], 1]));
  gl.uniform2fv(uniforms.uFocal, cam.focal);
  gl.uniform2fv(uniforms.uCenter, cam.center);
  gl.uniform2f(uniforms.uViewport, canvas.width, canvas.height);
  gl.uniform1f(uniforms.uSplatScale, state.splatScale);
  gl.viewport(0, 0, canvas.width, canvas.height);
  if (state.light) gl.clearColor(0.941, 0.929, 0.969, 1); else gl.clearColor(0.082, 0.075, 0.114, 1);  // theme --bg
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count);

  if (now - state.statsAt > 500) {  // the details panel does not need 60 updates a second
    state.statsAt = now;
    $('d-count').textContent = number(count);
    $('d-render').textContent = `${Math.round(state.fps)} fps`;
  }
}

// ----------------------------------------------------------------- data ---

// A single-file export carries the frames inside the page, base64-encoded in
// <script type="application/octet-stream"> elements: nothing is downloaded, so
// the file opens with a double click (browsers block fetch() on file://).
function embeddedFrame(i) {
  const node = $(`mister4d-frame-${i}`);
  const text = atob(node.textContent);
  node.remove();  // free the encoded copy
  const bytes = new Uint8Array(text.length);
  for (let k = 0; k < text.length; k++) bytes[k] = text.charCodeAt(k);
  return bytes.buffer;
}

async function loadFrames() {
  const s = state.scene;
  const slots = s.tex_width * s.tex_height;
  let next = 0;
  async function worker() {
    while (next < s.num_frames) {
      const i = next++;
      let buffer;
      if (state.embedded) {
        buffer = embeddedFrame(i);
        await new Promise((resolve) => setTimeout(resolve));  // keep the page responsive
      } else {
        const response = await fetch(state.base + s.frames[i]);
        if (!response.ok) throw new Error(`${s.frames[i]}: HTTP ${response.status}`);
        buffer = await response.arrayBuffer();
      }
      state.frames[i] = {
        pos: new Uint16Array(buffer, 0, slots * 3),
        rot: new Uint8Array(buffer, slots * 6, slots * 4),
        scale: new Uint8Array(buffer, slots * 10, slots * 3),
        color: new Uint8Array(buffer, slots * 13, slots * 4),
      };
      state.loaded++;
      showLoading();
    }
  }
  await Promise.all([worker(), worker(), worker(), worker()]);
}

// ------------------------------------------------------------ interface ---

function showLoading() {
  const s = state.scene;
  const done = state.loaded >= s.num_frames;
  $('loading').hidden = done;
  $('loading').firstElementChild.style.width = `${100 * state.loaded / s.num_frames}%`;
  $('status').textContent = done
    ? (s.num_frames > 1
      ? `${s.num_frames} frame · ${number(s.fps, s.fps % 1 ? 2 : 0)} fps · ${number(s.num_frames / s.fps, 2)} s`
      : 'Scena statica')
    : `Caricamento ${state.loaded} di ${s.num_frames} frame`;
}

function fail(text) {
  $('loading').hidden = true;
  $('status').textContent = text;
  $('status').classList.add('error');
  $('status').setAttribute('role', 'alert');
}

function setPlaying(playing) {
  state.playing = playing;
  $('iconPlay').toggleAttribute('hidden', playing);
  $('iconPause').toggleAttribute('hidden', !playing);
  $('playLabel').textContent = playing ? 'Pausa' : 'Riproduci';
}

function seek(frame) {
  const s = state.scene;
  const f = Math.max(0, Math.min(s.num_frames - 1, frame));
  if (state.frames[f]) { state.frame = f; state.clock = f; }
}

function step(delta) { setPlaying(false); seek(state.frame + delta); }

function syncTimeline() {
  const s = state.scene;
  const last = Math.max(s.num_frames - 1, 1);
  $('time').value = state.frame;
  $('time').style.setProperty('--pct', `${100 * state.frame / last}%`);
  $('time').setAttribute('aria-valuetext', `frame ${state.frame + 1} di ${s.num_frames}`);
  $('clock').textContent = `${number(state.frame / s.fps, 2)} s`;
  $('frame').textContent = `frame ${state.frame + 1} di ${s.num_frames}`;
}

function setPanel(open) {
  $('sidebar').hidden = !open;
  $('togglePanel').setAttribute('aria-expanded', String(open));
  document.body.classList.toggle('docked', open && wide.matches);
  state.dirty = true;
}

function setTheme(light) {
  state.light = light;
  document.body.dataset.theme = light ? 'light' : 'dark';
}

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else if (document.documentElement.requestFullscreen) document.documentElement.requestFullscreen().catch(() => {});
}

function fillDetails() {
  const s = state.scene;
  const size = s.bbox_max.map((v, i) => number(v - s.bbox_min[i], 2));
  const held = s.cameras.filter((c) => c.holdout).length;
  $('d-frames').textContent = number(s.num_frames);
  $('d-fps').textContent = `${number(s.fps, s.fps % 1 ? 2 : 0)} fps`;
  $('d-duration').textContent = `${number(s.num_frames / s.fps, 2)} s`;
  $('d-size').textContent = `${size.join(' × ')} m`;
  $('d-cameras').textContent = held ? `${s.cameras.length - held} + ${held} di verifica` : number(s.cameras.length);
}

function buildCameraChoices() {
  const group = $('cameras');
  const add = (value, label, note) => {
    const el = document.createElement('label');
    const input = Object.assign(document.createElement('input'), { type: 'radio', name: 'camera', value });
    const text = document.createElement('span');
    text.textContent = label;
    if (note) text.appendChild(Object.assign(document.createElement('small'), { textContent: note }));
    el.append(input, text);
    group.appendChild(el);
    return input;
  };
  add('', 'Libera').checked = true;
  for (const c of state.scene.cameras) {
    const input = add(c.name, c.name, c.holdout ? 'verifica' : '');
    input.parentElement.title = c.holdout
      ? 'Camera esclusa dal training: mostra come regge un punto di vista mai visto'
      : 'Camera usata per il training';
  }
  group.addEventListener('change', (e) => {
    const cam = state.scene.cameras.find((c) => c.name === e.target.value);
    if (cam) state.preset = cam; else leavePreset();
    state.dirty = true;
  });
}

function setupControls() {
  const s = state.scene;

  // Transport: only for scenes that actually move.
  $('transport').hidden = s.num_frames <= 1;
  $('time').max = s.num_frames - 1;
  $('duration').textContent = `${number(s.num_frames / s.fps, 2)} s`;
  $('play').addEventListener('click', () => {
    if (!state.playing && !state.loop && state.frame >= s.num_frames - 1) seek(0);
    setPlaying(!state.playing);
  });
  $('prev').addEventListener('click', () => step(-1));
  $('next').addEventListener('click', () => step(1));
  $('time').addEventListener('input', (e) => seek(Number(e.target.value)));
  $('speed').addEventListener('change', (e) => { state.speed = Number(e.target.value); });
  $('loop').addEventListener('click', () => {
    state.loop = !state.loop;
    $('loop').setAttribute('aria-pressed', String(state.loop));
  });

  // Point of view and appearance.
  buildCameraChoices();
  $('resetView').addEventListener('click', homeView);
  for (const input of document.querySelectorAll('input[name="background"]')) {
    input.addEventListener('change', () => setTheme(input.value === 'light'));
  }
  $('splat').addEventListener('input', (e) => {
    state.splatScale = Number(e.target.value) / 100;
    $('splatValue').textContent = `${e.target.value}%`;
    e.target.style.setProperty('--pct', `${100 * (e.target.value - e.target.min) / (e.target.max - e.target.min)}%`);
  });
  $('splat').dispatchEvent(new Event('input'));
  fillDetails();

  // Panels.
  $('togglePanel').addEventListener('click', () => setPanel($('sidebar').hidden));
  $('closePanel').addEventListener('click', () => { setPanel(false); $('togglePanel').focus(); });
  wide.addEventListener('change', () => setPanel(!$('sidebar').hidden));
  setPanel(wide.matches);
  $('fullscreen').addEventListener('click', toggleFullscreen);
  document.addEventListener('fullscreenchange', () => {
    $('fullscreen').setAttribute('aria-pressed', String(Boolean(document.fullscreenElement)));
  });
  const toggleUi = () => { document.body.classList.toggle('ui-hidden'); state.dirty = true; };
  $('showUi').addEventListener('click', toggleUi);
  new ResizeObserver(() => {
    document.documentElement.style.setProperty('--transport-h', `${$('transport').offsetHeight}px`);
    state.dirty = true;
  }).observe($('transport'));

  window.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const tag = e.target.tagName;
    const typing = tag === 'INPUT' || tag === 'SELECT' || tag === 'SUMMARY';
    if (e.code === 'Space' && !typing && tag !== 'BUTTON') { e.preventDefault(); $('play').click(); return; }
    if (typing) return;  // arrows and letters belong to the focused control
    const actions = {
      ArrowLeft: () => step(-1), ArrowRight: () => step(1),
      Home: () => step(-state.frame), End: () => step(s.num_frames),
      KeyR: homeView, KeyF: toggleFullscreen, KeyH: toggleUi,
    };
    if (actions[e.code]) { e.preventDefault(); actions[e.code](); }
  });

  // Mouse, pen and touch on the scene: drag to orbit, shift/right drag to pan,
  // wheel or pinch to zoom.
  const pointers = new Map();
  let pinch = 0;
  const zoom = (factor) => {
    leavePreset();
    state.dist = Math.max(0.2, Math.min(50, state.dist * factor));
    state.dirty = true;
  };
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('pointerdown', (e) => {
    leavePreset();
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey });
    canvas.setPointerCapture(e.pointerId);
    canvas.classList.add('dragging');
    pinch = 0;
  });
  canvas.addEventListener('pointermove', (e) => {
    const p = pointers.get(e.pointerId);
    if (!p) return;
    const dx = e.clientX - p.x, dy = e.clientY - p.y;
    p.x = e.clientX;
    p.y = e.clientY;
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const spread = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinch > 0 && spread > 0) zoom(pinch / spread);
      pinch = spread;
    } else if (p.pan) {
      const cam = currentCamera();
      const k = state.dist / cam.focal[1] * (canvas.height / canvas.clientHeight);
      for (let i = 0; i < 3; i++) state.target[i] -= (cam.rows[0][i] * dx + cam.rows[1][i] * dy) * k;
    } else {
      state.az -= dx * 0.006;
      state.el = Math.max(-1.5, Math.min(1.5, state.el + dy * 0.006));
    }
    state.dirty = true;
  });
  const release = (e) => {
    pointers.delete(e.pointerId);
    pinch = 0;
    if (!pointers.size) canvas.classList.remove('dragging');
  };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);
  canvas.addEventListener('wheel', (e) => { e.preventDefault(); zoom(Math.exp(e.deltaY * 0.001)); },
    { passive: false });
  window.addEventListener('resize', () => { state.dirty = true; });
}

async function main() {
  if (!gl) return fail('Questo browser non supporta WebGL2: prova con una versione recente di Chrome, Edge o Firefox.');
  const url = new URLSearchParams(location.search).get('scene') || 'scene.json';
  state.base = url.includes('/') ? url.slice(0, url.lastIndexOf('/') + 1) : '';
  const embedded = $('mister4d-scene');
  state.embedded = Boolean(embedded);
  try {
    state.scene = embedded ? JSON.parse(embedded.textContent) : await (await fetch(url)).json();
    const s = state.scene;
    if (s.format !== 'mister4d-web' || s.version !== 2) throw new Error('il file non è un export mister4d (versione 2)');
    initGL();
    setupScene();
    setupControls();
    homeView();
    showLoading();
    syncTimeline();
    requestAnimationFrame(draw);
    await loadFrames();
  } catch (err) {
    fail(`Impossibile caricare la scena: ${err.message}`);
    throw err;
  }
}

// Small hook for automated tests and for driving the player from the console.
window.mister4d = {
  state,
  setFrame(i) { setPlaying(false); seek(i); },
  setCamera(name) {
    const input = document.querySelector(`input[name="camera"][value="${name || ''}"]`);
    if (input) { input.checked = true; input.dispatchEvent(new Event('change', { bubbles: true })); }
  },
};

main();
