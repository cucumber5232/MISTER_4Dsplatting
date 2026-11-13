// MISTER 4D Player: a small WebGL2 renderer for dynamic Gaussian scenes.
// No dependencies. Scene format: see mister4d/export.py ("mister4d-web", version 3).
//
// Every Gaussian is stored once, with the instant it is centred on, its life
// and its velocity. The vertex shader moves and fades it for the instant being
// shown; the page only chooses which ones can be visible and sorts them.
'use strict';

const VERTEX = `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
layout(location = 0) in vec2 aCorner;
layout(location = 1) in uint aIndex;
uniform usampler2D uA;    // position x y z, time (16 bit)
uniform usampler2D uB;    // velocity x y z, life (8 bit)
uniform sampler2D uC;     // rotation w x y z
uniform sampler2D uD;     // log-scale x y z, peak opacity
uniform sampler2D uE;     // colour
uniform int uTexWidth;
uniform vec3 uBoxMin;
uniform vec3 uBoxSize;
uniform vec2 uTimeRange;  // first instant and span of the chunk, in frames
uniform float uVelMax;    // metres per frame at full scale
uniform vec2 uLife;       // log of the shortest and of the longest life, in frames
uniform vec2 uLogScale;
uniform float uTime;      // the instant being shown, in frames
uniform float uMinOpacity;
uniform float uSplatScale;
uniform mat4 uView;     // world -> camera, camera looks along +z, y down
uniform vec2 uFocal;    // pixels
uniform vec2 uCenter;   // principal point, pixels
uniform vec2 uViewport; // pixels
out vec4 vColor;
out vec2 vOffset;

void main() {
  ivec2 uv = ivec2(int(aIndex) % uTexWidth, int(aIndex) / uTexWidth);
  uvec4 a = texelFetch(uA, uv, 0);
  uvec4 b = texelFetch(uB, uv, 0);
  vec4 d = texelFetch(uD, uv, 0);
  float dt = 0.0, fade = 1.0;
  if (b.a != 255u) {  // 255 = does not depend on time
    dt = uTime - (uTimeRange.x + float(a.a) / 65535.0 * uTimeRange.y);
    float sigma = exp(mix(uLife.x, uLife.y, float(b.a) / 254.0));
    fade = exp(-0.5 * dt * dt / (sigma * sigma));
  }
  float opacity = d.a * fade;
  vec3 code = (vec3(b.rgb) - 128.0) / 127.0;  // the velocity is the square of its code, with its sign
  vec3 pos = uBoxMin + vec3(a.rgb) / 65535.0 * uBoxSize + code * abs(code) * uVelMax * dt;
  vec4 cam = uView * vec4(pos, 1.0);
  if (cam.z < 0.05 || opacity < uMinOpacity) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }
  vec4 q = normalize(texelFetch(uC, uv, 0) * 2.0 - 1.0);  // w x y z
  float w = q.x, x = q.y, y = q.z, z = q.w;
  mat3 R = mat3(
    1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y),
    2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
    2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y));
  vec3 s = uSplatScale * exp(mix(vec3(uLogScale.x), vec3(uLogScale.y), d.rgb));
  mat3 M = R * mat3(s.x, 0.0, 0.0, 0.0, s.y, 0.0, 0.0, 0.0, s.z);
  mat3 sigma3 = M * transpose(M);

  // Project the 3D covariance to the image plane (EWA splatting).
  float iz = 1.0 / cam.z;
  mat3 J = mat3(
    uFocal.x * iz, 0.0, 0.0,
    0.0, uFocal.y * iz, 0.0,
    -uFocal.x * cam.x * iz * iz, -uFocal.y * cam.y * iz * iz, 0.0);
  mat3 T = J * mat3(uView);
  mat3 cov = T * sigma3 * transpose(T);
  float ca = cov[0][0] + 0.3, cb = cov[0][1], cc = cov[1][1] + 0.3;

  // Principal axes of the 2D Gaussian.
  float mid = 0.5 * (ca + cc);
  float rad = sqrt(max(mid * mid - (ca * cc - cb * cb), 1e-8));
  float l1 = mid + rad, l2 = max(mid - rad, 0.05);
  vec2 v1 = abs(cb) > 1e-6 ? normalize(vec2(cb, l1 - ca)) : (ca >= cc ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
  vec2 v2 = vec2(-v1.y, v1.x);
  vec2 center = uFocal * cam.xy * iz + uCenter;
  vec2 px = center + aCorner.x * 3.33 * sqrt(l1) * v1 + aCorner.y * 3.33 * sqrt(l2) * v2;
  gl_Position = vec4(px.x / uViewport.x * 2.0 - 1.0, 1.0 - px.y / uViewport.y * 2.0, 0.0, 1.0);
  vOffset = aCorner * 3.33;
  vColor = vec4(texelFetch(uE, uv, 0).rgb, opacity);
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

const TEX_WIDTH = 2048;
const state = {
  scene: null, base: '', embedded: false, chunks: [], active: -1, ready: false, drawn: 0,
  frame: 0, shown: -1, clock: 0, playing: true, speed: 1, loop: true,
  az: 0, el: 0.15, dist: 3, target: [0, 0, 0], fovY: 40 * Math.PI / 180, home: null,
  preset: null, dirty: true, splatScale: 1, lastTime: 0, fps: 0, statsAt: 0,
};
let program, uniforms, indexBuffer, order, ids, depth, keys, counts;

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

// One block of a chunk as a texture, TEX_WIDTH Gaussians per row. The block
// is uploaded as it is in the file: whole rows first, then what is left.
function blockTexture(internal, format, type, data, channels, count) {
  const tex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0 + 7);  // a unit the draw call does not use
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const rows = Math.floor(count / TEX_WIDTH), rest = count - rows * TEX_WIDTH;
  gl.texImage2D(gl.TEXTURE_2D, 0, internal, TEX_WIDTH, rows + (rest ? 1 : 0) || 1, 0, format, type, null);
  if (rows) gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, TEX_WIDTH, rows, format, type, data, 0);
  if (rest) gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, rows, rest, 1, format, type, data, rows * TEX_WIDTH * channels);
  return tex;
}

function initGL() {
  program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, VERTEX));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, FRAGMENT));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  gl.useProgram(program);
  uniforms = {};
  for (const name of ['uA', 'uB', 'uC', 'uD', 'uE', 'uTexWidth', 'uBoxMin', 'uBoxSize', 'uTimeRange',
    'uVelMax', 'uLife', 'uLogScale', 'uTime', 'uMinOpacity', 'uSplatScale', 'uView', 'uFocal', 'uCenter',
    'uViewport']) {
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
  const n = Math.max(...s.chunks.map((c) => c.count));  // buffers sized for the largest chunk
  order = new Uint32Array(n);
  ids = new Uint32Array(n);
  depth = new Float32Array(n);
  keys = new Uint16Array(n);
  counts = new Uint32Array(65536);
  gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, order.byteLength, gl.DYNAMIC_DRAW);
  ['uA', 'uB', 'uC', 'uD', 'uE'].forEach((name, unit) => gl.uniform1i(uniforms[name], unit));
  gl.uniform1i(uniforms.uTexWidth, TEX_WIDTH);
  gl.uniform1f(uniforms.uMinOpacity, s.min_opacity);
}

// The chunk whose Gaussians the next draw call reads.
function bindChunk(index) {
  const chunk = state.chunks[index], info = chunk.info;
  chunk.textures.forEach((tex, unit) => {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
  });
  gl.uniform3fv(uniforms.uBoxMin, info.bbox_min);
  gl.uniform3fv(uniforms.uBoxSize, info.bbox_max.map((v, i) => v - info.bbox_min[i]));
  gl.uniform2fv(uniforms.uTimeRange, info.time);
  gl.uniform1f(uniforms.uVelMax, info.vel_max);
  gl.uniform2fv(uniforms.uLife, info.life);
  gl.uniform2fv(uniforms.uLogScale, info.log_scale);
  state.active = index;
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

// First position in the time-sorted part of a chunk whose time code is >= code.
function firstAtLeast(a, from, to, code) {
  let lo = from, hi = to;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[4 * mid + 3] < code) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// The Gaussians of a chunk visible at instant t, back to front for the camera
// whose viewing direction is `row`. Returns how many, with their indices in `order`.
function visibleBackToFront(chunk, t, row) {
  const info = chunk.info, a = chunk.a, b = chunk.b, reach = chunk.reach;
  const t0 = info.time[0], ts = info.time[1] / 65535, vmax = info.vel_max;
  const bx = info.bbox_min[0], by = info.bbox_min[1], bz = info.bbox_min[2];
  const sx = (info.bbox_max[0] - bx) / 65535, sy = (info.bbox_max[1] - by) / 65535, sz = (info.bbox_max[2] - bz) / 65535;
  // Long-lived Gaussians come first and are always candidates; the others are
  // sorted by time, so only those around t are looked at.
  let from = info.long, to = info.count;
  if (info.reach !== null) {
    from = firstAtLeast(a, info.long, info.count, (t - info.reach - t0) / ts);
    to = firstAtLeast(a, from, info.count, (t + info.reach - t0) / ts + 1);
  }
  let n = 0, lo = Infinity, hi = -Infinity;
  for (let part = 0; part < 2; part++) {
    const start = part ? from : 0, end = part ? to : info.long;
    for (let i = start; i < end; i++) {
      const k = 4 * i, r = reach[i];
      let dt = 0;
      if (r !== Infinity) {
        dt = t - (t0 + a[k + 3] * ts);
        if (dt > r || -dt > r) continue;  // not born yet, or already gone
      }
      const cx = (b[k] - 128) / 127, cy = (b[k + 1] - 128) / 127, cz = (b[k + 2] - 128) / 127;
      const move = vmax * dt;  // the velocity is the square of its code, with its sign
      const z = row[0] * (bx + a[k] * sx + cx * Math.abs(cx) * move)
        + row[1] * (by + a[k + 1] * sy + cy * Math.abs(cy) * move)
        + row[2] * (bz + a[k + 2] * sz + cz * Math.abs(cz) * move) + row[3];
      depth[n] = z;
      ids[n++] = i;
      if (z < lo) lo = z;
      if (z > hi) hi = z;
    }
  }
  const scale = 65535 / Math.max(hi - lo, 1e-9);
  counts.fill(0);
  for (let i = 0; i < n; i++) {
    const key = 65535 - ((depth[i] - lo) * scale | 0);  // far splats first
    keys[i] = key;
    counts[key]++;
  }
  let total = 0;
  for (let i = 0; i < 65536; i++) { const c = counts[i]; counts[i] = total; total += c; }
  for (let i = 0; i < n; i++) order[counts[keys[i]]++] = ids[i];
  return n;
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
    if (chunkReady(chunkOf(Math.floor(clock)))) {  // otherwise wait: that part is still loading
      state.clock = clock;
      state.frame = Math.floor(clock);
    }
  }
  const index = chunkOf(state.frame);
  keepChunks(index);
  showLoading(!chunkReady(index));
  if (!chunkReady(index)) return;
  const chunk = state.chunks[index];
  resize();
  if (state.active !== index) { bindChunk(index); state.dirty = true; }
  if (state.shown !== state.frame) {
    state.shown = state.frame;
    state.dirty = true;
    syncTimeline();
  }
  const cam = currentCamera();
  if (state.dirty) {
    // The scene is shown at the instants it was trained on: whole frames.
    state.drawn = visibleBackToFront(chunk, state.frame, cam.rows[2]);
    gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, order, 0, state.drawn);
    state.dirty = false;
  }
  const count = state.drawn;
  const r = cam.rows;
  gl.uniformMatrix4fv(uniforms.uView, false, new Float32Array([
    r[0][0], r[1][0], r[2][0], 0, r[0][1], r[1][1], r[2][1], 0,
    r[0][2], r[1][2], r[2][2], 0, r[0][3], r[1][3], r[2][3], 1]));
  gl.uniform2fv(uniforms.uFocal, cam.focal);
  gl.uniform2fv(uniforms.uCenter, cam.center);
  gl.uniform2f(uniforms.uViewport, canvas.width, canvas.height);
  gl.uniform1f(uniforms.uSplatScale, state.splatScale);
  gl.uniform1f(uniforms.uTime, state.frame);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.941, 0.957, 0.969, 1);  // --bg of theme.css
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count);
  state.ready = true;

  if (now - state.statsAt > 500) {  // the details panel does not need 60 updates a second
    state.statsAt = now;
    $('d-count').textContent = number(count);
    $('d-render').textContent = `${Math.round(state.fps)} fps`;
  }
}

// ----------------------------------------------------------------- data ---

// The take is divided in chunks (one per segment of the training). Only the
// chunk being shown and the next one are kept: memory does not grow with the
// length of the take.

function chunkOf(frame) {
  const chunks = state.scene.chunks;
  for (let i = chunks.length - 1; i > 0; i--) if (frame >= chunks[i].first) return i;
  return 0;
}

const chunkReady = (index) => Boolean(state.chunks[index] && state.chunks[index].textures);

// Encoded bytes of a chunk. In a single-file export they sit in the page, in
// <script type="application/octet-stream"> elements; in a folder export each
// chunk is a small script that hands them over. Both work from a double click:
// a page opened from disk may load scripts but not fetch data.
const waiting = new Map();
window.mister4dChunk = (index, text) => {
  const done = waiting.get(index);
  if (done) { waiting.delete(index); done(text); }
};

function chunkText(index) {
  if (state.embedded) return Promise.resolve($(`mister4d-chunk-${index}`).textContent);
  const file = state.base + state.scene.chunks[index].file;
  return new Promise((resolve, reject) => {
    const tag = document.createElement('script');
    waiting.set(index, (text) => { tag.remove(); resolve(text); });
    tag.onerror = () => { waiting.delete(index); tag.remove(); reject(new Error(`${file} non trovato`)); };
    tag.src = file;
    document.head.appendChild(tag);
  });
}

async function unpack(text) {
  const binary = atob(text);
  const packed = new Uint8Array(binary.length);
  for (let k = 0; k < binary.length; k++) packed[k] = binary.charCodeAt(k);
  const stream = new Blob([packed]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function loadChunk(index) {
  if (state.chunks[index]) return;
  const info = state.scene.chunks[index], n = info.count;
  const chunk = state.chunks[index] = { info, textures: null };
  chunkText(index).then(unpack).then((raw) => {
    if (state.chunks[index] !== chunk) return;  // dropped while loading
    if (raw.length !== 23 * n) throw new Error(`blocco ${index} incompleto`);
    chunk.a = new Uint16Array(raw.buffer, 0, 4 * n);
    chunk.b = raw.subarray(8 * n, 12 * n);
    // For how many frames around its instant each Gaussian stays visible.
    const opacity = raw.subarray(16 * n, 20 * n), floor = state.scene.min_opacity;
    chunk.reach = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const code = chunk.b[4 * i + 3], peak = opacity[4 * i + 3] / 255;
      if (code === 255) chunk.reach[i] = Infinity;
      else if (peak <= floor) chunk.reach[i] = -1;
      else {
        const sigma = Math.exp(info.life[0] + (info.life[1] - info.life[0]) * code / 254);
        chunk.reach[i] = sigma * Math.sqrt(2 * Math.log(peak / floor));
      }
    }
    chunk.textures = [
      blockTexture(gl.RGBA16UI, gl.RGBA_INTEGER, gl.UNSIGNED_SHORT, chunk.a, 4, n),
      blockTexture(gl.RGBA8UI, gl.RGBA_INTEGER, gl.UNSIGNED_BYTE, chunk.b, 4, n),
      blockTexture(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, raw.subarray(12 * n, 16 * n), 4, n),
      blockTexture(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, opacity, 4, n),
      blockTexture(gl.RGB8, gl.RGB, gl.UNSIGNED_BYTE, raw.subarray(20 * n, 23 * n), 3, n),
    ];
    state.dirty = true;
  }).catch((err) => fail(`Impossibile caricare la scena: ${err.message}`));
}

// Ask for the chunk being shown and, when it is under way, for the one after
// it; free the others.
function keepChunks(index) {
  const chunks = state.scene.chunks, info = chunks[index];
  const wanted = new Set([index]);
  if (chunkReady(index) && chunks.length > 1 && state.frame >= (info.first + info.last) / 2) {
    if (index + 1 < chunks.length) wanted.add(index + 1);
    else if (state.loop) wanted.add(0);
  }
  for (const i of wanted) loadChunk(i);
  state.chunks.forEach((chunk, i) => {
    if (!chunk || wanted.has(i) || (chunks.length <= 2)) return;
    if (chunk.textures) chunk.textures.forEach((tex) => gl.deleteTexture(tex));
    if (state.active === i) state.active = -1;
    delete state.chunks[i];
  });
}

// ------------------------------------------------------------ interface ---

function showLoading(busy) {
  const s = state.scene;
  $('loading').hidden = !busy;
  if (busy === state.busy) return;
  state.busy = busy;
  $('status').textContent = busy ? 'Caricamento…'
    : (s.num_frames > 1
      ? `${s.num_frames} frame · ${number(s.fps, s.fps % 1 ? 2 : 0)} fps · ${number(s.num_frames / s.fps, 2)} s`
      : 'Scena statica');
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
  state.frame = f;
  state.clock = f;
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
  // The current point of view as a key view of a camera move (mister4d render --path).
  $('copyView').addEventListener('click', () => {
    const cam = currentCamera(), r = cam.rows;
    const eye = [0, 1, 2].map((i) => -(r[0][i] * r[0][3] + r[1][i] * r[1][3] + r[2][i] * r[2][3]));
    const reach = Math.hypot(...sub(state.target, eye)) || state.dist;
    const round = (v, digits) => Number(v.toFixed(digits));
    const text = JSON.stringify({
      time: round(state.frame / s.fps, 3),
      eye: eye.map((v) => round(v, 4)),
      target: eye.map((v, i) => round(v + r[2][i] * reach, 4)),
      fov: round(2 * Math.atan(stage().h / 2 / cam.focal[1]) * 180 / Math.PI, 2),
    });
    const done = () => {
      $('copied').textContent = 'Vista copiata';
      setTimeout(() => { $('copied').textContent = ''; }, 2500);
    };
    const byHand = () => window.prompt('Copia questa vista:', text);
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, byHand);
    else byHand();
  });
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

function start(scene, embedded) {
  try {
    if (scene.format !== 'mister4d-web' || scene.version !== 3) throw new Error('il file non è un export mister4d (versione 3)');
    if (typeof DecompressionStream === 'undefined') throw new Error('questo browser è troppo vecchio per aprire la scena');
    state.scene = scene;
    state.embedded = embedded;
    initGL();
    setupScene();
    setupControls();
    homeView();
    showLoading(true);
    syncTimeline();
    requestAnimationFrame(draw);
  } catch (err) {
    fail(`Impossibile caricare la scena: ${err.message}`);
    throw err;
  }
}

function main() {
  if (!gl) return fail('Questo browser non supporta WebGL2: prova con una versione recente di Chrome, Edge o Firefox.');
  const embedded = $('mister4d-scene');
  if (embedded) return start(JSON.parse(embedded.textContent), true);
  const url = new URLSearchParams(location.search).get('scene');
  if (url) {  // a scene somewhere else, from a web server
    state.base = url.includes('/') ? url.slice(0, url.lastIndexOf('/') + 1) : '';
    return fetch(url).then((response) => response.json()).then((scene) => start(scene, false))
      .catch((err) => fail(`Impossibile caricare la scena: ${err.message}`));
  }
  // A folder export: scene.js, next to this page, hands the description over.
  window.mister4dScene = (scene) => start(scene, false);
  window.addEventListener('load', () => {
    if (!state.scene) fail('Accanto a questa pagina non c\'è nessuna scena da mostrare.');
  });
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
