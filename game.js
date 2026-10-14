// Head Dodge — head-controlled browser game, 1 or 2 players.
// Face tracking runs locally via MediaPipe Face Landmarker (pinned version).
import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

// Tuning
const CALIBRATION_MS = 2000;
const CALIBRATION_TIMEOUT_MS = 8000;
const HEAD_RANGE = 0.15;    // nose shift (fraction of camera width) that reaches the lane edge
const SMOOTH_TIME = 0.15;   // seconds; higher = smoother but laggier
const FACE_LOST_MS = 600;
const KEY_SPEED = 0.9;      // lane widths per second
const CRASH_MS = 1000;

// Landmark indices
const NOSE = 1;

// Per-mode player setup. Face positions are stored mirrored (0 = left of the
// mirrored preview), so "left player" is the face on the left of the preview.
const PLAYER_SETUP = {
  1: [{ name: "", color: "#4fd1ff", left: ["KeyA", "ArrowLeft"], right: ["KeyD", "ArrowRight"], keyLabel: "← / →", center: 0.5 }],
  2: [
    { name: "Left", color: "#4fd1ff", left: ["KeyA"], right: ["KeyD"], keyLabel: "A / D", center: 0.3 },
    { name: "Right", color: "#ff8de0", left: ["ArrowLeft"], right: ["ArrowRight"], keyLabel: "← / →", center: 0.7 },
  ],
};

const $ = (id) => document.getElementById(id);
const canvas = $("game");
const ctx = canvas.getContext("2d");
const video = $("video");
const preview = $("preview");
const pctx = preview.getContext("2d");
const overlay = $("overlay");
const titleEl = $("title");
const messageEl = $("message");
const modesEl = $("modes");
const startBtn = $("start-btn");
const keysBtn = $("keys-btn");
const menuBtn = $("menu-btn");
const hintEl = $("hint");
const statusEl = $("status");
const bestEl = $("best");

// ---------- High score ----------
function loadBest() {
  try { return Number(localStorage.getItem("headDodgeBest")) || 0; } catch { return 0; }
}
function saveBest(v) {
  try { localStorage.setItem("headDodgeBest", String(v)); } catch { /* ignore */ }
}
let best = loadBest();
bestEl.textContent = best;

// ---------- Layout ----------
let W = 0, H = 0;
function resize() {
  const dpr = window.devicePixelRatio || 1;
  W = window.innerWidth;
  H = window.innerHeight;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener("resize", resize);
resize();

// ---------- Status messages ----------
let statusTimer = 0;
function showStatus(text, ms = 0) {
  statusEl.textContent = text;
  statusEl.hidden = false;
  clearTimeout(statusTimer);
  if (ms) statusTimer = setTimeout(() => (statusEl.hidden = true), ms);
}
function hideStatus() {
  clearTimeout(statusTimer);
  statusEl.hidden = true;
}

// ---------- Keyboard ----------
const pressed = new Set();
window.addEventListener("keydown", (e) => {
  pressed.add(e.code);
  if (e.code.startsWith("Arrow")) e.preventDefault();
  if ((e.key === " " || e.key === "Enter") && state === "over") {
    e.preventDefault();
    startRound();
  }
});
window.addEventListener("keyup", (e) => pressed.delete(e.code));
window.addEventListener("blur", () => pressed.clear());

// ---------- Players ----------
let players = [];

function makePlayers(m) {
  return PLAYER_SETUP[m].map((s, idx) => ({
    ...s,
    idx,
    center: s.center,     // calibrated mirrored nose x
    face: null,           // { mx, my } mirrored nose position of latest detection
    lastFaceAt: 0,
    headTarget: 0.5,      // desired x (0..1) within the lane
    faceCrop: null,       // { cx, cy, size } square around the face, video pixels, smoothed
    x: 0.5,
    obstacles: [],
    stars: [],
    score: 0,
    crack: null,          // { t, lines } once hit
    out: false,
  }));
}

function lane(p) {
  const w = W / players.length;
  return { x0: p.idx * w, w };
}
function playerRadius(w) {
  return clamp(w * 0.08, 16, 34);
}
function playerPos(p) {
  const { x0, w } = lane(p);
  const r = playerRadius(w);
  return [x0 + p.x * (w - 2 * r) + r, H - 60, r];
}

// ---------- Face tracking ----------
let landmarker = null;
let cameraReady = false;    // stream + model loaded
let cameraOn = false;       // tracking used for the current game
let lastVideoTime = -1;

function cameraErrorMessage(err) {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia)
    return "Camera needs HTTPS or localhost. Using the keyboard instead.";
  switch (err?.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Camera permission was denied. Using the keyboard instead.";
    case "NotReadableError":
    case "AbortError":
      return "Camera is busy (used by another app?). Using the keyboard instead.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No camera found. Using the keyboard instead.";
    default:
      return "Couldn't start the camera. Using the keyboard instead.";
  }
}

async function startCamera() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    throw new Error("insecure");
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
}

async function createLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numFaces: 2,
  });
  try {
    return await FaceLandmarker.createFromOptions(fileset, opts("GPU"));
  } catch {
    return await FaceLandmarker.createFromOptions(fileset, opts("CPU"));
  }
}

function detect(now) {
  if (!cameraOn || !landmarker || video.readyState < 2) return;
  if (video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  const res = landmarker.detectForVideo(video, now);
  const faces = (res.faceLandmarks || []).map((lm) => ({
    lm,
    mx: 1 - lm[NOSE].x, // mirrored, so left of preview = left player
    my: lm[NOSE].y,
    box: faceBox(lm),
  }));

  for (const p of players) p.face = null;
  if (!faces.length) return;

  // Largest faces first (closest people), then map to players.
  faces.sort((a, b) => b.box.size - a.box.size);
  if (players.length === 1) {
    setFace(players[0], faces[0], now);
  } else if (faces.length >= 2) {
    const [a, b] = faces.slice(0, 2).sort((f, g) => f.mx - g.mx);
    setFace(players[0], a, now);
    setFace(players[1], b, now);
  } else {
    const split = (players[0].center + players[1].center) / 2;
    setFace(players[faces[0].mx < split ? 0 : 1], faces[0], now);
  }
}

function setFace(p, f, now) {
  p.face = { mx: f.mx, my: f.my };
  p.lastFaceAt = now;
  // Linear map: calibrated center -> lane center, HEAD_RANGE away -> lane edge.
  p.headTarget = clamp(0.5 + ((f.mx - p.center) / HEAD_RANGE) * 0.5, 0, 1);
  // Smooth the crop so the face avatar doesn't jitter.
  if (!p.faceCrop) p.faceCrop = { ...f.box };
  else for (const k in f.box) p.faceCrop[k] += (f.box[k] - p.faceCrop[k]) * 0.35;
}

// Square crop around the face landmarks, in video pixels.
function faceBox(lm) {
  let minX = 1, maxX = 0, minY = 1, maxY = 0;
  for (const p of lm) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  const vw = video.videoWidth, vh = video.videoHeight;
  return {
    cx: ((minX + maxX) / 2) * vw,
    cy: ((minY + maxY) / 2) * vh,
    size: Math.max((maxX - minX) * vw, (maxY - minY) * vh) * 1.25,
  };
}

function drawPreview() {
  if (!cameraOn || video.readyState < 2) return;
  const w = preview.width, h = preview.height;
  pctx.save();
  pctx.translate(w, 0);
  pctx.scale(-1, 1);
  pctx.drawImage(video, 0, 0, w, h);
  pctx.restore();
  for (const p of players) {
    if (!p.face) continue;
    pctx.fillStyle = p.color;
    pctx.beginPath();
    pctx.arc(p.face.mx * w, p.face.my * h, 5, 0, Math.PI * 2);
    pctx.fill();
  }
}

const faceVisible = (p, now) => cameraOn && p.face && now - p.lastFaceAt < FACE_LOST_MS;

// ---------- Calibration ----------
let calib = null; // { started, samples: [[mx...] per player], allSeenAt }

function beginCalibration() {
  state = "calibrating";
  calib = { started: performance.now(), samples: players.map(() => []), allSeenAt: 0 };
  setOverlay({ title: "Calibrating", msg: calibrationText() });
}

function calibrationText(secondsLeft) {
  const who = players.length === 1 ? "Look" : "Both players look";
  let text = `${who} straight at the camera…` + (secondsLeft ? ` ${secondsLeft}` : "");
  if (players.length > 1) {
    const now = performance.now();
    text += "\n" + players.map((p) => `${p.name}: ${now - p.lastFaceAt < 200 ? "✓" : "waiting"}`).join(" · ");
  }
  return text;
}

function updateCalibration(now) {
  let all = true;
  players.forEach((p, i) => {
    if (p.face && now - p.lastFaceAt < 100) calib.samples[i].push(p.face.mx);
    else all = false;
  });
  if (all && !calib.allSeenAt) calib.allSeenAt = now;

  if (calib.allSeenAt) {
    const left = Math.max(0, CALIBRATION_MS - (now - calib.allSeenAt));
    messageEl.textContent = calibrationText(Math.ceil(left / 1000));
    if (left === 0) finishCalibration();
  } else {
    messageEl.textContent = calibrationText();
    if (now - calib.started > CALIBRATION_TIMEOUT_MS) {
      const missing = finishCalibration();
      const who = players.length === 1 ? "your face" : `a face for: ${missing.map((p) => p.name).join(", ")}`;
      showStatus(`Couldn't find ${who}. Use the keyboard — tracking kicks in when a face appears.`, 6000);
    }
  }
}

// Set each player's center from their samples; returns players with too few.
function finishCalibration() {
  const missing = [];
  players.forEach((p, i) => {
    const s = calib.samples[i];
    if (s.length >= 5) p.center = s.reduce((a, b) => a + b, 0) / s.length;
    else missing.push(p);
  });
  startRound();
  return missing;
}

// ---------- Game ----------
let state = "menu"; // menu | loading | calibrating | playing | over
let elapsed = 0, spawnTimer = 0, starTimer = 0;

function startRound() {
  hideOverlay();
  state = "playing";
  for (const p of players) {
    Object.assign(p, { x: 0.5, obstacles: [], stars: [], score: 0, crack: null, out: false });
  }
  elapsed = 0;
  spawnTimer = 0;
  starTimer = 2;
}

function speedFactor() {
  return 1 + elapsed / 20; // gets faster over time
}

// Obstacles and stars use lane-relative x (0..1), and every lane gets an
// identical copy so both players face exactly the same drops.
function spawnToAll(make) {
  const item = make(lane(players[0]).w);
  for (const p of players) if (!p.crack) (item.r ? p.stars : p.obstacles).push({ ...item });
}

function newObstacle(laneW) {
  const w = (40 + Math.random() * 70) * Math.min(1, laneW / 500);
  const fw = w / laneW;
  return { fx: Math.random() * (1 - fw), fw, y: -40, h: 26 + Math.random() * 20 };
}

function newStar(laneW) {
  const m = 20 / laneW;
  return { fx: m + Math.random() * (1 - 2 * m), y: -20, r: 14 };
}

function update(dt, now) {
  elapsed += dt;
  const sf = speedFactor();

  spawnTimer -= dt;
  if (spawnTimer <= 0) {
    spawnToAll(newObstacle);
    spawnTimer = Math.max(0.25, 0.9 / sf);
  }
  starTimer -= dt;
  if (starTimer <= 0) {
    spawnToAll(newStar);
    starTimer = 2.5 + Math.random() * 2;
  }

  const fall = 220 * sf * dt;
  for (const p of players) {
    if (p.crack) {
      // Frozen: let the crack play out, then mark OUT.
      p.crack.t += dt;
      if (p.crack.t * 1000 >= CRASH_MS) p.out = true;
      continue;
    }
    updatePlayer(p, dt, now, fall);
  }

  if (players.every((p) => p.out)) endGame();
}

function updatePlayer(p, dt, now, fall) {
  // Input: keys override head while held; otherwise follow head if visible.
  const dir = (p.right.some((k) => pressed.has(k)) ? 1 : 0) - (p.left.some((k) => pressed.has(k)) ? 1 : 0);
  if (dir) {
    p.x = clamp(p.x + dir * KEY_SPEED * dt, 0, 1);
  } else if (faceVisible(p, now)) {
    p.x += (p.headTarget - p.x) * (1 - Math.exp(-dt / SMOOTH_TIME));
  }

  const { x0, w } = lane(p);
  const [px, py, r] = playerPos(p);

  for (const o of p.obstacles) {
    o.y += fall;
    const rect = { x: x0 + o.fx * w, y: o.y, w: o.fw * w, h: o.h };
    if (circleRect(px, py, r * 0.85, rect)) return startCrash(p, px, py, r, rect);
  }
  p.obstacles = p.obstacles.filter((o) => o.y < H + 50);

  for (const s of p.stars) {
    s.y += fall * 0.8;
    if (Math.hypot(px - (x0 + s.fx * w), py - s.y) < r + s.r) {
      s.taken = true;
      p.score += 50;
    }
  }
  p.stars = p.stars.filter((s) => !s.taken && s.y < H + 30);

  p.score += dt * 10;
}

// Freeze this player and grow cracks out from the impact point.
function startCrash(p, px, py, r, o) {
  const dx = clamp(px, o.x, o.x + o.w) - px;
  const dy = clamp(py, o.y, o.y + o.h) - py;
  const d = Math.hypot(dx, dy) || 1;
  const ix = (dx / d) * r * 0.45, iy = (dy / d) * r * 0.45;

  const lines = [];
  const n = 9;
  for (let i = 0; i < n; i++) {
    let a = (i / n) * Math.PI * 2 + Math.random() * 0.5;
    let x = ix, y = iy;
    const pts = [{ x, y }];
    while (Math.hypot(x, y) < r * 1.1) {
      a += (Math.random() - 0.5) * 0.7;
      const step = r * (0.15 + Math.random() * 0.2);
      x += Math.cos(a) * step;
      y += Math.sin(a) * step;
      pts.push({ x, y });
    }
    lines.push(pts);
  }
  // Ring cracks connecting neighbouring radial lines.
  for (const k of [1, 2]) {
    for (let i = 0; i < n; i++) {
      const a = lines[i][k], b = lines[(i + 1) % n][k];
      if (a && b && Math.random() < 0.7) lines.push([a, b]);
    }
  }
  p.crack = { t: 0, lines };
}

function endGame() {
  state = "over";
  hideStatus();
  const scores = players.map((p) => Math.floor(p.score));
  const top = Math.max(...scores);
  if (top > best) {
    best = top;
    saveBest(best);
    bestEl.textContent = best;
  }

  if (players.length === 1) {
    setOverlay({ title: "Game over", msg: `Score: ${scores[0]} · Best: ${best}`, start: "Play again", menu: true });
  } else {
    const [a, b] = scores;
    const title = a === b ? "It's a tie!" : `${players[a > b ? 0 : 1].name} player wins!`;
    const msg = players.map((p, i) => `${p.name}: ${scores[i]}`).join(" · ") + ` · Best: ${best}`;
    setOverlay({ title, msg, start: "Play again", menu: true });
  }
}

// ---------- Rendering ----------
function render(now) {
  ctx.fillStyle = "#0d1020";
  ctx.fillRect(0, 0, W, H);
  if (!players.length || state === "menu") return;

  for (const p of players) {
    const { x0, w } = lane(p);
    ctx.save();
    ctx.beginPath();
    ctx.rect(x0, 0, w, H);
    ctx.clip();

    ctx.fillStyle = "#ff4d6d";
    for (const o of p.obstacles) ctx.fillRect(x0 + o.fx * w, o.y, o.fw * w, o.h);
    for (const s of p.stars) drawStar(x0 + s.fx * w, s.y, s.r);

    let [px, py, r] = playerPos(p);
    const c = p.crack;
    if (c && c.t < 0.3) px += Math.sin(c.t * 90) * 6 * (1 - c.t / 0.3); // shake
    drawPlayer(p, px, py, r);

    drawLaneText(p, x0, w, now);
    ctx.restore();
  }

  if (players.length > 1) {
    ctx.fillStyle = "rgba(255,255,255,0.25)";
    ctx.fillRect(W / 2 - 1, 0, 2, H);
  }
}

function drawLaneText(p, x0, w, now) {
  const cx = x0 + w / 2;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.font = "bold 22px system-ui, sans-serif";
  ctx.fillStyle = p.color;
  ctx.fillText(`${p.name ? p.name + " " : "Score "}${Math.floor(p.score)}`, cx, 14);

  if (p.out) {
    ctx.fillStyle = "rgba(5,7,18,0.6)";
    ctx.fillRect(x0, 0, w, H);
    ctx.textBaseline = "middle";
    ctx.font = `bold ${Math.min(120, w / 4)}px system-ui, sans-serif`;
    ctx.fillStyle = "#ff4d6d";
    ctx.fillText("OUT", cx, H / 2);
  } else if (cameraOn && state === "playing" && !p.crack && !faceVisible(p, now)) {
    ctx.textBaseline = "middle";
    ctx.font = "600 18px system-ui, sans-serif";
    const text = `No face detected — use ${p.keyLabel}`;
    const tw = Math.min(w - 20, ctx.measureText(text).width + 24);
    ctx.fillStyle = "rgba(255,170,0,0.92)";
    ctx.fillRect(cx - tw / 2, 60, tw, 36);
    ctx.fillStyle = "#1a1200";
    ctx.fillText(text, cx, 78, w - 32);
  }
}

function drawPlayer(p, px, py, r) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(px, py, r, 0, Math.PI * 2);
  ctx.clip();

  if (cameraOn && p.faceCrop && video.readyState >= 2) {
    // Live face, mirrored like the preview.
    const vw = video.videoWidth, vh = video.videoHeight;
    const size = Math.min(p.faceCrop.size, vw, vh);
    const sx = clamp(p.faceCrop.cx - size / 2, 0, vw - size);
    const sy = clamp(p.faceCrop.cy - size / 2, 0, vh - size);
    ctx.save();
    ctx.translate(px, py);
    ctx.scale(-1, 1);
    ctx.drawImage(video, sx, sy, size, size, -r, -r, 2 * r, 2 * r);
    ctx.restore();
  } else {
    drawSmiley(px, py, r);
  }

  if (p.crack) drawCracks(p.crack, px, py, r);
  ctx.restore();

  // Glowing outline (turns red on impact).
  ctx.save();
  const color = p.crack ? "#ff4d6d" : p.color;
  ctx.strokeStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = 24;
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.arc(px, py, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.stroke(); // second pass makes the glow stronger
  ctx.restore();
}

function drawSmiley(x, y, r) {
  ctx.fillStyle = "#ffd84d";
  ctx.fillRect(x - r, y - r, 2 * r, 2 * r);
  ctx.fillStyle = "#2a2140";
  ctx.beginPath();
  ctx.arc(x - r * 0.35, y - r * 0.25, r * 0.12, 0, Math.PI * 2);
  ctx.arc(x + r * 0.35, y - r * 0.25, r * 0.12, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#2a2140";
  ctx.lineWidth = r * 0.1;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.arc(x, y + r * 0.05, r * 0.5, 0.15 * Math.PI, 0.85 * Math.PI);
  ctx.stroke();
}

function drawCracks(crack, px, py, r) {
  const grow = Math.min(1, crack.t / 0.25);
  // Brief white flash on impact.
  const flash = Math.max(0, 0.6 * (1 - crack.t / 0.3));
  if (flash) {
    ctx.fillStyle = `rgba(255,255,255,${flash})`;
    ctx.fillRect(px - r, py - r, 2 * r, 2 * r);
  }
  ctx.lineJoin = "round";
  for (const [width, style] of [[3, "rgba(0,0,0,0.45)"], [1.4, "rgba(255,255,255,0.95)"]]) {
    ctx.lineWidth = width;
    ctx.strokeStyle = style;
    ctx.beginPath();
    for (const pts of crack.lines) {
      const count = Math.max(2, Math.ceil(pts.length * grow));
      ctx.moveTo(px + pts[0].x, py + pts[0].y);
      for (let i = 1; i < Math.min(count, pts.length); i++) ctx.lineTo(px + pts[i].x, py + pts[i].y);
    }
    ctx.stroke();
  }
}

function drawStar(x, y, r) {
  ctx.fillStyle = "#ffd84d";
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + (i * Math.PI) / 5;
    const rr = i % 2 ? r * 0.45 : r;
    ctx.lineTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr);
  }
  ctx.closePath();
  ctx.fill();
}

// ---------- Overlay & menu ----------
let selectedMode = 1;

function setOverlay({ title, msg, start = null, keys = false, menu = false, modes = false, hint = "" }) {
  titleEl.textContent = title;
  messageEl.textContent = msg;
  startBtn.hidden = !start;
  if (start) startBtn.textContent = start;
  keysBtn.hidden = !keys;
  menuBtn.hidden = !menu;
  modesEl.hidden = !modes;
  hintEl.hidden = !hint;
  hintEl.textContent = hint;
  overlay.hidden = false;
}
function hideOverlay() {
  overlay.hidden = true;
}

function showMenu() {
  state = "menu";
  hideStatus();
  for (const b of modesEl.children) b.classList.toggle("selected", Number(b.dataset.mode) === selectedMode);
  setOverlay({
    title: "Head Dodge",
    msg: selectedMode === 1
      ? "Move your head left and right to dodge the falling blocks. Grab stars for bonus points."
      : "Sit side by side in front of the camera. Each player steers their own half — same blocks, last one standing wins.",
    start: "Start",
    keys: true,
    modes: true,
    hint: selectedMode === 1
      ? "Arrow keys (or A / D) always work too."
      : "Left player: A / D · Right player: ← / →. Keys always work too.",
  });
}

async function startGame(m, useCamera) {
  players = makePlayers(m);

  if (!useCamera) {
    cameraOn = false;
    $("preview-wrap").hidden = true;
    return startRound();
  }

  if (!cameraReady) {
    state = "loading";
    setOverlay({ title: "Starting…", msg: "Allow camera access and loading face tracking…" });
    try {
      if (!video.srcObject) await startCamera();
    } catch (err) {
      showStatus(cameraErrorMessage(err), 6000);
      return startGame(m, false);
    }
    try {
      landmarker = await createLandmarker();
    } catch (err) {
      console.error(err);
      showStatus("Couldn't load face tracking. Using the keyboard instead.", 6000);
      return startGame(m, false);
    }
    cameraReady = true;
  }

  cameraOn = true;
  $("preview-wrap").hidden = false;
  beginCalibration();
}

for (const b of modesEl.children) {
  b.addEventListener("click", () => {
    selectedMode = Number(b.dataset.mode);
    showMenu();
  });
}

startBtn.addEventListener("click", () => {
  if (state === "over") return startRound();
  if (state === "menu") startGame(selectedMode, true);
});
keysBtn.addEventListener("click", () => {
  if (state === "menu") startGame(selectedMode, false);
});
menuBtn.addEventListener("click", showMenu);

showMenu();

// ---------- Main loop ----------
let lastT = performance.now();
function frame(now) {
  const dt = Math.min(0.05, (now - lastT) / 1000);
  lastT = now;

  detect(now);
  drawPreview();

  if (state === "calibrating") updateCalibration(now);
  else if (state === "playing") update(dt, now);

  render(now);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// ---------- Helpers ----------
function clamp(v, a, b) {
  return Math.max(a, Math.min(b, v));
}
function circleRect(cx, cy, r, o) {
  const nx = clamp(cx, o.x, o.x + o.w);
  const ny = clamp(cy, o.y, o.y + o.h);
  return (cx - nx) ** 2 + (cy - ny) ** 2 < r * r;
}
