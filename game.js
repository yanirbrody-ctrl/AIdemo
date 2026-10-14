// Head Dodge — head-controlled browser game.
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
const HEAD_RANGE = 0.15;    // nose shift (fraction of camera width) that reaches the screen edge
const SMOOTH_TIME = 0.15;   // seconds; higher = smoother but laggier
const FACE_LOST_MS = 600;
const KEY_SPEED = 0.9;      // screen widths per second

// Landmark indices
const NOSE = 1;

const $ = (id) => document.getElementById(id);
const canvas = $("game");
const ctx = canvas.getContext("2d");
const video = $("video");
const preview = $("preview");
const pctx = preview.getContext("2d");
const overlay = $("overlay");
const titleEl = $("title");
const messageEl = $("message");
const startBtn = $("start-btn");
const keysBtn = $("keys-btn");
const statusEl = $("status");
const scoreEl = $("score");
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
const keys = { left: false, right: false };
window.addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft" || e.key === "a" || e.key === "A") keys.left = true;
  if (e.key === "ArrowRight" || e.key === "d" || e.key === "D") keys.right = true;
  if ((e.key === " " || e.key === "Enter") && state === "over") {
    e.preventDefault();
    startRound();
  }
});
window.addEventListener("keyup", (e) => {
  if (e.key === "ArrowLeft" || e.key === "a" || e.key === "A") keys.left = false;
  if (e.key === "ArrowRight" || e.key === "d" || e.key === "D") keys.right = false;
});

// ---------- Face tracking ----------
let landmarker = null;
let cameraOn = false;
let lastVideoTime = -1;
let face = null;            // { noseX, noseY } of latest detection
let lastFaceAt = 0;
let center = { noseX: 0.5 };
let headTarget = 0.5;       // desired player x (0..1) from head

function cameraErrorMessage(err) {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia)
    return "Camera needs HTTPS or localhost. Using arrow keys instead.";
  switch (err?.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Camera permission was denied. Using arrow keys instead.";
    case "NotReadableError":
    case "AbortError":
      return "Camera is busy (used by another app?). Using arrow keys instead.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No camera found. Using arrow keys instead.";
    default:
      return "Couldn't start the camera. Using arrow keys instead.";
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
    numFaces: 1,
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
  const lm = res.faceLandmarks?.[0];
  if (!lm) { face = null; return; }

  face = { noseX: lm[NOSE].x, noseY: lm[NOSE].y };
  lastFaceAt = now;

  // Linear map: calibrated center -> screen center, HEAD_RANGE away -> edge.
  // Camera image is not mirrored, so moving your head to your right lowers nose x.
  const offset = -(face.noseX - center.noseX) / HEAD_RANGE;
  headTarget = clamp(0.5 + offset * 0.5, 0, 1);
}

function drawPreview() {
  if (!cameraOn || video.readyState < 2) return;
  const w = preview.width, h = preview.height;
  pctx.save();
  pctx.translate(w, 0);
  pctx.scale(-1, 1);
  pctx.drawImage(video, 0, 0, w, h);
  pctx.restore();
  if (face) {
    pctx.fillStyle = "#ff3b5c";
    pctx.beginPath();
    pctx.arc((1 - face.noseX) * w, face.noseY * h, 5, 0, Math.PI * 2);
    pctx.fill();
  }
}

const faceVisible = (now) => cameraOn && face && now - lastFaceAt < FACE_LOST_MS;

// ---------- Calibration ----------
let calib = null; // { started, samples: [], firstFaceAt }

function beginCalibration() {
  state = "calibrating";
  calib = { started: performance.now(), samples: [], firstFaceAt: 0 };
  showOverlay("Calibrating", "Look straight at the camera…", false);
}

function updateCalibration(now) {
  if (face && now - lastFaceAt < 100) {
    if (!calib.firstFaceAt) calib.firstFaceAt = now;
    calib.samples.push(face.noseX);
  }
  if (calib.firstFaceAt) {
    const left = Math.max(0, CALIBRATION_MS - (now - calib.firstFaceAt));
    messageEl.textContent = `Look straight at the camera… ${Math.ceil(left / 1000)}`;
    if (left === 0 && calib.samples.length >= 5) {
      const n = calib.samples.length;
      center = { noseX: calib.samples.reduce((s, x) => s + x, 0) / n };
      startRound();
    }
  } else if (now - calib.started > CALIBRATION_TIMEOUT_MS) {
    showStatus("No face detected. Using arrow keys — tracking will kick in if your face appears.", 5000);
    startRound();
  }
}

// ---------- Game ----------
let state = "menu"; // menu | loading | calibrating | playing | over
let player, obstacles, stars, score, elapsed, spawnTimer, starTimer;

function startRound() {
  hideOverlay();
  state = "playing";
  player = { x: 0.5, r: 22 };
  obstacles = [];
  stars = [];
  score = 0;
  elapsed = 0;
  spawnTimer = 0;
  starTimer = 2;
}

function speedFactor() {
  return 1 + elapsed / 20; // gets faster over time
}

function spawnObstacle() {
  const w = 40 + Math.random() * 70;
  obstacles.push({ x: Math.random() * (W - w), y: -40, w, h: 26 + Math.random() * 20 });
}

function spawnStar() {
  stars.push({ x: 20 + Math.random() * (W - 40), y: -20, r: 14 });
}

function update(dt, now) {
  elapsed += dt;
  const sf = speedFactor();

  // Input: keys override head while held; otherwise follow head if visible.
  const dir = (keys.right ? 1 : 0) - (keys.left ? 1 : 0);
  if (dir) {
    player.x = clamp(player.x + dir * KEY_SPEED * dt, 0, 1);
  } else if (faceVisible(now)) {
    player.x += (headTarget - player.x) * (1 - Math.exp(-dt / SMOOTH_TIME));
  }

  // Face-lost notice (only when the camera is actually on).
  if (cameraOn) {
    if (!faceVisible(now)) showStatus("No face detected — use arrow keys or look at the camera.");
    else if (statusEl.textContent.startsWith("No face detected —")) hideStatus();
  }

  // Spawning
  spawnTimer -= dt;
  if (spawnTimer <= 0) {
    spawnObstacle();
    spawnTimer = Math.max(0.25, 0.9 / sf);
  }
  starTimer -= dt;
  if (starTimer <= 0) {
    spawnStar();
    starTimer = 2.5 + Math.random() * 2;
  }

  const fall = 220 * sf * dt;
  const px = player.x * (W - 2 * player.r) + player.r;
  const py = H - 50;

  for (const o of obstacles) {
    o.y += fall;
    if (circleRect(px, py, player.r * 0.85, o)) return gameOver();
  }
  obstacles = obstacles.filter((o) => o.y < H + 50);

  for (const s of stars) {
    s.y += fall * 0.8;
    if (Math.hypot(px - s.x, py - s.y) < player.r + s.r) {
      s.taken = true;
      score += 50;
    }
  }
  stars = stars.filter((s) => !s.taken && s.y < H + 30);

  score += dt * 10;
  scoreEl.textContent = Math.floor(score);
}

function gameOver() {
  state = "over";
  hideStatus();
  const final = Math.floor(score);
  if (final > best) {
    best = final;
    saveBest(best);
    bestEl.textContent = best;
  }
  showOverlay("Game over", `Score: ${final} · Best: ${best}`, true, "Play again");
}

function render() {
  ctx.fillStyle = "#0d1020";
  ctx.fillRect(0, 0, W, H);

  if (!player) return;

  ctx.fillStyle = "#ff4d6d";
  for (const o of obstacles) ctx.fillRect(o.x, o.y, o.w, o.h);

  for (const s of stars) drawStar(s.x, s.y, s.r);

  const px = player.x * (W - 2 * player.r) + player.r;
  const py = H - 50;
  ctx.fillStyle = "#4fd1ff";
  ctx.beginPath();
  ctx.arc(px, py, player.r, 0, Math.PI * 2);
  ctx.fill();
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

// ---------- Overlay ----------
function showOverlay(title, msg, showButtons, startLabel = "Start") {
  titleEl.textContent = title;
  messageEl.textContent = msg;
  startBtn.textContent = startLabel;
  startBtn.hidden = !showButtons;
  keysBtn.hidden = !showButtons || state === "over";
  overlay.hidden = false;
}
function hideOverlay() {
  overlay.hidden = true;
}

startBtn.addEventListener("click", async () => {
  if (state === "over") return startRound();
  if (state !== "menu") return;
  state = "loading";
  showOverlay("Starting…", "Allow camera access and loading face tracking…", false);

  try {
    await startCamera();
  } catch (err) {
    showStatus(cameraErrorMessage(err), 6000);
    return startRound();
  }

  try {
    landmarker = await createLandmarker();
  } catch (err) {
    console.error(err);
    showStatus("Couldn't load face tracking. Using arrow keys instead.", 6000);
    return startRound();
  }

  cameraOn = true;
  $("preview-wrap").hidden = false;
  beginCalibration();
});

keysBtn.addEventListener("click", () => {
  if (state === "menu") startRound();
});

// ---------- Main loop ----------
let lastT = performance.now();
function frame(now) {
  const dt = Math.min(0.05, (now - lastT) / 1000);
  lastT = now;

  detect(now);
  drawPreview();

  if (state === "calibrating") updateCalibration(now);
  else if (state === "playing") update(dt, now);

  render();
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
