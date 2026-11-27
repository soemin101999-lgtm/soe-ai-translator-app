import express from "express";
import cors from "cors";
import multer from "multer";
import { GoogleGenAI } from "@google/genai";
import { MsEdgeTTS, OUTPUT_FORMAT } from "msedge-tts";
import ffmpegPath from "ffmpeg-static";
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";

const app = express();
const PORT = process.env.PORT || 3000;
app.set("trust proxy", 1);

app.use(cors());
app.use(express.json({ limit: "10mb" }));

const apiKey = process.env.GEMINI_API_KEY;
const groqKey = process.env.GROQ_API_KEY;
const APP_SECRET = process.env.APP_SECRET || ""; // ထားရင် app ကနေ x-app-secret header ပို့ရမည်
const ai = apiKey ? new GoogleGenAI({ apiKey, httpOptions: { apiVersion: "v1beta" } }) : null;

/* ---------------- settings ---------------- */
const TEXT_MODELS = [
  process.env.GEMINI_MODEL,
  "gemini-3.8-flash",
  "gemini-3.7-flash",
  "gemini-3.6-flash",
  "gemini-2.5-flash"
].filter(Boolean);

const TTS_MODELS = [
  process.env.GEMINI_TTS_MODEL,
  "gemini-3.8-flash-tts",
  "gemini-3.1-flash-tts-preview",
  "gemini-2.5-flash-preview-tts"
].filter(Boolean);

const TARGETS = {
  my: { label: "Myanmar (Burmese)", mm: "my", edge: { male: "my-MM-ThihaNeural", female: "my-MM-NilarNeural" } },
  th: { label: "Thai", mm: "th", edge: { male: "th-TH-NiwatNeural", female: "th-TH-PremwadeeNeural" } },
  en: { label: "English", mm: "en", edge: { male: "en-US-GuyNeural", female: "en-US-JennyNeural" } },
  zh: { label: "Simplified Chinese (Mandarin)", mm: "zh-CN", edge: { male: "zh-CN-YunxiNeural", female: "zh-CN-XiaoxiaoNeural" } }
};
const getTarget = (code) => TARGETS[code] || TARGETS.my;

// ၁ စက္ကန့်အတွင်း ပြောနိုင်တဲ့ စာလုံးအရေအတွက် (ခန့်မှန်း) - ဘာသာပြန်ကို slot နဲ့ ကိုက်စေရန်
const CHARS_PER_SEC = { my: 11, th: 11, en: 15, zh: 4.5 };

const VOICES = { male: "Charon", female: "Kore", child: "Puck" };
const MAX_VIDEO_SECONDS = 7 * 60;
const MAX_UPLOAD_BYTES = 300 * 1024 * 1024;
const TTS_SAMPLE_RATE = 24000;
const MAX_TEMPO = Number(process.env.MAX_TEMPO) || 1.5;
const GEMINI_TIMEOUT_MS = 90 * 1000;
const TTS_CONCURRENCY = 2;
const JOB_CONCURRENCY = Number(process.env.JOB_CONCURRENCY) || 1;
const RATE_LIMIT_PER_HOUR = Number(process.env.RATE_LIMIT_PER_HOUR) || 30;
const TTS_PRIORITY = (process.env.TTS_PRIORITY || "edge").toLowerCase();

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function shortError(err, max = 160) {
  let msg = String(err?.message || err || "");
  try {
    const j = JSON.parse(msg);
    if (j?.error?.message) msg = `${j.error.code || ""} ${j.error.message}`.trim();
  } catch (e) { /* JSON မဟုတ် */ }
  return msg.replace(/\s+/g, " ").slice(0, max);
}

/* ---------------- auth + rate limit ---------------- */
app.use("/api", (req, res, next) => {
  if (!APP_SECRET) return next();
  const given = req.get("x-app-secret") || req.query.key;
  if (given !== APP_SECRET) return res.status(401).json({ error: "ခွင့်ပြုချက် မရှိပါ။" });
  next();
});

const hits = new Map();
function rateLimit(req, res, next) {
  const now = Date.now();
  const list = (hits.get(req.ip) || []).filter((t) => now - t < 3600 * 1000);
  if (list.length >= RATE_LIMIT_PER_HOUR) {
    return res.status(429).json({ error: "တစ်နာရီအတွင်း အကြိမ်များလွန်းပါသည်။ ခဏစောင့်ပါ။" });
  }
  list.push(now);
  hits.set(req.ip, list);
  next();
}

/* ---------------- daily quota (တစ်ရက် ၃ ကြိမ်၊ ကြော်ငြာ ၁ ခု = ၁ ကြိမ်) ---------------- */
const QUOTA_ENABLED = process.env.QUOTA_ENABLED === "true"; // default: ပိတ်ထား
const DAILY_FREE = Number(process.env.DAILY_FREE) || 3;
const MAX_ADS_PER_DAY = Number(process.env.MAX_ADS_PER_DAY) || 10;
const AD_COOLDOWN_MS = Number(process.env.AD_COOLDOWN_MS) || 15 * 1000;
const usage = new Map(); // deviceId -> { day, used, ads, lastAd }

// မြန်မာအချိန် (UTC+6:30) သန်းခေါင်ကျော်ရင် ရက်အသစ်
const dayKey = () => new Date(Date.now() + 6.5 * 3600 * 1000).toISOString().slice(0, 10);

function getUsage(id) {
  const day = dayKey();
  let u = usage.get(id);
  if (!u || u.day !== day) {
    u = { day, used: 0, ads: 0, lastAd: 0 };
    usage.set(id, u);
  }
  return u;
}
const remainingOf = (u) => Math.max(0, DAILY_FREE + u.ads - u.used);

function deviceId(req) {
  const id = String(req.get("x-device-id") || req.query.deviceId || "").trim();
  return /^[\w-]{8,64}$/.test(id) ? id : null;
}

function refundQuota(job) {
  if (!job.deviceId || job.refunded) return;
  const u = getUsage(job.deviceId);
  if (u.day === job.quotaDay) u.used = Math.max(0, u.used - 1);
  job.refunded = true;
}

app.get("/api/quota", (req, res) => {
  const id = deviceId(req);
  if (!id) return res.status(400).json({ error: "x-device-id မပါပါ။" });
  const u = getUsage(id);
  res.json({
    remaining: remainingOf(u),
    freePerDay: DAILY_FREE,
    adsWatched: u.ads,
    canWatchAd: u.ads < MAX_ADS_PER_DAY
  });
});

// ကြော်ငြာအပြည့်ကြည့်ပြီးတိုင်း app က ဒါကို ခေါ်ပါ (+၁ ကြိမ်)
app.post("/api/reward-ad", (req, res) => {
  const id = deviceId(req);
  if (!id) return res.status(400).json({ error: "x-device-id မပါပါ။" });
  const u = getUsage(id);
  if (u.ads >= MAX_ADS_PER_DAY) {
    return res.status(429).json({ error: "ယနေ့ ကြော်ငြာကြည့်ခွင့် ကုန်သွားပါပြီ။", remaining: remainingOf(u) });
  }
  if (Date.now() - u.lastAd < AD_COOLDOWN_MS) {
    return res.status(429).json({ error: "ခဏစောင့်ပြီး ပြန်စမ်းပါ။", remaining: remainingOf(u) });
  }
  u.ads++;
  u.lastAd = Date.now();
  res.json({ remaining: remainingOf(u) });
});

/* ---------------- basic routes ---------------- */
app.get("/", (req, res) => {
  res.json({ status: "ok", service: "SOE AI TRANSLATOR Backend" });
});

app.get("/health", (req, res) => {
  res.json({ ok: true, hasApiKey: Boolean(apiKey), hasGroqKey: Boolean(groqKey), hasFfmpeg: Boolean(ffmpegPath) });
});

/* ---------------- Gemini helpers ---------------- */
function languageLabel(code = "auto") {
  if (code === "my") return "Burmese (Myanmar)";
  if (code === "th") return "Thai";
  if (code === "en") return "English";
  if (String(code).startsWith("zh")) return "Chinese";
  return "auto-detected language";
}

let workingTextModel = null;
const modelCooldown = new Map();
function coolDown(model, err) {
  const m = String(err?.message || err);
  let ms = 0;
  if (/PerDay|per day/i.test(m)) ms = 60 * 60 * 1000;
  else if (/429|RESOURCE_EXHAUSTED/i.test(m)) ms = 2 * 60 * 1000;
  else if (/503|UNAVAILABLE|abort|timeout|timed out/i.test(m)) ms = 2 * 60 * 1000;
  else if (/404|NOT_FOUND|no longer available/i.test(m)) ms = 6 * 60 * 60 * 1000;
  if (ms) modelCooldown.set(model, Date.now() + ms);
}

async function generateText(contents) {
  if (!ai) throw new Error("GEMINI_API_KEY မရှိပါ။");
  let order = [...new Set([workingTextModel, ...TEXT_MODELS].filter(Boolean))];
  const active = order.filter((m) => (modelCooldown.get(m) || 0) <= Date.now());
  if (active.length) order = active;
  const errors = [];
  for (const model of order) {
    try {
      const response = await ai.models.generateContent({
        model,
        contents,
        config: { abortSignal: AbortSignal.timeout(GEMINI_TIMEOUT_MS) }
      });
      const text = (response.text || "").trim();
      if (text) {
        workingTextModel = model;
        return text;
      }
      errors.push(`${model}: ရလဒ်အလွတ်`);
    } catch (err) {
      console.error(`Model ${model} failed:`, err.message);
      coolDown(model, err);
      errors.push(`${model}: ${shortError(err)}`);
    }
  }
  throw new Error("Gemini မအောင်မြင်ပါ။ " + (errors.join(" | ") || "model မရပါ။"));
}

function decodeAudio(buf, mime = "") {
  let rate = Number((/rate=(\d+)/.exec(mime) || [])[1]) || TTS_SAMPLE_RATE;
  if (buf.length > 44 && buf.toString("ascii", 0, 4) === "RIFF") {
    let offset = 12;
    while (offset + 8 <= buf.length) {
      const id = buf.toString("ascii", offset, offset + 4);
      const size = buf.readUInt32LE(offset + 4);
      if (id === "fmt ") rate = buf.readUInt32LE(offset + 12);
      if (id === "data") {
        const end = size === 0xffffffff || offset + 8 + size > buf.length ? buf.length : offset + 8 + size;
        return { pcm: buf.subarray(offset + 8, end), rate };
      }
      offset += 8 + size + (size % 2);
    }
  }
  return { pcm: buf, rate };
}

let workingTtsModel = null;
async function ttsChunk(text, voiceName) {
  const order = [...new Set([workingTtsModel, ...TTS_MODELS].filter(Boolean))];
  const errors = [];
  for (const model of order) {
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const response = await ai.models.generateContent({
          model,
          contents: [{ role: "user", parts: [{ text }] }],
          config: {
            responseModalities: ["AUDIO"],
            speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName } } },
            abortSignal: AbortSignal.timeout(GEMINI_TIMEOUT_MS)
          }
        });
        const parts = response.candidates?.[0]?.content?.parts || [];
        const part = parts.find((p) => p.inlineData?.data);
        if (!part) throw new Error("TTS မှ အသံ ပြန်မလာပါ။");
        workingTtsModel = model;
        return decodeAudio(Buffer.from(part.inlineData.data, "base64"), part.inlineData.mimeType || "");
      } catch (err) {
        console.error(`TTS ${model} attempt ${attempt + 1} failed:`, err.message);
        if (attempt === 1) errors.push(`${model}: ${shortError(err)}`);
        else await sleep(2000);
      }
    }
  }
  throw new Error("TTS မအောင်မြင်ပါ။ " + (errors.join(" | ") || "model မရပါ။"));
}

/* ---------------- Edge TTS (အခမဲ့) ---------------- */
function escapeXml(t) {
  return t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

async function edgeTtsChunk(text, voiceKey, dir, idx, target) {
  const tts = new MsEdgeTTS();
  try {
    const edgeVoiceKey = voiceKey === "female" ? "female" : "male";
    await tts.setMetadata(
      getTarget(target).edge[edgeVoiceKey] || getTarget(target).edge.male,
      OUTPUT_FORMAT.WEBM_24KHZ_16BIT_MONO_OPUS
    );
    const ssmlText = `<prosody rate="+0%">${escapeXml(text)}</prosody>`;
    const { audioStream } = await tts.toStream(ssmlText);
    const bufs = [];
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("Edge TTS အချိန်ကြာလွန်းသည်")), 40000);
      const done = () => { clearTimeout(timer); resolve(); };
      audioStream.on("data", (d) => bufs.push(d));
      audioStream.on("end", done);
      audioStream.on("close", done);
      audioStream.on("error", (e) => { clearTimeout(timer); reject(e); });
    });
    const audio = Buffer.concat(bufs);
    if (audio.length < 200) throw new Error("Edge TTS မှ အသံ ပြန်မလာပါ။");
    const inFile = path.join(dir, `edge-${idx}.webm`);
    const outFile = path.join(dir, `edge-${idx}.pcm`);
    fs.writeFileSync(inFile, audio);
    await runFfmpeg(["-i", inFile, "-f", "s16le", "-ar", String(TTS_SAMPLE_RATE), "-ac", "1", outFile]);
    return { pcm: fs.readFileSync(outFile), rate: TTS_SAMPLE_RATE };
  } finally {
    try { if (typeof tts.close === "function") tts.close(); } catch (e) { /* ignore */ }
  }
}

async function edgeWithRetry(text, voiceKey, dir, idx, target) {
  let lastErr;
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      return await edgeTtsChunk(text, voiceKey, dir, idx, target);
    } catch (err) {
      lastErr = err;
      console.error(`Edge TTS attempt ${attempt + 1} failed:`, err.message);
      if (attempt === 0) await sleep(600);
    }
  }
  throw lastErr;
}

function geminiVoiceFor(voiceKey, tag) {
  return tag === "C" ? VOICES.child : VOICES[voiceKey] || VOICES.male;
}

async function synthChunk(text, voiceKey, dir, idx, target, tag = "M") {
  if (TTS_PRIORITY === "gemini" && ai) {
    try {
      return await ttsChunk(text, geminiVoiceFor(voiceKey, tag));
    } catch (err) {
      console.error("Gemini TTS failed, falling back to Edge:", err.message);
      return await edgeWithRetry(text, voiceKey, dir, idx, target);
    }
  }
  try {
    return await edgeWithRetry(text, voiceKey, dir, idx, target);
  } catch (err) {
    console.error("Edge TTS failed twice, falling back to Gemini:", err.message);
    if (!ai) throw err;
    return await ttsChunk(text, geminiVoiceFor(voiceKey, tag));
  }
}

function splitForTts(text, max = 450) {
  const pieces = text.match(/[^.!?။。！？\n]+[.!?။。！？\n]*/g) || [text];
  const chunks = [];
  let current = "";
  for (let piece of pieces) {
    piece = piece.trim();
    if (!piece) continue;
    while (piece.length > max) {
      if (current) { chunks.push(current); current = ""; }
      chunks.push(piece.slice(0, max));
      piece = piece.slice(max);
    }
    if ((current + " " + piece).length > max) {
      if (current) chunks.push(current);
      current = piece;
    } else {
      current = current ? current + " " + piece : piece;
    }
  }
  if (current.trim()) chunks.push(current.trim());
  return chunks;
}

/* ---------------- ffmpeg helpers ---------------- */
function runFfmpeg(args) {
  return new Promise((resolve, reject) => {
    const p = spawn(ffmpegPath, ["-y", "-hide_banner", "-loglevel", "error", ...args]);
    let err = "";
    p.stderr.on("data", (d) => {
      err += d;
      if (err.length > 20000) err = err.slice(-20000);
    });
    p.on("error", reject);
    p.on("close", (code) =>
      code === 0 ? resolve() : reject(new Error("ffmpeg error: " + err.slice(-300)))
    );
  });
}

function probeVideo(file) {
  return new Promise((resolve, reject) => {
    const p = spawn(ffmpegPath, ["-hide_banner", "-i", file]);
    let err = "";
    p.stderr.on("data", (d) => (err += d));
    p.on("error", reject);
    p.on("close", () => {
      const m = err.match(/Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)/);
      const duration = m ? Number(m[1]) * 3600 + Number(m[2]) * 60 + Number(m[3]) : 0;
      resolve({ duration, hasAudio: /Audio:/.test(err), hasVideo: /Video:/.test(err) });
    });
  });
}

/* ---------------- timing helpers ---------------- */
const TS_PREFIX = /^\s*\[?\s*\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*\]?\s*[|:]?\s*/;

function parseTimedSegments(scriptText, duration, fallbackVoice = "male") {
  const defaultVoice = fallbackVoice === "female" ? "female" : "male";
  const lines = String(scriptText || "").split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  const out = [];
  for (const line of lines) {
    const m = line.match(/^\[(M|F|C)\]\s*\[?\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*\]?\s*[|:]?\s*(.+)$/i);
    if (!m) continue;
    const tag = m[1].toUpperCase();
    const start = Number(m[2]);
    const end = Number(m[3]);
    const text = m[4].trim();
    if (!text || !(end > start) || start > duration + 1) continue;
    out.push({
      tag,
      voiceKey: tag === "F" ? "female" : tag === "M" || tag === "C" ? "male" : defaultVoice,
      text,
      start: Math.min(start, Math.max(0, duration - 0.3)),
      end: Math.min(end, duration)
    });
  }
  if (!lines.length || out.length < Math.ceil(lines.length * 0.6)) return [];
  out.sort((a, b) => a.start - b.start);
  return out;
}

function detectSpeech(audioPath, duration) {
  return new Promise((resolve, reject) => {
    const p = spawn(ffmpegPath, ["-hide_banner", "-nostats", "-i", audioPath, "-af", "silencedetect=noise=-35dB:d=0.35", "-f", "null", "-"]);
    let err = "";
    p.stderr.on("data", (d) => { err += d; if (err.length > 200000) err = err.slice(-200000); });
    p.on("error", reject);
    p.on("close", () => {
      const starts = [...err.matchAll(/silence_start:\s*(-?[\d.]+)/g)].map((m) => Number(m[1]));
      const ends = [...err.matchAll(/silence_end:\s*([\d.]+)/g)].map((m) => Number(m[1]));
      let speechStart = 0;
      if (starts.length && starts[0] < 0.1 && ends.length) speechStart = ends[0];
      let speechEnd = duration;
      if (starts.length > ends.length) speechEnd = Math.max(speechStart, starts[starts.length - 1]);
      resolve({ speechStart, speechEnd, onsets: ends });
    });
  });
}

// Gemini ပေးတဲ့ အချိန်ကို တကယ့်စကားပြောချိန်နဲ့ ပြန်ညှိ (start နဲ့ end နှစ်ခုလုံး)
function calibrateTimed(timed, sp, duration) {
  const gs = Math.min(...timed.map((t) => t.start));
  const ge = Math.max(...timed.map((t) => t.end));
  const span = ge - gs;
  const real = sp.speechEnd - sp.speechStart;
  console.log(`[calibrate] gemini ${gs.toFixed(1)}-${ge.toFixed(1)}s, real speech ${sp.speechStart.toFixed(1)}-${sp.speechEnd.toFixed(1)}s`);
  let out = timed.map((t) => ({ ...t }));
  if (span > 1 && real > 1 && (Math.abs(ge - sp.speechEnd) > 1 || Math.abs(gs - sp.speechStart) > 0.8)) {
    const k = real / span;
    out = out.map((t) => ({
      ...t,
      start: sp.speechStart + (t.start - gs) * k,
      end: sp.speechStart + (t.end - gs) * k
    }));
  }
  const onsets = [sp.speechStart, ...sp.onsets];
  const used = new Set();
  out = out.map((t) => {
    let best = null, bd = 1.2;
    for (const o of onsets) {
      const d = Math.abs(o - t.start);
      if (d < bd && !used.has(o)) { bd = d; best = o; }
    }
    if (best !== null) {
      used.add(best);
      const shift = best - t.start;
      return { ...t, start: best, end: t.end + shift };
    }
    return t;
  });
  out.sort((a, b) => a.start - b.start);
  for (const t of out) {
    t.start = Math.min(Math.max(0, t.start), Math.max(0, duration - 0.3));
    t.end = Math.min(duration, Math.max(t.end, t.start + 0.3));
  }
  return out;
}

// အချိန်မပါလာရင် စာလုံးအရေအတွက်အလိုက် စကားပြောချိန်တစ်လျှောက် ဖြန့်ချ
function autoTime(segments, sp) {
  const total = segments.reduce((a, s) => a + s.text.length, 0) || 1;
  const span = Math.max(1, sp.speechEnd - sp.speechStart);
  let t = sp.speechStart;
  return segments.map((s) => {
    const d = (span * s.text.length) / total;
    const o = { ...s, start: t, end: t + d };
    t += d;
    return o;
  });
}

async function synthSegment(seg, dir, idx, target) {
  const pieces = splitForTts(seg.text, 450);
  const outs = [];
  for (let k = 0; k < pieces.length; k++) {
    outs.push(await synthChunk(pieces[k], seg.voiceKey, dir, `${idx}_${k}`, target, seg.tag));
  }
  return { pcm: Buffer.concat(outs.map((o) => o.pcm)), rate: outs[0].rate };
}

async function tempoPcm(pcm, rate, t, dir, idx) {
  const inF = path.join(dir, `seg-${idx}.pcm`);
  const outF = path.join(dir, `seg-${idx}-t.pcm`);
  fs.writeFileSync(inF, pcm);
  await runFfmpeg([
    "-f", "s16le", "-ar", String(rate), "-ac", "1", "-i", inF,
    "-filter:a", `atempo=${t.toFixed(3)}`,
    "-f", "s16le", "-ar", String(rate), "-ac", "1", outF
  ]);
  return fs.readFileSync(outF);
}

// ဝါကျတစ်ခုချင်းကို မူရင်း start နေရာမှာ ချပေးပြီး slot (end) ကို မကျော်စေရ
async function buildTimeline(parts, timed, duration, rate, dir) {
  const bytesPerSec = rate * 2;
  const placed = [];
  for (let i = 0; i < timed.length; i++) {
    let pcm = parts[i].pcm;
    const start = timed[i].start;
    const nextStart = i + 1 < timed.length ? timed[i + 1].start : duration;
    const hardEnd = Math.min(timed[i].end + 0.3, nextStart - 0.05, duration);
    const slot = Math.max(0.5, hardEnd - start);
    const dur = pcm.length / bytesPerSec;
    if (dur > slot) {
      pcm = await tempoPcm(pcm, rate, Math.min(dur / slot, MAX_TEMPO), dir, i);
    }
    const maxBytes = Math.floor(slot * bytesPerSec) & ~1;
    if (pcm.length > maxBytes) {
      pcm = Buffer.from(pcm.subarray(0, maxBytes));
      const fade = Math.min(Math.floor(rate * 0.08), Math.floor(pcm.length / 2));
      for (let k = 0; k < fade; k++) {
        const off = pcm.length - 2 - k * 2;
        pcm.writeInt16LE(Math.round(pcm.readInt16LE(off) * (k / fade)), off);
      }
    }
    placed.push({ start, pcm });
  }
  const out = Buffer.alloc(Math.ceil(duration * rate) * 2);
  for (const p of placed) {
    const off = Math.floor(p.start * rate) * 2;
    if (off >= out.length) continue;
    const n = Math.min(p.pcm.length, out.length - off) & ~1;
    for (let b = 0; b < n; b += 2) {
      const v = out.readInt16LE(off + b) + p.pcm.readInt16LE(b);
      out.writeInt16LE(Math.max(-32768, Math.min(32767, v)), off + b);
    }
  }
  return out;
}

function parseSpeakerSegments(scriptText, fallbackVoice = "male") {
  const defaultVoice = fallbackVoice === "female" ? "female" : "male";
  const lines = String(scriptText || "").split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  const raw = [];
  for (const line of lines) {
    const match = line.match(/^\[(M|F|C)\]\s*(.+)$/i);
    const tag = match ? match[1].toUpperCase() : null;
    const speech = (match ? match[2] : line).replace(TS_PREFIX, "").trim();
    if (!speech) continue;
    let voiceKey = defaultVoice;
    if (tag === "F") voiceKey = "female";
    else if (tag === "M" || tag === "C") voiceKey = "male";
    raw.push({ tag: tag || (voiceKey === "female" ? "F" : "M"), voiceKey, text: speech });
  }
  const segments = [];
  for (const s of raw) {
    for (const part of splitForTts(s.text, 450)) {
      segments.push({ tag: s.tag, voiceKey: s.voiceKey, text: part });
    }
  }
  return segments;
}

/* ---------------- diagnostic ---------------- */
app.get("/api/test-gemini", async (req, res) => {
  if (!ai) return res.status(500).json({ ok: false, error: "GEMINI_API_KEY မရှိပါ။" });
  try {
    const text = await generateText("Reply with the single word: OK");
    res.json({ ok: true, textModel: workingTextModel, reply: text.slice(0, 40) });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

/* ---------------- text translate ---------------- */
app.post("/api/text-translate", rateLimit, async (req, res) => {
  try {
    const { text, sourceLanguage = "auto", targetLanguage = "my" } = req.body || {};
    if (!text || !String(text).trim()) {
      return res.status(400).json({ error: "ဘာသာပြန်မည့်စာသား လိုအပ်ပါသည်။" });
    }
    if (!ai) {
      return res.status(500).json({ error: "Render Environment ထဲတွင် GEMINI_API_KEY မရှိသေးပါ။" });
    }
    const prompt =
      `Translate the following ${languageLabel(sourceLanguage)} text into natural, soft, and clear ${getTarget(targetLanguage).label}. ` +
      `Return ONLY the translated ${getTarget(targetLanguage).label} text without explanations or quotes:\n\n${text}`;
    const translatedText = await generateText(prompt);
    return res.json({ translatedText });
  } catch (err) {
    console.error("Translate error:", err);
    return res.status(500).json({ error: err.message || "Server ဘက်တွင် ဘာသာပြန် Error တက်နေပါသည်။" });
  }
});

/* ---------------- jobs ---------------- */
const jobs = new Map();
const upload = multer({
  dest: path.join(os.tmpdir(), "soe-uploads"),
  limits: { fileSize: MAX_UPLOAD_BYTES }
});

function setJob(job, patch) {
  Object.assign(job, patch, { updatedAt: Date.now() });
}

function safeRm(p) {
  if (p) fs.rm(p, { recursive: true, force: true }, () => {});
}

class NoSpeechError extends Error {
  constructor() {
    super("ဗီဒီယိုထဲတွင် စကားပြောသံ မတွေ့ပါ။");
    this.noSpeech = true;
  }
}

/* ---- Groq Whisper: segment timestamp တိကျသည် ---- */
async function groqTranscribeSegments(filePath, sourceLanguage) {
  const form = new FormData();
  form.append("file", new Blob([fs.readFileSync(filePath)], { type: "audio/mpeg" }), "audio.mp3");
  form.append("model", process.env.GROQ_STT_MODEL || "whisper-large-v3-turbo");
  form.append("response_format", "verbose_json");
  form.append("timestamp_granularities[]", "segment");
  form.append("temperature", "0");
  const lang = sourceLanguage === "en" ? "en" : sourceLanguage === "th" ? "th" : sourceLanguage === "my" ? "my" :
    String(sourceLanguage).startsWith("zh") ? "zh" : null;
  if (lang) form.append("language", lang);

  const r = await fetch("https://api.groq.com/openai/v1/audio/transcriptions", {
    method: "POST",
    headers: { Authorization: `Bearer ${groqKey}` },
    body: form,
    signal: AbortSignal.timeout(180000)
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(`Groq ${r.status}: ${data?.error?.message || "error"}`);
  return (data.segments || [])
    .filter((s) => String(s.text || "").trim().length > 1 && !(s.no_speech_prob > 0.6))
    .map((s) => ({ start: Number(s.start), end: Number(s.end), text: String(s.text).trim() }));
}

// တိုလွန်းတဲ့ segment တွေကို ကပ်ပေါင်း (အပိုင်းတိုလွန်းရင် TTS မကိုက်)
function mergeShortSegments(segs, maxLen = 9) {
  const out = [];
  for (const s of segs) {
    const last = out[out.length - 1];
    if (last && (last.end - last.start < 1.5 || s.end - s.start < 0.8) &&
        s.start - last.end < 0.6 && s.end - last.start <= maxLen) {
      last.end = s.end;
      last.text += " " + s.text;
    } else {
      out.push({ ...s });
    }
  }
  return out;
}

async function myMemoryTranslate(text, sourceLanguage, targetLanguage = "my") {
  const from = sourceLanguage === "auto" ? "Autodetect" : sourceLanguage === "zh" ? "zh-CN" : sourceLanguage;
  const out = [];
  for (const chunk of splitForTts(text, 450)) {
    const url = "https://api.mymemory.translated.net/get?q=" + encodeURIComponent(chunk) +
      "&langpair=" + encodeURIComponent(from + "|" + getTarget(targetLanguage).mm);
    const r = await fetch(url, { signal: AbortSignal.timeout(20000) });
    const d = await r.json().catch(() => ({}));
    const t = d?.responseData?.translatedText;
    if (!r.ok || !t || /MYMEMORY WARNING/i.test(t)) throw new Error("MyMemory ဘာသာပြန်မရပါ။");
    out.push(t);
  }
  return out.join(" ");
}

// segment တစ်ခုချင်းကို slot အရှည်နဲ့ကိုက်အောင် စာလုံးကန့်ပြီး ဘာသာပြန်
async function translateSegments(segs, sourceLanguage, targetLanguage) {
  const cps = CHARS_PER_SEC[targetLanguage] || 11;
  const items = segs.map((s, i) => ({
    id: i,
    seconds: Number((s.end - s.start).toFixed(1)),
    maxChars: Math.max(6, Math.round((s.end - s.start) * cps)),
    text: s.text
  }));
  const prompt =
    `Translate each item's "text" from ${languageLabel(sourceLanguage)} into natural, clear ${getTarget(targetLanguage).label} for voice dubbing. ` +
    `STRICT RULE: each translation must be speakable within that item's "seconds", so it must NOT exceed "maxChars" characters. ` +
    `Use short simple words, drop filler, keep the meaning. ` +
    `Return ONLY a JSON array of strings, exactly ${items.length} items, in the same order as the input, with no markdown and no explanations.\n\n` +
    JSON.stringify(items);
  try {
    const raw = await generateText(prompt);
    const arr = JSON.parse(raw.replace(/```json|```/g, "").trim());
    if (Array.isArray(arr) && arr.length === segs.length && arr.every((x) => String(x).trim())) {
      return arr.map((x) => String(x).trim());
    }
    throw new Error("JSON အရေအတွက် မကိုက်ပါ။");
  } catch (err) {
    console.error("Gemini segment translate failed, using MyMemory:", err.message);
    const out = [];
    for (const s of segs) out.push(await myMemoryTranslate(s.text, sourceLanguage, targetLanguage));
    return out;
  }
}

async function whisperPath(audioPath, opts, job) {
  setJob(job, { progress: 20, message: "စကားပြောသံနဲ့ အချိန်တွေကို နားထောင်နေပါသည်..." });
  let segs = await groqTranscribeSegments(audioPath, opts.sourceLanguage);
  if (!segs.length) throw new NoSpeechError();
  segs = mergeShortSegments(segs);
  setJob(job, { progress: 28, message: "ဘာသာပြန်နေပါသည်..." });
  const translated = await translateSegments(segs, opts.sourceLanguage, opts.targetLanguage);
  const voiceKey = opts.voice === "female" ? "female" : "male";
  const tag = voiceKey === "female" ? "F" : "M";
  return segs.map((s, i) => ({ tag, voiceKey, text: translated[i], start: s.start, end: s.end }));
}

/* ---- Gemini audio: Whisper မရရင် သုံး ---- */
async function geminiPath(audioBase64, opts, job, duration) {
  setJob(job, {
    progress: 20,
    message: "ယောက်ျား၊ မိန်းကလေး၊ ကလေး အသံများကို နားထောင်ပြီး ခွဲခြားဘာသာပြန်နေပါသည်..."
  });
  const cps = CHARS_PER_SEC[opts.targetLanguage] || 11;
  const prompt =
    `Listen carefully to the actual voices in this audio (spoken language: ${languageLabel(opts.sourceLanguage)}). ` +
    `Identify who is speaking each line by voice pitch and tone: use [M] for a male voice, [F] for a female voice, and [C] for a child voice. ` +
    `Translate everything spoken into natural, clear ${getTarget(opts.targetLanguage).label} suitable for voice dubbing. ` +
    `Write one sentence per line in EXACTLY this format: [M] 3.2-6.8 | translated sentence  (the two numbers are the start and end time in seconds, relative to the start of the audio, when that sentence is spoken in the ORIGINAL audio). Return ONLY these lines with no explanations. ` +
    `TIMING RULE: the video is ${Math.round(duration)} seconds long. Each translated line must be speakable within its (end - start) seconds: at most about ${cps} characters per second. Use short simple words, no filler. Never make a line longer than the original speech. ` +
    `If there is no speech, return exactly: NO_SPEECH`;

  const text = await generateText([
    { role: "user", parts: [{ inlineData: { mimeType: "audio/mp3", data: audioBase64 } }, { text: prompt }] }
  ]);
  if (/^NO_SPEECH$/i.test(text.trim())) throw new NoSpeechError();
  return text;
}

const queue = [];
let running = 0;
function enqueue(job, file, opts) {
  queue.push({ job, file, opts });
  setJob(job, { message: `တန်းစီစောင့်နေပါသည်... (${queue.length})` });
  pump();
}
function pump() {
  while (running < JOB_CONCURRENCY && queue.length) {
    const { job, file, opts } = queue.shift();
    running++;
    processJob(job, file, opts).finally(() => {
      running--;
      pump();
    });
  }
}

async function processJob(job, file, opts) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "soe-job-"));
  const t0 = Date.now();
  const lap = (name) => console.log(`[job ${job.id.slice(0, 6)}] ${name}: ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  try {
    setJob(job, { status: "processing", progress: 3, message: "ဗီဒီယိုကို စစ်ဆေးနေပါသည်..." });
    const info = await probeVideo(file.path);
    if (!info.hasVideo || !info.duration) throw new Error("ဗီဒီယိုဖိုင်ကို ဖတ်၍မရပါ။");
    if (!info.hasAudio) throw new Error("ဤဗီဒီယိုတွင် အသံ မပါပါ။");
    if (info.duration > MAX_VIDEO_SECONDS) {
      throw new Error("ဗီဒီယိုက အရှည်လွန်းပါသည်။ ၇ မိနစ်အောက် သုံးပါ။");
    }

    setJob(job, { progress: 8, message: "ဗီဒီယိုထဲက အသံကို ခွဲထုတ်နေပါသည်..." });
    const audioPath = path.join(dir, "audio.mp3");
    await runFfmpeg(["-i", file.path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", audioPath]);

    /* ---- အချိန်ပါတဲ့ ဘာသာပြန် segments ရယူ ---- */
    let timed = null;
    let displayText = "";

    if (groqKey) {
      try {
        timed = await whisperPath(audioPath, opts, job);
        displayText = timed.map((t) => `[${t.tag}] ${t.text}`).join("\n");
        console.log(`[job ${job.id.slice(0, 6)}] WHISPER timed mode, segments: ${timed.length}`);
      } catch (err) {
        if (err.noSpeech) throw err;
        console.error("Whisper path failed, trying Gemini audio:", err.message);
        timed = null;
      }
    }

    if (!timed) {
      if (!ai) throw new Error("Whisper မအောင်မြင်ပါ၊ GEMINI_API_KEY လည်း မရှိပါ။");
      const audioBase64 = fs.readFileSync(audioPath).toString("base64");
      const script = await geminiPath(audioBase64, opts, job, info.duration);
      displayText = String(script)
        .split(/\r?\n/)
        .map((l) => l.replace(/^(\[(?:M|F|C)\])\s*\[?\s*\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*\]?\s*[|:]?\s*/i, "$1 "))
        .join("\n");

      let sp = { speechStart: 0, speechEnd: info.duration, onsets: [] };
      try {
        sp = await detectSpeech(audioPath, info.duration);
      } catch (e) {
        console.error("Speech detect failed:", e.message);
      }

      timed = parseTimedSegments(script, info.duration, opts.voice);
      if (timed.length) {
        console.log(`[job ${job.id.slice(0, 6)}] GEMINI timed mode, segments: ${timed.length}`);
        timed = calibrateTimed(timed, sp, info.duration);
      } else {
        const segments = parseSpeakerSegments(script, opts.voice);
        if (!segments.length) throw new Error("ဘာသာပြန်စာသားမှ အသံဖန်တီးရန် စာကြောင်း မတွေ့ပါ။");
        console.log(`[job ${job.id.slice(0, 6)}] AUTO-timed mode, segments: ${segments.length}`);
        timed = autoTime(segments, sp);
      }
    }
    lap("speech→translated");

    /* ---- TTS ---- */
    const parts = new Array(timed.length);
    let nextT = 0, doneT = 0;
    const worker = async () => {
      while (true) {
        const i = nextT++;
        if (i >= timed.length) return;
        parts[i] = await synthSegment(timed[i], dir, i, opts.targetLanguage);
        doneT++;
        setJob(job, {
          progress: 35 + Math.round((doneT / timed.length) * 45),
          message: `အသံ ဖန်တီးနေပါသည်... (${doneT}/${timed.length})`
        });
      }
    };
    await Promise.all(Array.from({ length: Math.min(TTS_CONCURRENCY, timed.length) }, worker));
    lap("TTS");

    const ttsRate = parts[0].rate;
    const pcm = await buildTimeline(parts, timed, info.duration, ttsRate, dir);

    const pcmPath = path.join(dir, "tts.pcm");
    const ttsWav = path.join(dir, "tts.wav");
    fs.writeFileSync(pcmPath, pcm);
    await runFfmpeg(["-f", "s16le", "-ar", String(ttsRate), "-ac", "1", "-i", pcmPath, ttsWav]);

    /* ---- mux (global tempo မသုံးတော့ - timeline က ဗီဒီယိုအရှည်အတိုင်းပြီးသား) ---- */
    setJob(job, { progress: 85, message: "အသံအသစ်ကို ဗီဒီယိုနဲ့ ပေါင်းနေပါသည်..." });
    const fadeOutStart = Math.max(0, info.duration - 0.15);
    const filter = `[1:a]afade=t=in:st=0:d=0.05,afade=t=out:st=${fadeOutStart.toFixed(2)}:d=0.15,apad[a]`;

    const outPath = path.join(dir, "output.mp4");
    const baseArgs = ["-i", file.path, "-i", ttsWav, "-filter_complex", filter, "-map", "0:v:0", "-map", "[a]"];
    const tailArgs = ["-c:a", "aac", "-b:a", "128k", "-t", String(info.duration), "-movflags", "+faststart", outPath];

    try {
      await runFfmpeg([...baseArgs, "-c:v", "copy", ...tailArgs]);
    } catch (copyErr) {
      console.error("Video copy failed, re-encoding:", copyErr.message);
      setJob(job, { progress: 90, message: "ဗီဒီယိုကို ပြန်လည်ပြင်ဆင်နေပါသည်... (အချိန်ကြာနိုင်သည်)" });
      await runFfmpeg([
        ...baseArgs,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p",
        ...tailArgs
      ]);
    }

    lap("mux done");
    setJob(job, {
      status: "done",
      progress: 100,
      message: "ပြီးစီးပါပြီ။",
      outPath,
      outDir: dir,
      translatedText: displayText
    });
  } catch (err) {
    console.error("Job failed:", err);
    refundQuota(job);
    setJob(job, { status: "error", progress: 0, error: err.message || "Video ဘာသာပြန်မအောင်မြင်ပါ။" });
    safeRm(dir);
  } finally {
    safeRm(file.path);
  }
}

app.post("/api/translate", rateLimit, upload.single("video"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "Video ဖိုင် မပါပါ။" });
  }
  if (!ai && !groqKey) {
    safeRm(req.file.path);
    return res.status(500).json({ error: "Render Environment ထဲတွင် GROQ_API_KEY သို့မဟုတ် GEMINI_API_KEY မရှိသေးပါ။" });
  }

  // QUOTA_ENABLED=true ထားမှသာ quota စစ်မည် (မထားရင် ရှိပြီးသား page အတိုင်း အလုပ်လုပ်)
  let devId = null;
  let quota = null;
  if (QUOTA_ENABLED) {
    devId = deviceId(req);
    if (!devId) {
      safeRm(req.file.path);
      return res.status(400).json({ error: "x-device-id မပါပါ။" });
    }
    quota = getUsage(devId);
    if (remainingOf(quota) <= 0) {
      safeRm(req.file.path);
      return res.status(402).json({
        error: "ယနေ့ အကြိမ်ရေ ကုန်သွားပါပြီ။ ကြော်ငြာကြည့်ပြီး ထပ်ရယူပါ။",
        needAd: true,
        remaining: 0
      });
    }
    quota.used++;
  }

  const jobId = crypto.randomUUID();
  const job = {
    id: jobId,
    deviceId: devId,
    quotaDay: quota ? quota.day : null,
    status: "queued",
    progress: 1,
    message: "အလုပ်စတင်ရန် စီစဉ်နေပါသည်...",
    createdAt: Date.now(),
    updatedAt: Date.now()
  };
  jobs.set(jobId, job);

  const opts = {
    sourceLanguage: req.body?.sourceLanguage || "auto",
    targetLanguage: TARGETS[req.body?.targetLanguage] ? req.body.targetLanguage : "my",
    voice: req.body?.voice || "male"
  };

  enqueue(job, req.file, opts);
  return res.json({ jobId });
});

app.get("/api/job/:id", (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) {
    return res.status(404).json({ error: "အလုပ်ကို မတွေ့ပါ။ (Server ပြန်စသွားနိုင်သည်၊ ပြန်စမ်းပါ)" });
  }
  res.json({
    status: job.status,
    progress: job.progress,
    message: job.message,
    error: job.error,
    translatedText: job.status === "done" ? job.translatedText : undefined
  });
});

app.get("/api/download/:id", (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job || job.status !== "done" || !job.outPath || !fs.existsSync(job.outPath)) {
    return res.status(404).json({ error: "ဖိုင်ကို မတွေ့ပါ။" });
  }
  if (req.query.inline) {
    return res.type("video/mp4").sendFile(job.outPath);
  }
  return res.download(job.outPath, "soe-translated.mp4");
});

setInterval(() => {
  const now = Date.now();
  for (const [id, job] of jobs) {
    if (now - job.updatedAt > 30 * 60 * 1000 && job.status !== "processing") {
      safeRm(job.outDir);
      jobs.delete(id);
    }
  }
  for (const [ip, list] of hits) {
    if (!list.some((t) => now - t < 3600 * 1000)) hits.delete(ip);
  }
}, 5 * 60 * 1000);

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
