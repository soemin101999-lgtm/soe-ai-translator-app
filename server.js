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

app.use(cors());
app.use(express.json({ limit: "10mb" }));

const apiKey = process.env.GEMINI_API_KEY;
const groqKey = process.env.GROQ_API_KEY; // အခမဲ့ Whisper (စကားနားထောင်ရန်)
// TTS model များသည် v1beta လိုအပ်သည်
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

// ပြန်ဆိုမည့်ဘာသာ (target): မြန်မာ / ထိုင်း / အင်္ဂလိပ် / တရုတ်
const TARGETS = {
  my: { label: "Myanmar (Burmese)", mm: "my", edge: { male: "my-MM-ThihaNeural", female: "my-MM-NilarNeural" } },
  th: { label: "Thai", mm: "th", edge: { male: "th-TH-NiwatNeural", female: "th-TH-PremwadeeNeural" } },
  en: { label: "English", mm: "en", edge: { male: "en-US-GuyNeural", female: "en-US-JennyNeural" } },
  zh: { label: "Simplified Chinese (Mandarin)", mm: "zh-CN", edge: { male: "zh-CN-YunxiNeural", female: "zh-CN-XiaoxiaoNeural" } }
};
const getTarget = (code) => TARGETS[code] || TARGETS.my;

const VOICES = { male: "Charon", female: "Kore", child: "Puck" };
const MAX_VIDEO_SECONDS = 7 * 60;
const MAX_UPLOAD_BYTES = 300 * 1024 * 1024;
const TTS_SAMPLE_RATE = 24000;
const MAX_TEMPO = 1.6;                 // အသံမပျက်စေဘဲ အများဆုံး မြန်နိုင်သည့်နှုန်း
const GEMINI_TIMEOUT_MS = 90 * 1000;   // Gemini တစ်ခါခေါ်တာ ၉၀ စက္ကန့်ထက် မကြာစေရ
const TTS_CONCURRENCY = 2;             // Edge TTS တည်ငြိမ်စေရန်
// TTS_PRIORITY=gemini ထားရင် Gemini အသံကို အရင်သုံး (ပိုသဘာဝကျ၊ ဒါပေမဲ့ quota နည်း)
const TTS_PRIORITY = (process.env.TTS_PRIORITY || "edge").toLowerCase();

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Google error က JSON ပါလာတတ်လို့ အဓိကစာကိုပဲ ဆွဲထုတ်
function shortError(err, max = 160) {
  let msg = String(err?.message || err || "");
  try {
    const j = JSON.parse(msg);
    if (j?.error?.message) msg = `${j.error.code || ""} ${j.error.message}`.trim();
  } catch (e) { /* JSON မဟုတ် */ }
  return msg.replace(/\s+/g, " ").slice(0, max);
}

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
  else if (/503|UNAVAILABLE|abort|timeout|timed out/i.test(m)) ms = 2 * 60 * 1000; // အလုပ်များ/နှေးနေတဲ့ model ကို ခဏကျော်
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

function buildGentleTtsInput(text, tag = "M") {
  const styleHint =
    tag === "C"
      ? "Speak softly, gently, and naturally like a sweet child: "
      : tag === "F"
      ? "Speak warmly, softly, and naturally: "
      : "Speak calmly, smoothly, and naturally: ";
  return `${styleHint}${text}`;
}

let workingTtsModel = null;
async function ttsChunk(text, voiceName, tag = "M") {
  const order = [...new Set([workingTtsModel, ...TTS_MODELS].filter(Boolean))];
  const errors = [];
  const spokenText = buildGentleTtsInput(text, tag);
  for (const model of order) {
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const response = await ai.models.generateContent({
          model,
          contents: [{ role: "user", parts: [{ text: spokenText }] }],
          config: {
            responseModalities: ["AUDIO"],
            speechConfig: {
              voiceConfig: { prebuiltVoiceConfig: { voiceName } }
            },
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
        await sleep(2000);
      }
    }
  }
  throw new Error("TTS မအောင်မြင်ပါ။ " + (errors.join(" | ") || "model မရပါ။"));
}

/* ---------------- အခမဲ့ အသံ (Microsoft Edge TTS) ---------------- */
function escapeXml(t) {
  return t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

async function edgeTtsChunk(text, voiceKey, dir, idx, target) {
  const tts = new MsEdgeTTS();
  const edgeVoiceKey = voiceKey === "female" ? "female" : "male";
  await tts.setMetadata(getTarget(target).edge[edgeVoiceKey] || getTarget(target).edge.male, OUTPUT_FORMAT.WEBM_24KHZ_16BIT_MONO_OPUS);
  const { audioStream } = await tts.toStream(escapeXml(text));
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
}

// Edge ကို ၂ ကြိမ်စမ်း (ပထမကျရင် ပြန်စမ်း)
async function edgeWithRetry(text, voiceKey, dir, idx, target) {
  let lastErr;
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      return await edgeTtsChunk(text, voiceKey, dir, idx, target);
    } catch (err) {
      lastErr = err;
      console.error(`Edge TTS attempt ${attempt + 1} failed:`, err.message);
      await sleep(600);
    }
  }
  throw lastErr;
}

function geminiVoiceFor(voiceKey, tag) {
  return tag === "C" ? VOICES.child : VOICES[voiceKey] || VOICES.male;
}

// Edge (အခမဲ့) ကို အရင်၊ ၂ ကြိမ်ကျမှ Gemini (TTS_PRIORITY=gemini ဆိုရင် ပြောင်းပြန်)
async function synthChunk(text, voiceKey, dir, idx, target, tag = "M") {
  if (TTS_PRIORITY === "gemini" && ai) {
    try {
      return await ttsChunk(text, geminiVoiceFor(voiceKey, tag), tag);
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
    return await ttsChunk(text, geminiVoiceFor(voiceKey, tag), tag);
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

/* [M], [F], [C] Tag အလိုက် အသံခွဲခြင်း — အသံတူ ဆက်တိုက်စာကြောင်းတွေကို ပေါင်းပြီး TTS ခေါ်ရတဲ့အကြိမ် လျှော့သည် */
function parseSpeakerSegments(scriptText, fallbackVoice = "male") {
  const defaultVoice = fallbackVoice === "female" ? "female" : "male";
  const lines = String(scriptText || "")
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);

  const raw = [];
  for (const line of lines) {
    const match = line.match(/^\[(M|F|C)\]\s*(.+)$/i);
    const tag = match ? match[1].toUpperCase() : null;
    const speech = (match ? match[2] : line).trim();
    if (!speech) continue;

    let voiceKey = defaultVoice;
    if (tag === "F") voiceKey = "female";
    else if (tag === "M" || tag === "C") voiceKey = "male";

    raw.push({
      tag: tag || (voiceKey === "female" ? "F" : "M"),
      voiceKey,
      text: speech
    });
  }

  // အသံတူ ဆက်တိုက်လာရင် ၄၅၀ လုံးအထိ ပေါင်း
  const merged = [];
  for (const s of raw) {
    const last = merged[merged.length - 1];
    if (last && last.tag === s.tag && (last.text + " " + s.text).length <= 450) {
      last.text += " " + s.text;
    } else {
      merged.push({ ...s });
    }
  }

  // ၄၅၀ လုံးထက်ရှည်တာကို ခွဲ
  const segments = [];
  for (const s of merged) {
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
app.post("/api/text-translate", async (req, res) => {
  try {
    const { text, sourceLanguage = "auto", targetLanguage = "my" } = req.body || {};

    if (!text || !String(text).trim()) {
      return res.status(400).json({ error: "ဘာသာပြန်မည့်စာသား လိုအပ်ပါသည်။" });
    }
    if (!ai) {
      return res.status(500).json({
        error: "Render Environment ထဲတွင် GEMINI_API_KEY မရှိသေးပါ။"
      });
    }

    const prompt =
      `Translate the following ${languageLabel(sourceLanguage)} text into natural, soft, and clear ${getTarget(targetLanguage).label}. ` +
      `Return ONLY the translated ${getTarget(targetLanguage).label} text without explanations or quotes:\n\n${text}`;

    const translatedText = await generateText(prompt);
    return res.json({ translatedText });
  } catch (err) {
    console.error("Translate error:", err);
    return res.status(500).json({
      error: err.message || "Server ဘက်တွင် ဘာသာပြန် Error တက်နေပါသည်။"
    });
  }
});

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

/* ---------------- အခမဲ့ စကားနားထောင်ခြင်း (Groq Whisper) ---------------- */
async function groqTranscribe(filePath, sourceLanguage) {
  const form = new FormData();
  form.append("file", new Blob([fs.readFileSync(filePath)], { type: "audio/mpeg" }), "audio.mp3");
  form.append("model", process.env.GROQ_STT_MODEL || "whisper-large-v3-turbo");
  form.append("response_format", "json");
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
  return String(data.text || "").trim();
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
  return out.join("\n");
}

async function translateTranscript(transcript, sourceLanguage, targetLanguage = "my") {
  const prompt =
    `Translate the following ${languageLabel(sourceLanguage)} transcript into natural, soft, and clear ${getTarget(targetLanguage).label}, ` +
    `suitable for smooth voice narration. Prefix each sentence line with [M], [F], or [C] if speaker gender/age is clear from context. ` +
    `Write one sentence per line. Return ONLY the ${getTarget(targetLanguage).label} text with no explanations or notes:\n\n${transcript}`;
  try {
    return await generateText(prompt);
  } catch (err) {
    console.error("Gemini text translate failed, using MyMemory:", err.message);
    return await myMemoryTranslate(transcript, sourceLanguage, targetLanguage);
  }
}

async function speechToBurmese(audioPath, audioBase64, opts, job) {
  if (groqKey) {
    try {
      setJob(job, { progress: 20, message: "AI က စကားကို နားထောင်နေပါသည်..." });
      const transcript = await groqTranscribe(audioPath, opts.sourceLanguage);
      if (transcript.length < 2) {
        throw Object.assign(new Error("ဗီဒီယိုထဲတွင် စကားပြောသံ မတွေ့ပါ။"), { noSpeech: true });
      }
      setJob(job, { progress: 28, message: "ယောက်ျား / မိန်းကလေး / ကလေး အသံများ ခွဲခြားဘာသာပြန်နေပါသည်..." });
      return await translateTranscript(transcript, opts.sourceLanguage, opts.targetLanguage);
    } catch (err) {
      if (err.noSpeech) throw err;
      console.error("Groq path failed, falling back to Gemini audio:", err.message);
    }
  }
  setJob(job, { progress: 20, message: "ယောက်ျား၊ မိန်းကလေး၊ ကလေး အသံများကို ခွဲခြားဘာသာပြန်နေပါသည်..." });
  const prompt =
    `Listen carefully to this audio (spoken language: ${languageLabel(opts.sourceLanguage)}). ` +
    `The audio may contain male voices, female voices, and/or child voices. ` +
    `Translate everything that is spoken into natural, soft, and clear ${getTarget(opts.targetLanguage).label}, suitable for smooth voice narration. ` +
    `Prefix each line with [M] for male speaker, [F] for female speaker, or [C] for child speaker based on the voice you hear. ` +
    `Write one sentence per line. Return ONLY the tagged ${getTarget(opts.targetLanguage).label} narration text, with no explanations, timestamps or notes. ` +
    `If there is no speech, return exactly: NO_SPEECH`;
  const text = await generateText([
    { role: "user", parts: [{ inlineData: { mimeType: "audio/mp3", data: audioBase64 } }, { text: prompt }] }
  ]);
  if (/^NO_SPEECH$/i.test(text.trim())) {
    throw new Error("ဗီဒီယိုထဲတွင် စကားပြောသံ မတွေ့ပါ။");
  }
  return text;
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

    // 1) အသံထုတ်
    setJob(job, { progress: 8, message: "ဗီဒီယိုထဲက အသံကို ခွဲထုတ်နေပါသည်..." });
    const audioPath = path.join(dir, "audio.mp3");
    await runFfmpeg(["-i", file.path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", audioPath]);
    const audioBase64 = fs.readFileSync(audioPath).toString("base64");

    // 2) စကားနားထောင်ပြီး ဘာသာပြန်
    const burmese = await speechToBurmese(audioPath, audioBase64, opts, job);
    lap("speech→translated");

    // 3) [M], [F], [C] အလိုက် အသံဖန်တီး
    const segments = parseSpeakerSegments(burmese, opts.voice);
    if (!segments.length) {
      throw new Error("ဘာသာပြန်စာသားမှ အသံဖန်တီးရန် စာကြောင်း မတွေ့ပါ။");
    }
    console.log(`[job ${job.id.slice(0, 6)}] TTS segments: ${segments.length}`);

    const pcmParts = new Array(segments.length);
    let nextIdx = 0, doneCount = 0;
    const worker = async () => {
      while (true) {
        const i = nextIdx++;
        if (i >= segments.length) return;
        const seg = segments[i];
        pcmParts[i] = await synthChunk(seg.text, seg.voiceKey, dir, i, opts.targetLanguage, seg.tag);
        doneCount++;
        const label = seg.tag === "C" ? "ကလေးအသံ" : seg.tag === "F" ? "မိန်းကလေးအသံ" : "ယောက်ျားအသံ";
        setJob(job, {
          progress: 35 + Math.round((doneCount / segments.length) * 45),
          message: `${label} ဖန်တီးနေပါသည်... (${doneCount}/${segments.length})`
        });
      }
    };
    await Promise.all(Array.from({ length: Math.min(TTS_CONCURRENCY, segments.length) }, worker));
    lap("TTS");

    const ttsRate = pcmParts[0].rate;
    const pcm = Buffer.concat(pcmParts.map((x) => x.pcm));
    const pcmPath = path.join(dir, "tts.pcm");
    const ttsWav = path.join(dir, "tts.wav");
    fs.writeFileSync(pcmPath, pcm);
    await runFfmpeg(["-f", "s16le", "-ar", String(ttsRate), "-ac", "1", "-i", pcmPath, ttsWav]);

    // 4) အသံနဲ့ ဗီဒီယိုကို ပိုနူးညံ့ချောမွေ့အောင် ပေါင်း
    setJob(job, { progress: 85, message: "အသံအသစ်ကို ဗီဒီယိုနဲ့ ညှိပြီး ပေါင်းနေပါသည်..." });
    const ttsDuration = pcm.length / (ttsRate * 2);
    let tempo = ttsDuration / info.duration;
    let note = "";
    let tempoFilter = "anull";
    if (tempo > 1.02) {
      if (tempo > MAX_TEMPO) {
        // အသံက ဗီဒီယိုထက် ရှည်လွန်းလို့ အမြန်ညှိပြီးတောင် နောက်ဆုံးပိုင်း ဖြတ်ခံရမယ်
        note = " (အသံက ဗီဒီယိုထက် ရှည်လွန်းလို့ နောက်ဆုံးပိုင်း အနည်းငယ် ဖြတ်ခံရပါသည်)";
        tempo = MAX_TEMPO;
      } else {
        note = " (အသံကို ဗီဒီယိုအရှည်နဲ့ ကိုက်အောင် ညင်သာစွာ ညှိပေးထားသည်)";
      }
      tempoFilter = `atempo=${tempo.toFixed(3)}`;
    }

    const fadeOutStart = Math.max(0, info.duration - 0.15);
    const filter = `[1:a]${tempoFilter},afade=t=in:st=0:d=0.05,afade=t=out:st=${fadeOutStart.toFixed(2)}:d=0.15,apad[a]`;

    const outPath = path.join(dir, "output.mp4");
    const baseArgs = [
      "-i", file.path,
      "-i", ttsWav,
      "-filter_complex", filter,
      "-map", "0:v:0",
      "-map", "[a]"
    ];
    const tailArgs = [
      "-c:a", "aac", "-b:a", "128k",
      "-t", String(info.duration),
      "-movflags", "+faststart",
      outPath
    ];

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
      message: "ပြီးစီးပါပြီ။" + note,
      outPath,
      outDir: dir,
      translatedText: burmese
    });
  } catch (err) {
    console.error("Job failed:", err);
    setJob(job, { status: "error", progress: 0, error: err.message || "Video ဘာသာပြန်မအောင်မြင်ပါ။" });
    safeRm(dir);
  } finally {
    safeRm(file.path);
  }
}

app.post("/api/translate", upload.single("video"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "Video ဖိုင် မပါပါ။" });
  }
  if (!ai && !groqKey) {
    safeRm(req.file.path);
    return res.status(500).json({ error: "Render Environment ထဲတွင် GROQ_API_KEY သို့မဟုတ် GEMINI_API_KEY မရှိသေးပါ။" });
  }

  const jobId = crypto.randomUUID();
  const job = {
    id: jobId,
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

  processJob(job, req.file, opts);

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
    if (now - job.updatedAt > 30 * 60 * 1000) {
      safeRm(job.outDir);
      jobs.delete(id);
    }
  }
}, 5 * 60 * 1000);

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
