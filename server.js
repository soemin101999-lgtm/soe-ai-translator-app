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

const VOICES = { male: "Charon", female: "Kore" };
const MAX_VIDEO_SECONDS = 7 * 60;      // 5 မိနစ်ဗီဒီယိုအတွက် အနည်းငယ်ပိုခွင့်ပြု
const MAX_UPLOAD_BYTES = 300 * 1024 * 1024;
const TTS_SAMPLE_RATE = 24000;
const MAX_TEMPO = 2.0;

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
  if (code === "th") return "Thai";
  if (code === "en") return "English";
  if (String(code).startsWith("zh")) return "Chinese";
  return "auto-detected language";
}

let workingTextModel = null;
async function generateText(contents) {
  if (!ai) throw new Error("GEMINI_API_KEY မရှိပါ။");
  const order = [...new Set([workingTextModel, ...TEXT_MODELS].filter(Boolean))];
  const errors = [];
  for (const model of order) {
    try {
      const response = await ai.models.generateContent({ model, contents });
      const text = (response.text || "").trim();
      if (text) {
        workingTextModel = model;
        return text;
      }
      errors.push(`${model}: ရလဒ်အလွတ်`);
    } catch (err) {
      console.error(`Model ${model} failed:`, err.message);
      errors.push(`${model}: ${shortError(err)}`);
    }
  }
  throw new Error("Gemini မအောင်မြင်ပါ။ " + (errors.join(" | ") || "model မရပါ။"));
}

// WAV (RIFF) ဖြစ်ရင် header ဖယ်ပြီး raw PCM ယူ၊ မဟုတ်ရင် PCM အတိုင်းသုံး
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
            speechConfig: {
              voiceConfig: { prebuiltVoiceConfig: { voiceName } }
            }
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

/* ---------------- အခမဲ့ မြန်မာအသံ (Microsoft Edge TTS) ----------------
   API key မလို၊ quota မရှိ။ မရမှ Gemini TTS ကို အရန်သုံးသည်။ */
const EDGE_VOICES = { male: "my-MM-ThihaNeural", female: "my-MM-NilarNeural" };

function escapeXml(t) {
  return t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

async function edgeTtsChunk(text, voiceKey, dir, idx) {
  const tts = new MsEdgeTTS();
  await tts.setMetadata(EDGE_VOICES[voiceKey] || EDGE_VOICES.male, OUTPUT_FORMAT.WEBM_24KHZ_16BIT_MONO_OPUS);
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

// Edge (အခမဲ့) ကို အရင်၊ မရမှ Gemini
async function synthChunk(text, voiceKey, dir, idx) {
  try {
    return await edgeTtsChunk(text, voiceKey, dir, idx);
  } catch (err) {
    console.error("Edge TTS failed, falling back to Gemini:", err.message);
    if (!ai) throw err;
    return await ttsChunk(text, VOICES[voiceKey] || VOICES.male);
  }
}

function splitForTts(text, max = 600) {
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

/* ---------------- diagnostic: Gemini ချိတ်ဆက်မှု စစ်ရန် ----------------
   Browser မှာ  https://soe-ai-translator-app.onrender.com/api/test-gemini  ကို ဖွင့်ကြည့်ပါ */
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
    const { text, sourceLanguage = "auto" } = req.body || {};

    if (!text || !String(text).trim()) {
      return res.status(400).json({ error: "ဘာသာပြန်မည့်စာသား လိုအပ်ပါသည်။" });
    }
    if (!ai) {
      return res.status(500).json({
        error: "Render Environment ထဲတွင် GEMINI_API_KEY မရှိသေးပါ။"
      });
    }

    const prompt =
      `Translate the following ${languageLabel(sourceLanguage)} text into natural, clear Myanmar (Burmese). ` +
      `Return ONLY the translated Myanmar text without explanations or quotes:\n\n${text}`;

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

/* ---------------- အခမဲ့ စကားနားထောင်ခြင်း (Groq Whisper) ----------------
   Gemini quota မသုံးဘဲ စကားကို စာသားပြောင်း။ ပြီးမှ စာသားသက်သက်ကို မြန်မာပြန်။ */
async function groqTranscribe(filePath, sourceLanguage) {
  const form = new FormData();
  form.append("file", new Blob([fs.readFileSync(filePath)], { type: "audio/mpeg" }), "audio.mp3");
  form.append("model", process.env.GROQ_STT_MODEL || "whisper-large-v3-turbo");
  form.append("response_format", "json");
  form.append("temperature", "0");
  const lang = sourceLanguage === "en" ? "en" : sourceLanguage === "th" ? "th" :
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

// MyMemory (အခမဲ့၊ key မလို) — နောက်ဆုံးအရန်
async function myMemoryTranslate(text, sourceLanguage) {
  const from = sourceLanguage === "auto" ? "Autodetect" : sourceLanguage === "zh" ? "zh-CN" : sourceLanguage;
  const out = [];
  for (const chunk of splitForTts(text, 450)) {
    const url = "https://api.mymemory.translated.net/get?q=" + encodeURIComponent(chunk) +
      "&langpair=" + encodeURIComponent(from + "|my");
    const r = await fetch(url, { signal: AbortSignal.timeout(20000) });
    const d = await r.json().catch(() => ({}));
    const t = d?.responseData?.translatedText;
    if (!r.ok || !t || /MYMEMORY WARNING/i.test(t)) throw new Error("MyMemory ဘာသာပြန်မရပါ။");
    out.push(t);
  }
  return out.join("\n");
}

async function translateTranscript(transcript, sourceLanguage) {
  const prompt =
    `Translate the following ${languageLabel(sourceLanguage)} transcript into natural, clear Myanmar (Burmese), ` +
    `suitable for voice narration. Write one sentence per line. ` +
    `Return ONLY the Myanmar text with no explanations or notes:\n\n${transcript}`;
  try {
    return await generateText(prompt);
  } catch (err) {
    console.error("Gemini text translate failed, using MyMemory:", err.message);
    return await myMemoryTranslate(transcript, sourceLanguage);
  }
}

// ဗီဒီယိုအသံ → မြန်မာစာသား (Groq အရင်၊ မရမှ Gemini audio)
async function speechToBurmese(audioPath, audioBase64, opts, job) {
  if (groqKey) {
    try {
      setJob(job, { progress: 20, message: "AI က စကားကို နားထောင်နေပါသည်..." });
      const transcript = await groqTranscribe(audioPath, opts.sourceLanguage);
      if (transcript.length < 2) {
        throw Object.assign(new Error("ဗီဒီယိုထဲတွင် စကားပြောသံ မတွေ့ပါ။"), { noSpeech: true });
      }
      setJob(job, { progress: 28, message: "မြန်မာလို ဘာသာပြန်နေပါသည်..." });
      return await translateTranscript(transcript, opts.sourceLanguage);
    } catch (err) {
      if (err.noSpeech) throw err;
      console.error("Groq path failed, falling back to Gemini audio:", err.message);
    }
  }
  setJob(job, { progress: 20, message: "AI က စကားကို နားထောင်ပြီး မြန်မာလို ဘာသာပြန်နေပါသည်..." });
  const prompt =
    `Listen to this audio (spoken language: ${languageLabel(opts.sourceLanguage)}). ` +
    `Translate everything that is spoken into natural, clear Myanmar (Burmese), suitable for voice narration. ` +
    `Write one sentence per line. Return ONLY the Myanmar narration text, with no explanations, timestamps or notes. ` +
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

    // 2) စကားနားထောင်ပြီး မြန်မာလို ဘာသာပြန်
    const burmese = await speechToBurmese(audioPath, audioBase64, opts, job);

    // 3) မြန်မာအသံ ဖန်တီး
    const chunks = splitForTts(burmese);
    const pcmParts = [];
    for (let i = 0; i < chunks.length; i++) {
      setJob(job, {
        progress: 35 + Math.round((i / chunks.length) * 45),
        message: `မြန်မာအသံ ဖန်တီးနေပါသည်... (${i + 1}/${chunks.length})`
      });
      pcmParts.push(await synthChunk(chunks[i], opts.voice, dir, i));
    }
    const ttsRate = pcmParts[0].rate;
    const pcm = Buffer.concat(pcmParts.map((x) => x.pcm));
    const pcmPath = path.join(dir, "tts.pcm");
    const ttsWav = path.join(dir, "tts.wav");
    fs.writeFileSync(pcmPath, pcm);
    await runFfmpeg(["-f", "s16le", "-ar", String(ttsRate), "-ac", "1", "-i", pcmPath, ttsWav]);

    // 4) အသံနဲ့ ဗီဒီယိုကို ပေါင်း
    setJob(job, { progress: 85, message: "အသံအသစ်ကို ဗီဒီယိုနဲ့ ပေါင်းနေပါသည်..." });
    const ttsDuration = pcm.length / (ttsRate * 2);
    let tempo = ttsDuration / info.duration;
    let note = "";
    let ttsFilter = "anull";
    if (tempo > 1.02) {
      if (tempo > MAX_TEMPO) {
        note = " (မြန်မာအသံက ဗီဒီယိုထက် အလွန်ရှည်နေလို့ နောက်ပိုင်း ဖြတ်ထားနိုင်ပါသည်)";
        tempo = MAX_TEMPO;
      } else {
        note = " (အသံကို ဗီဒီယိုအရှည်နဲ့ ကိုက်အောင် အမြန်ပြုပြင်ထားသည်)";
      }
      ttsFilter = `atempo=${tempo.toFixed(3)}`;
    }

    // မူရင်းအသံ (နောက်ခံအပါအဝင်) ကို လုံးဝဖယ်ပြီး မြန်မာအသံသာ ထည့်သည်
    const filter = `[1:a]${ttsFilter},apad[a]`;

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
    voice: req.body?.voice || "male"
  };

  // နောက်ကွယ်မှာ လုပ်ဆောင်
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
  // sendFile က Range request (iPhone Safari အတွက် လိုအပ်) ကို အလိုအလျောက် ကိုင်တွယ်
  if (req.query.inline) {
    return res.type("video/mp4").sendFile(job.outPath);
  }
  return res.download(job.outPath, "soe-translated.mp4");
});

// ၃၀ မိနစ်ကျော်တဲ့ အလုပ်တွေကို ရှင်း
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
