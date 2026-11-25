import express from "express";
import cors from "cors";
import { GoogleGenAI } from "@google/genai";

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: "10mb" }));

const apiKey = process.env.GEMINI_API_KEY;
const ai = apiKey ? new GoogleGenAI({ apiKey }) : null;

// Render Environment ထဲမှာ GEMINI_MODEL ထည့်ပြီး model ပြောင်းလို့ရသည်
const MODELS = [
  process.env.GEMINI_MODEL,
  "gemini-3.7-flash",
  "gemini-3.6-flash",
  "gemini-2.5-flash"
].filter(Boolean);

app.get("/", (req, res) => {
  res.json({ status: "ok", service: "SOE AI TRANSLATOR Backend" });
});

app.get("/health", (req, res) => {
  res.json({ ok: true, hasApiKey: Boolean(apiKey), models: MODELS });
});

function languageLabel(code = "auto") {
  if (code === "th") return "Thai";
  if (code === "en") return "English";
  if (String(code).startsWith("zh")) return "Chinese";
  return "auto-detected language";
}

async function translateWithFallback(prompt) {
  let lastError;
  for (const model of MODELS) {
    try {
      const response = await ai.models.generateContent({ model, contents: prompt });
      const text = (response.text || "").trim();
      if (text) return text;
      lastError = new Error("Gemini မှ ရလဒ်အလွတ် ပြန်လာပါသည်။");
    } catch (err) {
      console.error(`Model ${model} failed:`, err.message);
      lastError = err;
    }
  }
  throw lastError || new Error("Gemini model မရပါ။");
}

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

    const translatedText = await translateWithFallback(prompt);
    return res.json({ translatedText });
  } catch (err) {
    console.error("Translate error:", err);
    return res.status(500).json({
      error: err.message || "Server ဘက်တွင် ဘာသာပြန် Error တက်နေပါသည်။"
    });
  }
});

app.post("/api/translate", (req, res) => {
  return res.status(501).json({
    message: "Video ကို Server က လက်ခံပါပြီ။ AI Video ဘာသာပြန်စနစ် မချိတ်ဆက်ရသေးပါ။"
  });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
