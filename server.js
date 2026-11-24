import express from "express";
import cors from "cors";
import { GoogleGenAI } from "@google/genai";

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: "10mb" }));

const apiKey = process.env.GEMINI_API_KEY;
const ai = apiKey ? new GoogleGenAI({ apiKey }) : null;

app.get("/", (req, res) => {
  res.json({ status: "ok", service: "SOE AI TRANSLATOR Backend" });
});

app.get("/health", (req, res) => {
  res.json({ ok: true, hasApiKey: Boolean(apiKey) });
});

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

    const langLabel =
      sourceLanguage === "th" ? "Thai" :
      sourceLanguage === "zh" ? "Chinese" :
      sourceLanguage === "en" ? "English" : "auto-detected language";

    const response = await ai.models.generateContent({
      model: "gemini-3.8-flash",
      contents: `Translate the following ${langLabel} text into natural, clear Myanmar (Burmese). Return ONLY the translated Myanmar text without explanations or quotes:\n\n${text}`
    });

    const translatedText = (response.text || "").trim();
    if (!translatedText) {
      return res.status(500).json({ error: "Gemini မှ ဘာသာပြန်ရလဒ် ထွက်မလာပါ။" });
    }

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
