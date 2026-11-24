
const express = require("express");
const cors = require("cors");
const multer = require("multer");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 100 * 1024 * 1024 }
});

app.get("/", (req, res) => {
  res.json({
    message: "SOE AI TRANSLATOR",
    status: "Server is ready"
  });
});

// AI စာသားဘာသာပြန်စနစ်
app.post("/api/text-translate", async (req, res) => {
  try {
    const { text, sourceLanguage } = req.body;

    if (!text || !text.trim()) {
      return res.status(400).json({
        error: "ဘာသာပြန်မည့်စာသား ထည့်ပါ။"
      });
    }

    const apiKey = process.env.GEMINI_API_KEY;

    if (!apiKey) {
      return res.status(503).json({
        error: "AI API Key မထည့်ရသေးပါ။"
      });
    }

    const languages = {
      en: "English",
      zh: "Chinese",
      th: "Thai",
      auto: "Automatically detect the language"
    };

    const source = languages[sourceLanguage] || languages.auto;

    const response = await fetch(
  "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent",
  {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-goog-api-key": apiKey
    },
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          contents: [{
            parts: [{
              text:
                "Translate the following " + source +
                " text into natural Myanmar Burmese. " +
                "Return only the Burmese translation:\n\n" + text
            }]
          }]
        })
      }
    );

    const result = await response.json();

    if (!response.ok) {
  console.error("Gemini API Error:", response.status, result);

  return res.status(502).json({
    error: "AI ဘာသာပြန်မအောင်မြင်ပါ။ Render Logs ကိုစစ်ပါ။"
  });
}

    const translation =
      result.candidates?.[0]?.content?.parts
        ?.map(part => part.text || "")
        .join("") || "";

    res.json({
      originalText: text,
      translatedText: translation
    });

  } catch (error) {
    res.status(500).json({
      error: "Server အမှားဖြစ်နေပါသည်။"
    });
  }
});

// Video လက်ခံမှုကို ယာယီစမ်းသပ်ရန်
app.post("/api/translate", upload.single("video"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({
      error: "Please upload a video"
    });
  }

  res.status(501).json({
    message: "Video received",
    status: "Video dubbing is not connected yet"
  });
});

app.listen(PORT, "0.0.0.0", () => {
  console.log("SOE AI TRANSLATOR server running on port " + PORT);
});
