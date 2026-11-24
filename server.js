const express = require("express");
const cors = require("cors");
const multer = require("multer");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 100 * 1024 * 1024
  }
});

app.get("/", (req, res) => {
  res.json({
    message: "SOE AI TRANSLATOR",
    status: "Server is ready"
  });
});

app.post("/api/translate", upload.single("video"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({
      error: "Please upload a video"
    });
  }

  res.status(501).json({
    message: "Video received",
    status: "AI translation engine is not connected yet"
  });
});

app.listen(PORT, "0.0.0.0", () => {
  console.log("SOE AI TRANSLATOR server running on port " + PORT);
});

