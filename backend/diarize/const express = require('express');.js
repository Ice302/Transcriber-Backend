const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const fetch = require('node-fetch');
const { exec, execFileSync } = require('child_process');

const app = express();
app.use(cors());
app.use(express.json());

// --- Setup from both versions ---
const RECORDINGS_DIR = path.join(__dirname, 'recordings');
if (!fs.existsSync(RECORDINGS_DIR)) fs.mkdirSync(RECORDINGS_DIR, { recursive: true });

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, RECORDINGS_DIR),
  filename: (req, file, cb) => cb(null, Date.now() + '-' + (file.originalname || 'recording.webm'))
});
const upload = multer({ storage });

// --- Main Transcription Endpoint ---
app.post('/api/upload', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded.' });
  }
  
  const filepath = req.file.path;
  const engine = process.env.STT_ENGINE || 'whisper'; // Default to whisper
  let rawTranscript = '';

  try {
    // --- Reusable Engine Switch ---
    if (engine === 'whisper') {
      console.log('Using Whisper STT Engine...');
      rawTranscript = await runWhisper(filepath);
    } else if (engine === 'hf') {
      // Your existing Hugging Face logic
      // ...
    } else if (engine === 'local') {
      // Your existing local script logic
      // ...
    }

    // --- Reusable Post-Processing ---
    console.log('Cleaning transcript with LLM...');
    const cleanedTranscript = await callLLMPostProcess(rawTranscript, null);

    res.json({ transcript: cleanedTranscript });

  } catch (e) {
    res.status(500).json({ error: e.message });
  } finally {
    // Clean up the uploaded audio file after processing
    if (fs.existsSync(filepath)) {
      fs.unlinkSync(filepath);
    }
  }
});

// Helper function for our new Whisper logic
function runWhisper(audioFilePath) {
  return new Promise((resolve, reject) => {
    const command = `whisper "${audioFilePath}" --model tiny.en --language English --output_format json --output_dir "${RECORDINGS_DIR}"`;
    
    exec(command, (error, stdout, stderr) => {
      if (error) {
        return reject(new Error(`Whisper execution failed: ${stderr}`));
      }

      const jsonFileName = `${path.basename(audioFilePath, path.extname(audioFilePath))}.json`;
      const jsonFilePath = path.join(RECORDINGS_DIR, jsonFileName);

      fs.readFile(jsonFilePath, 'utf8', (err, data) => {
        // Clean up the JSON file immediately
        if (fs.existsSync(jsonFilePath)) {
          fs.unlinkSync(jsonFilePath);
        }
        if (err) {
          return reject(new Error(`Failed to read Whisper output: ${err.message}`));
        }
        const transcriptData = JSON.parse(data);
        resolve(transcriptData.text.trim());
      });
    });
  });
}

// --- Your Reusable Post-Processing Function ---
async function callLLMPostProcess(rawTranscript, segments) {
  const COHERE_API_KEY = process.env.COHERE_API_KEY;
  if (!COHERE_API_KEY) {
    console.warn('COHERE_API_KEY not set. Skipping post-processing.');
    return rawTranscript || '';
  }

  let prompt = 'You are an expert assistant that cleans up raw, unpunctuated speech-to-text transcripts. Your task is to add appropriate punctuation (periods, commas, question marks), fix capitalization, and correct minor grammatical errors to make the text readable and professional. Do not summarize or change the meaning of the words. Just return the cleaned transcript text.';
  prompt += '\n\nRAW TRANSCRIPT:\n' + (rawTranscript || '');

  const resp = await fetch('https://api.cohere.com/v1/generate', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${COHERE_API_KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'command',
      prompt,
      max_tokens: 2048,
      temperature: 0.3,
    })
  });

  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error('Cohere API failed: ' + txt);
  }
  const json = await resp.json();
  return (json?.generations?.[0]?.text || '').trim();
}


// --- Your Reusable Diarization Endpoint (can be used in a future step) ---
app.post('/api/diarize', async (req, res) => {
    // ... your existing /api/diarize logic ...
});


const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log('Backend listening on', PORT));