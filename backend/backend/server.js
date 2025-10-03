require('dotenv').config();
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const fetch = require('node-fetch');
const FormData = require('form-data');
const { execFile, execFileSync } = require('child_process'); // Re-added for 'local' engine

const app = express();
app.use(cors());
app.use(express.json());

const RECORDINGS_DIR = path.join(__dirname, 'recordings');
if (!fs.existsSync(RECORDINGS_DIR)) fs.mkdirSync(RECORDINGS_DIR, { recursive: true });

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, RECORDINGS_DIR),
  filename: (req, file, cb) => cb(null, Date.now() + '-' + (file.originalname || 'recording.webm'))
});
const upload = multer({ storage });

app.post('/api/upload', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded.' });
  }
  
  const filepath = req.file.path;
  // RESTORED: Engine switch defaults to our improved whisper setup
  const engine = process.env.STT_ENGINE || 'whisper'; 
  let rawTranscript = '';

  try {
    if (engine === 'whisper') {
      console.log('Using improved Whisper AI server...');
      rawTranscript = await runWhisper(filepath);
    } else if (engine === 'hf') {
      console.log('Using Hugging Face API...');
      const HF_API = process.env.HF_API_KEY;
      if (HF_API) {
        const MODEL = process.env.HF_STT_MODEL || 'openai/whisper-large-v2';
        const buffer = fs.readFileSync(filepath);
        const r = await fetch(`https://api-inference.huggingface.co/models/${MODEL}`, { method: 'POST', headers: { Authorization: `Bearer ${HF_API}` }, body: buffer });
        const json = await r.json();
        if (json && json.text) rawTranscript = json.text;
      }
    } else if (engine === 'local') {
      console.log('Using local shell script...');
      const out = execFileSync(path.join(__dirname, 'transcribe_local.sh'), [filepath], { encoding: 'utf8', maxBuffer: 1024 * 1024 * 10 });
      rawTranscript = (out || '').trim();
    }

    console.log('Cleaning transcript with LLM...');
    const cleanedTranscript = await callLLMPostProcess(rawTranscript, null);

    res.json({ transcript: cleanedTranscript });

  } catch (e) {
    res.status(500).json({ error: e.message });
  } finally {
    if (fs.existsSync(filepath)) {
      fs.unlinkSync(filepath);
    }
  }
});

// This function now calls the fast Python AI server
async function runWhisper(audioFilePath) {
  const formData = new FormData();
  formData.append('file', fs.createReadStream(audioFilePath));

  const response = await fetch('http://localhost:5001/transcribe', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'Whisper AI server failed');
  }

  const result = await response.json();
  return result.transcript;
}

// RESTORED: Your complete /api/diarize endpoint
app.post('/api/diarize', async (req, res) => {
  try {
    const { path: audioPath, rawTranscript } = req.body || {};
    const result = { ok: true, segments: null, cleanedTranscript: null };

    const diarizeScript = path.join(__dirname, '..', 'diarize', 'diarize.py');
    if (audioPath && fs.existsSync(diarizeScript)) {
      execFile('python', [diarizeScript, audioPath], { maxBuffer: 1024 * 1024 * 10 }, (err, stdout) => {
        if (err) {
          callLLMPostProcess(rawTranscript || '', null).then((out) => res.json({ ok: true, segments: null, cleanedTranscript: out })).catch((err2) => res.status(500).json({ error: '' + err2 }));
          return;
        }
        try {
          const segments = JSON.parse(stdout);
          result.segments = segments;
          callLLMPostProcess(rawTranscript || '', segments).then((cleaned) => {
            result.cleanedTranscript = cleaned;
            res.json(result);
          }).catch((err2) => res.status(500).json({ error: 'LLM postprocess failed: ' + err2 }));
        } catch (e) {
          callLLMPostProcess(rawTranscript || '', null).then((out) => res.json({ ok: true, segments: null, cleanedTranscript: out })).catch((err2) => res.status(500).json({ error: '' + err2 }));
        }
      });
      return;
    }

    const cleaned = await callLLMPostProcess(rawTranscript || '', null);
    result.cleanedTranscript = cleaned;
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// This is the improved post-processing function
async function callLLMPostProcess(rawTranscript, segments) {
  const COHERE_API_KEY = process.env.COHERE_API_KEY;
  // Read the model from the .env file, with a fallback
  const COHERE_MODEL = process.env.COHERE_MODEL || 'command-r'; 

  if (!COHERE_API_KEY) {
    console.warn('COHERE_API_KEY not set. Skipping post-processing.');
    return rawTranscript || '';
  }

  const prompt = `
You are an expert assistant that cleans up raw, unpunctuated speech-to-text transcripts. Your task is to add appropriate punctuation (periods, commas, question marks), fix capitalization, and correct minor grammatical errors to make the text readable and professional. Do not summarize, rephrase, or change the meaning of the words.

Here is an example:
RAW: "ok so what we need to do first is install the dependencies then we can run the server is that correct"
CLEANED: "Okay, so what we need to do first is install the dependencies. Then, we can run the server. Is that correct?"

Assume that you will have to handle two or three languages at once. Detect the language automatically and punctuate accordingly.
Trranslate non-English parts to English and tag them using [LANGUAGE] tags.

Now, clean the following transcript.

RAW TRANSCRIPT:
${rawTranscript || ''}
  `.trim();

  const resp = await fetch('https://api.cohere.com/v1/chat', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${COHERE_API_KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: COHERE_MODEL, // Use the model from the .env file
      message: prompt,
      temperature: 0.2,
    })
  });

  // --- NEW: Better Error Handling ---
  if (!resp.ok) {
    // Check for rate limiting specifically
    if (resp.status === 429) {
      throw new Error('Cohere API failed: You have exceeded your request limit. Please wait and try again.');
    }
    
    // Try to parse a detailed error message from Cohere
    let errorDetails = await resp.text();
    try {
      const errorJson = JSON.parse(errorDetails);
      errorDetails = errorJson.message || JSON.stringify(errorJson);
    } catch (e) {
      // Not a JSON error, just use the raw text
    }
    throw new Error(`Cohere API failed with status ${resp.status}: ${errorDetails}`);
  }

  const json = await resp.json();
  
  return (json?.text || '').trim();
}

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log('🚀 Backend listening on', PORT));