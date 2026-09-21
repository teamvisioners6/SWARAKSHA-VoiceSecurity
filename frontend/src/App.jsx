import { useState } from "react";

const translations = {
  en: {
    language: "Language",
    status: "AI DETECTION SYSTEM",
    eyebrow: "VOICE SECURITY INTELLIGENCE",
    title1: "Detect",
    title2: "AI-generated",
    title3: "voices in seconds.",
    subtitle:
      "Upload a voice recording and let VIGILVOICE analyze voice authenticity and scam risk.",
    analysis: "Voice Analysis",
    analysisDesc: "Upload an audio recording for deepfake detection.",
    engine: "SPECTRA-AASIST3",
    choose: "Choose an audio file",
    supported: "WAV, MP3, FLAC, OGG, M4A supported",
    analyze: "ANALYZE VOICE",
    analyzing: "ANALYZING VOICE...",
    complete: "ANALYSIS COMPLETE",
    result: "Detection Result",
    verdict: "VOICE VERDICT",
    risk: "FINAL RISK SCORE",
    real: "REAL VOICE",
    spoof: "AI SPOOF",
    suspicious: "SUSPICIOUS",
    highRisk: "HIGH RISK",
    lowRisk: "LOW RISK",
    spoofMessage:
      "Synthetic or manipulated voice characteristics detected.",
    realMessage:
      "Voice characteristics are consistent with genuine speech.",
    suspiciousMessage:
      "Voice requires additional verification.",
    aiScore: "AI VOICE SCORE",
    transcript: "TRANSCRIPT",
    scamRisk: "SCAM RISK",
    model: "MODEL",
    duration: "DURATION",
    segments: "SEGMENTS",
    reasons: "DETECTION REASONS",
    audioInput: "Audio Input",
    recording: "Voice recording",
    aasist: "AASIST3",
    antiSpoof: "Primary anti-spoofing",
    riskEngine: "Risk Engine",
    threat: "Threat assessment",
    footer: "VIGILVOICE • AI-POWERED VOICE AUTHENTICITY & SCAM PROTECTION",
    selectFile: "Please select an audio file first.",
    failed: "Voice analysis failed.",
    clear: "NEW ANALYSIS",
  },

  ta: {
    language: "மொழி",
    status: "AI கண்டறிதல் அமைப்பு",
    eyebrow: "குரல் பாதுகாப்பு நுண்ணறிவு",
    title1: "AI உருவாக்கிய",
    title2: "குரல்களை",
    title3: "சில நொடிகளில் கண்டறியுங்கள்.",
    subtitle:
      "குரல் பதிவை பதிவேற்றி அதன் உண்மைத்தன்மை மற்றும் மோசடி அபாயத்தை VIGILVOICE பகுப்பாய்வு செய்யும்.",
    analysis: "குரல் பகுப்பாய்வு",
    analysisDesc: "Deepfake கண்டறிதலுக்காக ஆடியோ பதிவை பதிவேற்றவும்.",
    engine: "SPECTRA-AASIST3",
    choose: "ஆடியோ கோப்பைத் தேர்ந்தெடுக்கவும்",
    supported: "WAV, MP3, FLAC, OGG, M4A ஆதரிக்கப்படுகிறது",
    analyze: "குரலை பகுப்பாய்வு செய்",
    analyzing: "பகுப்பாய்வு நடைபெறுகிறது...",
    complete: "பகுப்பாய்வு முடிந்தது",
    result: "கண்டறிதல் முடிவு",
    verdict: "குரல் முடிவு",
    risk: "இறுதி ஆபத்து மதிப்பெண்",
    real: "உண்மையான குரல்",
    spoof: "AI போலி குரல்",
    suspicious: "சந்தேகத்திற்குரியது",
    highRisk: "அதிக ஆபத்து",
    lowRisk: "குறைந்த ஆபத்து",
    spoofMessage:
      "செயற்கை அல்லது மாற்றியமைக்கப்பட்ட குரல் கண்டறியப்பட்டது.",
    realMessage:
      "குரல் உண்மையானதாகத் தெரிகிறது.",
    suspiciousMessage:
      "மேலும் சரிபார்ப்பு தேவை.",
    aiScore: "AI குரல் மதிப்பெண்",
    transcript: "உரை",
    scamRisk: "மோசடி ஆபத்து",
    model: "மாதிரி",
    duration: "நேரம்",
    segments: "பகுதிகள்",
    reasons: "கண்டறிதல் காரணங்கள்",
    audioInput: "ஆடியோ உள்ளீடு",
    recording: "குரல் பதிவு",
    aasist: "AASIST3",
    antiSpoof: "முதன்மை anti-spoofing",
    riskEngine: "Risk Engine",
    threat: "அச்சுறுத்தல் மதிப்பீடு",
    footer: "VIGILVOICE • AI குரல் உண்மைத்தன்மை & மோசடி பாதுகாப்பு",
    selectFile: "முதலில் ஒரு ஆடியோ கோப்பைத் தேர்ந்தெடுக்கவும்.",
    failed: "குரல் பகுப்பாய்வு தோல்வியடைந்தது.",
    clear: "புதிய பகுப்பாய்வு",
  },

  hi: {
    language: "भाषा",
    status: "AI पहचान प्रणाली",
    eyebrow: "वॉइस सुरक्षा इंटेलिजेंस",
    title1: "AI द्वारा बनाई गई",
    title2: "आवाज़ों को",
    title3: "कुछ ही सेकंड में पहचानें।",
    subtitle:
      "वॉइस रिकॉर्डिंग अपलोड करें और VIGILVOICE उसकी वास्तविकता तथा स्कैम जोखिम का विश्लेषण करेगा।",
    analysis: "वॉइस विश्लेषण",
    analysisDesc: "डीपफेक पहचान के लिए ऑडियो रिकॉर्डिंग अपलोड करें।",
    engine: "SPECTRA-AASIST3",
    choose: "ऑडियो फ़ाइल चुनें",
    supported: "WAV, MP3, FLAC, OGG, M4A समर्थित हैं",
    analyze: "वॉइस का विश्लेषण करें",
    analyzing: "वॉइस का विश्लेषण हो रहा है...",
    complete: "विश्लेषण पूरा हुआ",
    result: "पहचान परिणाम",
    verdict: "वॉइस निष्कर्ष",
    risk: "अंतिम जोखिम स्कोर",
    real: "वास्तविक आवाज़",
    spoof: "AI नकली आवाज़",
    suspicious: "संदिग्ध",
    highRisk: "उच्च जोखिम",
    lowRisk: "कम जोखिम",
    spoofMessage:
      "कृत्रिम या बदली हुई आवाज़ के संकेत मिले।",
    realMessage:
      "आवाज़ वास्तविक प्रतीत होती है।",
    suspiciousMessage:
      "अतिरिक्त सत्यापन आवश्यक है।",
    aiScore: "AI वॉइस स्कोर",
    transcript: "ट्रांसक्रिप्ट",
    scamRisk: "स्कैम जोखिम",
    model: "मॉडल",
    duration: "अवधि",
    segments: "सेगमेंट",
    reasons: "पहचान के कारण",
    audioInput: "ऑडियो इनपुट",
    recording: "वॉइस रिकॉर्डिंग",
    aasist: "AASIST3",
    antiSpoof: "प्राथमिक एंटी-स्पूफिंग",
    riskEngine: "Risk Engine",
    threat: "खतरे का आकलन",
    footer: "VIGILVOICE • AI वॉइस प्रामाणिकता एवं स्कैम सुरक्षा",
    selectFile: "कृपया पहले एक ऑडियो फ़ाइल चुनें।",
    failed: "वॉइस विश्लेषण विफल हुआ।",
    clear: "नया विश्लेषण",
  },
};

function App() {
  const [language, setLanguage] = useState("en");
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const t = translations[language];

  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0];

    if (!selectedFile) return;

    setFile(selectedFile);
    setResult(null);
    setError("");
  };

  const analyzeVoice = async () => {
    if (!file) {
      setError(t.selectFile);
      return;
    }

    setLoading(true);
    setResult(null);
    setError("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(
        "http://192.168.22.187:8000/api/analyze",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || t.failed);
      }

      setResult(data);
    } catch (err) {
      setError(err.message || t.failed);
    } finally {
      setLoading(false);
    }
  };

  const clearAnalysis = () => {
    setFile(null);
    setResult(null);
    setError("");

    const input = document.getElementById("audio-upload");

    if (input) {
      input.value = "";
    }
  };

  // ==========================================================
  // BACKEND RESPONSE MAPPING
  // ==========================================================

  const finalResult = result?.final_result || {};
  const voice = result?.voice_ai || {};
  const scam = result?.scam_analysis || {};
  const speech = result?.speech_analysis || {};
  const audio = result?.audio || {};
  const risk = result?.risk_analysis || {};

  const voiceVerdict = voice.prediction || "UNKNOWN";
  const finalVerdict = finalResult.verdict || "UNKNOWN";

  const riskScore = Number(
    finalResult.risk_score || 0
  );

  const aiProbability = Number(
    voice.ai_probability || 0
  );

  const realProbability = Number(
    voice.real_probability || 0
  );

  const scamRisk = Number(
    scam.risk_score || 0
  );

  const isSpoof =
    voiceVerdict === "AI SPOOF" ||
    voiceVerdict === "SPOOF";

  const isSuspicious =
    voiceVerdict === "SUSPICIOUS";

  const isHighRisk =
    finalVerdict === "HIGH RISK";

  let verdictClass = "real";

  if (isSpoof) {
    verdictClass = "spoof";
  } else if (isSuspicious) {
    verdictClass = "suspicious";
  }

  let verdictText = t.real;

  if (isSpoof) {
    verdictText = t.spoof;
  } else if (isSuspicious) {
    verdictText = t.suspicious;
  }

  let message = t.realMessage;

  if (isSpoof) {
    message = t.spoofMessage;
  } else if (isSuspicious) {
    message = t.suspiciousMessage;
  }

  return (
    <div className="app">

      {/* =====================================================
          HEADER
      ====================================================== */}

      <header className="header">

        <div className="logo">
          VIGILVOICE
        </div>

        <div className="header-right">

          <div className="language-selector">

            <span>
              {t.language}
            </span>

            <select
              value={language}
              onChange={(e) =>
                setLanguage(e.target.value)
              }
            >
              <option value="en">
                English
              </option>

              <option value="ta">
                தமிழ்
              </option>

              <option value="hi">
                हिन्दी
              </option>
            </select>

          </div>

          <div className="status">

            <span className="status-dot"></span>

            {t.status}

          </div>

        </div>

      </header>


      <main className="container">

        {/* ===================================================
            HERO
        ==================================================== */}

        <section className="hero">

          <p className="eyebrow">
            {t.eyebrow}
          </p>

          <h1>
            {t.title1}{" "}
            <span>
              {t.title2}
            </span>

            <br />

            {t.title3}
          </h1>

          <p className="subtitle">
            {t.subtitle}
          </p>

        </section>


        {/* ===================================================
            UPLOAD
        ==================================================== */}

        <section className="analysis-card">

          <div className="card-header">

            <div>

              <h2>
                {t.analysis}
              </h2>

              <p>
                {t.analysisDesc}
              </p>

            </div>

            <div className="engine-badge">
              {t.engine}
            </div>

          </div>


          <label
            className="upload-area"
            htmlFor="audio-upload"
          >

            <input
              id="audio-upload"
              type="file"
              accept=".wav,.mp3,.mpeg,.mpg,.flac,.ogg,.m4a,.mp4,audio/*"
              onChange={handleFileChange}
            />

            <div className="upload-icon">
              🎙
            </div>

            <h3>
              {file
                ? file.name
                : t.choose}
            </h3>

            <p>
              {file
                ? `${(
                    file.size / 1024
                  ).toFixed(1)} KB`
                : t.supported}
            </p>

          </label>


          {file && !result && (

            <button
              className="analyze-button"
              onClick={analyzeVoice}
              disabled={loading}
            >
              {loading
                ? t.analyzing
                : t.analyze}
            </button>

          )}


          {error && (

            <div className="error-box">
              ⚠ {error}
            </div>

          )}

        </section>


        {/* ===================================================
            RESULT
        ==================================================== */}

        {result && (

          <section className="result-card">

            <div className="result-title">

              <p className="eyebrow">
                {t.complete}
              </p>

              <h2>
                {t.result}
              </h2>

            </div>


            {/* =================================================
                MAIN VERDICT
            ================================================== */}

            <div
              className={`verdict ${verdictClass}`}
            >

              <div className="verdict-icon">

                {isSpoof
                  ? "⚠"
                  : isSuspicious
                  ? "!"
                  : "✓"}

              </div>

              <div>

                <span>
                  {t.verdict}
                </span>

                <strong>
                  {verdictText}
                </strong>

                <small>
                  {finalVerdict}
                </small>

              </div>

            </div>


            {/* =================================================
                RISK
            ================================================== */}

            <div className="risk-section">

              <div className="risk-header">

                <span>
                  {t.risk}
                </span>

                <strong>
                  {riskScore.toFixed(2)}%
                </strong>

              </div>

              <div className="progress">

                <div
                  className={`progress-fill ${
                    riskScore >= 70
                      ? "danger"
                      : riskScore >= 40
                      ? "warning"
                      : "safe"
                  }`}
                  style={{
                    width: `${Math.min(
                      100,
                      Math.max(
                        0,
                        riskScore
                      )
                    )}%`,
                  }}
                />

              </div>

            </div>


            {/* =================================================
                VOICE METRICS
            ================================================== */}

            <div className="metric-grid">

              <div className="metric-card">

                <span>
                  {t.aiScore}
                </span>

                <strong>
                  {aiProbability.toFixed(2)}%
                </strong>

                <p>
                  Spectra-AASIST3
                </p>

              </div>


              <div className="metric-card">

                <span>
                  {t.real}
                </span>

                <strong>
                  {realProbability.toFixed(2)}%
                </strong>

                <p>
                  Genuine probability
                </p>

              </div>


              <div className="metric-card">

                <span>
                  {t.scamRisk}
                </span>

                <strong>
                  {scamRisk.toFixed(2)}%
                </strong>

                <p>
                  Conversation analysis
                </p>

              </div>

            </div>


            {/* =================================================
                REASONS
            ================================================== */}

            {risk.reasons?.length > 0 && (

              <div className="reasons-box">

                <div className="section-label">
                  {t.reasons}
                </div>

                {risk.reasons.map(
                  (reason, index) => (

                    <div
                      className="reason"
                      key={index}
                    >

                      <span>
                        ✓
                      </span>

                      <p>
                        {reason}
                      </p>

                    </div>

                  )
                )}

              </div>

            )}


            {/* =================================================
                TRANSCRIPT
            ================================================== */}

            <div className="transcript-box">

              <div className="section-label">
                {t.transcript}
              </div>

              <p>
                {speech.transcript ||
                  "No transcript available."}
              </p>

            </div>


            {/* =================================================
                AUDIO INFO
            ================================================== */}

            <div className="audio-info">

              <div>
                <span>
                  {t.model}
                </span>

                <strong>
                  Spectra-AASIST3
                </strong>
              </div>

              <div>
                <span>
                  {t.duration}
                </span>

                <strong>
                  {Number(
                    audio.duration_seconds || 0
                  ).toFixed(2)} sec
                </strong>
              </div>

              <div>
                <span>
                  {t.segments}
                </span>

                <strong>
                  {audio.segments || 0}
                </strong>
              </div>

            </div>


            {/* =================================================
                NEW ANALYSIS
            ================================================== */}

            <button
              className="secondary-button"
              onClick={clearAnalysis}
            >
              {t.clear}
            </button>

          </section>

        )}


        {/* =====================================================
            PIPELINE
        ====================================================== */}

        <section className="pipeline">

          <div>

            <span>
              01
            </span>

            <strong>
              {t.audioInput}
            </strong>

            <p>
              {t.recording}
            </p>

          </div>

          <div className="arrow">
            →
          </div>

          <div>

            <span>
              02
            </span>

            <strong>
              {t.aasist}
            </strong>

            <p>
              {t.antiSpoof}
            </p>

          </div>

          <div className="arrow">
            →
          </div>

          <div>

            <span>
              03
            </span>

            <strong>
              {t.riskEngine}
            </strong>

            <p>
              {t.threat}
            </p>

          </div>

        </section>

      </main>


      <footer>
        {t.footer}
      </footer>

    </div>
  );
}

export default App;