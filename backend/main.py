from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.calling.webrtc.signaling import websocket_signaling
from backend.api.routes_analysis import (
    router as analysis_router
)

from backend.api.routes_speakers import (
    router as speakers_router
)


# ============================================================
# APP
# ============================================================

app = FastAPI(

    title="VIGILVOICE",

    description=(
        "AI-powered voice authenticity "
        "and scam-risk detection system"
    ),

    version="2.0.0"
)
app.websocket(
    "/api/calling/ws/{room_id}/{peer_id}"
)(websocket_signaling)

# ============================================================
# CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"]

)


# ============================================================
# ROUTES
# ============================================================

# Existing voice analysis route
app.include_router(
    analysis_router
)

# NEW: Protected speaker enrollment
# and speaker verification routes
app.include_router(
    speakers_router
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "service":
            "VIGILVOICE",

        "status":
            "online",

        "model":
            "Spectra-AASIST3",

        "speaker_verification":
            "ECAPA-TDNN",

        "database":
            "MongoDB",

        "purpose":
            "AI voice authenticity, "
            "speaker verification and "
            "scam-risk detection"

    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "voice_model":
            "Spectra-AASIST3",

        "speaker_verification": {

            "enabled":
                True,

            "model":
                "SpeechBrain ECAPA-TDNN",

            "embedding_dimension":
                192

        },

        "database": {

            "enabled":
                True,

            "type":
                "MongoDB",

            "database":
                "vigilvoice"

        },

        "voice_thresholds": {

            "real":
                "< 0.40",

            "suspicious":
                "0.40 - 0.69",

            "ai_spoof":
                ">= 0.70"

        }

    }