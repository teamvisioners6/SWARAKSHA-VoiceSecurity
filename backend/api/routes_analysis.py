# ============================================================
# VIGILVOICE - Voice Analysis API
# ============================================================

from fastapi import APIRouter, UploadFile, File, Form
import tempfile
import os
import traceback

from backend.intelligence.voice_detector import VoiceDetector
from backend.intelligence.risk_engine import RiskEngine
from backend.intelligence.speech_to_text import SpeechToText
from backend.intelligence.scam_detector import ScamDetector
from backend.intelligence.speaker_verifier import SpeakerVerifier
from backend.database.speaker_repository import SpeakerRepository

router = APIRouter()


# ============================================================
# INITIALIZE AI COMPONENTS
# ============================================================

print("[VIGILVOICE] Initializing intelligence modules...")

voice_detector = VoiceDetector()
risk_engine = RiskEngine()
speech_to_text = SpeechToText()
scam_detector = ScamDetector()
speaker_verifier = SpeakerVerifier()

print("[VIGILVOICE] Intelligence modules ready")


# ============================================================
# HEALTH CHECK
# ============================================================

@router.get("/health")
async def analysis_health():

    return {
        "status": "healthy",
        "voice_model": "Spectra-AASIST3",
        "speaker_verification": {
            "enabled": True,
            "model": "SpeechBrain ECAPA-TDNN",
            "embedding_dimension": 192
        },
        "database": {
            "enabled": True,
            "type": "MongoDB",
            "database": "vigilvoice"
        },
        "voice_thresholds": {
            "real": "< 0.40",
            "suspicious": "0.40 - 0.69",
            "ai_spoof": ">= 0.70"
        }
    }


# ============================================================
# MAIN VOICE ANALYSIS ENDPOINT
# ============================================================

@router.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),

    # Protected speaker
    speaker_id: str = Form(None),

    # Action-aware security context
    action_type: str = Form("NORMAL_CONVERSATION"),
    transaction_value: float = Form(0.0),
    urgency: bool = Form(False)
):

    temp_path = None

    try:

        print("\n" + "=" * 70)
        print("[VIGILVOICE] NEW AUDIO ANALYSIS")
        print("=" * 70)

        print(f"[INPUT] Filename      : {file.filename}")
        print(f"[INPUT] Content-Type  : {file.content_type}")
        print(f"[INPUT] Speaker ID    : {speaker_id}")
        print(f"[INPUT] Action Type   : {action_type}")
        print(f"[INPUT] Transaction   : {transaction_value}")
        print(f"[INPUT] Urgency       : {urgency}")

        # ====================================================
        # SAVE AUDIO TEMPORARILY
        # ====================================================

        suffix = os.path.splitext(
            file.filename or ".wav"
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            audio_bytes = await file.read()

            if not audio_bytes:

                return {
                    "success": False,
                    "error": "Uploaded audio file is empty"
                }

            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        print(
            f"[AUDIO] Temporary file created: "
            f"{temp_path}"
        )

        # ====================================================
        # 1. AI VOICE / SPOOF DETECTION
        # ====================================================

        print(
            "\n[1/4] Running Spectra-AASIST3..."
        )

        voice_result = voice_detector.predict(
            temp_path
        )

        print("[VOICE DETECTION]")

        print(
            f"Prediction     : "
            f"{voice_result.get('prediction')}"
        )

        print(
            f"AI Probability : "
            f"{voice_result.get('ai_probability', 0):.2f}%"
        )

        print(
            f"Real Probability : "
            f"{voice_result.get('real_probability', 0):.2f}%"
        )

        # ====================================================
        # 2. SPEAKER VERIFICATION
        # ====================================================

        print(
            "\n[2/4] Running speaker verification..."
        )

        speaker_result = None

        if speaker_id:

            speaker = SpeakerRepository.get_speaker(
                speaker_id
            )

            if speaker:

                print(
                    f"[SPEAKER] Protected speaker: "
                    f"{speaker.get('name')}"
                )

                try:

                    speaker_result = (
                        speaker_verifier.verify_speaker(
                            temp_path,
                            speaker
                        )
                    )

                    print(
                        f"[SPEAKER] Cosine similarity: "
                        f"{speaker_result.get('cosine_similarity')}"
                    )

                    print(
                        f"[SPEAKER] Match: "
                        f"{speaker_result.get('speaker_match')}"
                    )

                except Exception as e:

                    print(
                        f"[SPEAKER] Verification failed: "
                        f"{e}"
                    )

                    speaker_result = {
                        "speaker_id": speaker_id,
                        "speaker_name":
                            speaker.get("name"),
                        "speaker_match": False,
                        "verdict":
                            "VERIFICATION_ERROR",
                        "error": str(e)
                    }

            else:

                print(
                    f"[SPEAKER] Speaker not found: "
                    f"{speaker_id}"
                )

                speaker_result = {
                    "speaker_id": speaker_id,
                    "speaker_match": False,
                    "verdict":
                        "SPEAKER_NOT_FOUND"
                }

        else:

            print(
                "[SPEAKER] No protected speaker supplied"
            )

        # ====================================================
        # 3. SPEECH-TO-TEXT + SCAM ANALYSIS
        # ====================================================

        print(
            "\n[3/4] Running speech and scam analysis..."
        )

        transcript = ""
        scam_result = None

        # ----------------------------------------------------
        # Speech-to-text
        # ----------------------------------------------------

        try:

            transcript = speech_to_text.transcribe(
                temp_path
            )

            print(
                f"[STT] Transcript: "
                f"{transcript}"
            )

        except Exception as e:

            print(
                f"[STT] Transcription failed: "
                f"{e}"
            )

            transcript = ""

        # ----------------------------------------------------
        # Scam analysis
        # ----------------------------------------------------

        try:

            scam_result = scam_detector.analyze(
                transcript
            )

            print(
                f"[SCAM] Result: "
                f"{scam_result}"
            )

        except Exception as e:

            print(
                f"[SCAM] Analysis failed: "
                f"{e}"
            )

            scam_result = {
                "scam_probability": 0.0,
                "prediction": "UNKNOWN"
            }

        # ====================================================
        # 4. ACTION-AWARE SECURITY ENGINE
        # ====================================================

        print(
            "\n[4/4] Running Action-Aware Security Gate..."
        )

        action_context = {
            "action_type": action_type,
            "transaction_value": transaction_value,
            "urgency": urgency
        }

        risk_result = risk_engine.analyze(
            voice_result,
            scam_result,
            action_context
        )

        # ====================================================
        # IMPERSONATION ANALYSIS
        # ====================================================

        impersonation_result = {
            "enabled": False,
            "status": "NOT_EVALUATED",
            "risk_level": "UNKNOWN",
            "reason": (
                "Protected speaker verification "
                "was not requested."
            )
        }

        if speaker_result:

            speaker_match = speaker_result.get(
                "speaker_match",
                False
            )

            # ------------------------------------------------
            # Get AI probability safely
            # ------------------------------------------------

            ai_probability = voice_result.get(
                "ai_probability",
                voice_result.get(
                    "spoof_probability",
                    0.0
                )
            )

            # ------------------------------------------------
            # Normalize probability
            #
            # Handles both:
            # 99.16  -> 0.9916
            # 0.9916 -> 0.9916
            # ------------------------------------------------

            if ai_probability > 1.0:

                ai_probability_normalized = (
                    ai_probability / 100.0
                )

            else:

                ai_probability_normalized = (
                    ai_probability
                )

            # ------------------------------------------------
            # CASE 1:
            # AI-generated + SAME protected speaker
            # ------------------------------------------------

            if (
                speaker_match
                and ai_probability_normalized >= 0.70
            ):

                impersonation_result = {
                    "enabled": True,
                    "status":
                        "POSSIBLE_VOICE_CLONING_IMPERSONATION",
                    "risk_level": "HIGH",
                    "reason":
                        "The voice matches the protected "
                        "speaker profile but is classified "
                        "as AI-generated."
                }

            # ------------------------------------------------
            # CASE 2:
            # Genuine + SAME protected speaker
            # ------------------------------------------------

            elif (
                speaker_match
                and ai_probability_normalized < 0.40
            ):

                impersonation_result = {
                    "enabled": True,
                    "status":
                        "PROTECTED_SPEAKER_LIKELY_GENUINE",
                    "risk_level": "LOW",
                    "reason":
                        "The voice matches the protected "
                        "speaker profile and is classified "
                        "as genuine."
                }

            # ------------------------------------------------
            # CASE 3:
            # AI-generated + DIFFERENT speaker
            # ------------------------------------------------

            elif (
                not speaker_match
                and ai_probability_normalized >= 0.70
            ):

                impersonation_result = {
                    "enabled": True,
                    "status":
                        "AI_SPOOF_DIFFERENT_SPEAKER",
                    "risk_level": "HIGH",
                    "reason":
                        "Audio is classified as AI-generated "
                        "and does not sufficiently match the "
                        "protected speaker profile."
                }

            # ------------------------------------------------
            # CASE 4:
            # Genuine + DIFFERENT speaker
            # ------------------------------------------------

            elif (
                not speaker_match
                and ai_probability_normalized < 0.40
            ):

                impersonation_result = {
                    "enabled": True,
                    "status":
                        "DIFFERENT_GENUINE_SPEAKER",
                    "risk_level": "LOW",
                    "reason":
                        "The audio appears genuine but does "
                        "not match the protected speaker "
                        "profile."
                }

            # ------------------------------------------------
            # CASE 5:
            # Suspicious / uncertain
            # ------------------------------------------------

            else:

                impersonation_result = {
                    "enabled": True,
                    "status":
                        "SUSPICIOUS_SPEAKER_AUTHENTICITY",
                    "risk_level": "MEDIUM",
                    "reason":
                        "Voice authenticity or speaker "
                        "identity requires additional "
                        "verification."
                }

        # ====================================================
        # FINAL RESULT
        # ====================================================

        final_result = {

            "verdict":
                risk_result.get(
                    "final_verdict",
                    "UNKNOWN"
                ),

            "risk_score":
                risk_result.get(
                    "risk_score",
                    0.0
                ),

            "confidence":
                risk_result.get(
                    "confidence",
                    "UNKNOWN"
                ),

            "security_action":
                risk_result.get(
                    "security_action",
                    {}
                )
        }

        # ====================================================
        # COMPLETE RESPONSE
        # ====================================================

        response = {

            "success": True,

            "filename":
                file.filename,

            # ------------------------------------------------
            # FINAL DECISION
            # ------------------------------------------------

            "final_result":
                final_result,

            # ------------------------------------------------
            # VOICE AI DETECTION
            # ------------------------------------------------

            "voice_ai":
                voice_result,

            # ------------------------------------------------
            # SPEAKER VERIFICATION
            # ------------------------------------------------

            "speaker_verification":
                speaker_result,

            # ------------------------------------------------
            # IMPERSONATION
            # ------------------------------------------------

            "impersonation_detection":
                impersonation_result,

            # ------------------------------------------------
            # TRANSCRIPT
            # ------------------------------------------------

            "transcript":
                transcript,

            # ------------------------------------------------
            # RISK ANALYSIS
            # ------------------------------------------------

            "risk_analysis": {

                "voice_verdict":
                    risk_result.get(
                        "voice_verdict"
                    ),

                "voice_risk_score":
                    risk_result.get(
                        "voice_risk_score"
                    ),

                "final_verdict":
                    risk_result.get(
                        "final_verdict"
                    ),

                "final_risk_score":
                    risk_result.get(
                        "risk_score"
                    ),

                "reasons":
                    risk_result.get(
                        "reasons",
                        []
                    )
            },

            # ------------------------------------------------
            # SECURITY GATE
            # ------------------------------------------------

            "security_action":
                risk_result.get(
                    "security_action",
                    {}
                ),

            # ------------------------------------------------
            # ACTION CONTEXT
            # ------------------------------------------------

            "action_context":
                action_context
        }

        # ====================================================
        # TERMINAL SUMMARY
        # ====================================================

        print(
            "\n" + "=" * 70
        )

        print(
            "[VIGILVOICE] ANALYSIS COMPLETE"
        )

        print(
            "=" * 70
        )

        print(
            f"Final Verdict : "
            f"{final_result['verdict']}"
        )

        print(
            f"Risk Score    : "
            f"{final_result['risk_score']}"
        )

        print(
            f"Security Gate : "
            f"{final_result['security_action'].get('decision')}"
        )

        print(
            f"Action Type   : "
            f"{action_type}"
        )

        print(
            f"Transaction   : "
            f"{transaction_value}"
        )

        print(
            f"Urgency       : "
            f"{urgency}"
        )

        print(
            "=" * 70 + "\n"
        )

        return response

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        print(
            "\n[VIGILVOICE] ANALYSIS ERROR"
        )

        print(
            str(e)
        )

        traceback.print_exc()

        return {
            "success": False,
            "error": str(e),
            "error_type":
                type(e).__name__
        }

    # ========================================================
    # CLEANUP
    # ========================================================

    finally:

        if (
            temp_path
            and os.path.exists(temp_path)
        ):

            try:

                os.remove(
                    temp_path
                )

                print(
                    f"[CLEANUP] Removed temporary file: "
                    f"{temp_path}"
                )

            except Exception as e:

                print(
                    f"[CLEANUP] Could not remove temporary "
                    f"file: {e}"
                )