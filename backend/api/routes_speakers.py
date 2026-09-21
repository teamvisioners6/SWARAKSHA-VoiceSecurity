import os
import shutil
import tempfile
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.database.speaker_repository import SpeakerRepository
from backend.database.mongodb import security_events_collection
from backend.intelligence.speaker_verifier import speaker_verifier


router = APIRouter(
    prefix="/api/speakers",
    tags=["Speaker Verification"]
)


# ============================================================
# ENROLL SPEAKER
# ============================================================

@router.post("/enroll")
async def enroll_speaker(
    name: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Enroll a protected speaker.

    The voice recording is converted into an ECAPA-TDNN
    speaker embedding. Only the embedding is stored in MongoDB.
    """

    if not name.strip():
        raise HTTPException(
            status_code=400,
            detail="Speaker name is required."
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Audio file is required."
        )

    temp_path = None

    try:
        suffix = os.path.splitext(file.filename)[1] or ".wav"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_path = temp_file.name

            shutil.copyfileobj(
                file.file,
                temp_file
            )

        print(
            f"[SpeakerEnrollment] Processing: {file.filename}"
        )

        embedding = speaker_verifier.create_embedding(
            temp_path
        )

        result = SpeakerRepository.create_speaker(
            name=name.strip(),
            embedding=embedding,
            enrollment_audio_count=1
        )

        print(
            f"[SpeakerEnrollment] Enrolled: "
            f"{name} ({result['speaker_id']})"
        )

        return {
            "success": True,
            "message": "Speaker enrolled successfully.",
            "speaker": result
        }

    except Exception as e:

        print(
            f"[SpeakerEnrollment] Error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Speaker enrollment failed: {str(e)}"
        )

    finally:

        if temp_path and os.path.exists(temp_path):

            try:
                os.remove(temp_path)
            except Exception:
                pass


# ============================================================
# LIST SPEAKERS
# ============================================================

@router.get("")
async def get_speakers():

    speakers = SpeakerRepository.list_speakers()

    return {
        "success": True,
        "count": len(speakers),
        "speakers": speakers
    }


# ============================================================
# GET SPEAKER
# ============================================================

@router.get("/{speaker_id}")
async def get_speaker(
    speaker_id: str
):

    speaker = SpeakerRepository.get_speaker(
        speaker_id
    )

    if not speaker:

        raise HTTPException(
            status_code=404,
            detail="Speaker not found."
        )

    return {
        "success": True,
        "speaker": speaker
    }


# ============================================================
# VERIFY SPEAKER
# ============================================================

@router.post("/verify")
async def verify_speaker(
    speaker_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Compare an incoming voice against a protected
    speaker profile stored in MongoDB.
    """

    # --------------------------------------------------------
    # Find protected speaker
    # --------------------------------------------------------

    speaker = SpeakerRepository.get_speaker(
        speaker_id
    )

    if not speaker:

        raise HTTPException(
            status_code=404,
            detail="Protected speaker profile not found."
        )

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Audio file is required."
        )

    temp_path = None

    try:

        # ----------------------------------------------------
        # Save incoming audio temporarily
        # ----------------------------------------------------

        suffix = os.path.splitext(file.filename)[1] or ".wav"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_path = temp_file.name

            shutil.copyfileobj(
                file.file,
                temp_file
            )

        print(
            f"[SpeakerVerification] Processing: "
            f"{file.filename}"
        )

        # ----------------------------------------------------
        # Generate incoming speaker embedding
        # ----------------------------------------------------

        incoming_embedding = (
            speaker_verifier.create_embedding(
                temp_path
            )
        )

        # ----------------------------------------------------
        # Get stored protected-speaker embedding
        # ----------------------------------------------------

        stored_embedding = speaker["embedding"]

        # ----------------------------------------------------
        # Cosine similarity
        # ----------------------------------------------------

        similarity = (
            speaker_verifier.cosine_similarity(
                incoming_embedding,
                stored_embedding
            )
        )

        similarity_percentage = (
            speaker_verifier.similarity_to_percentage(
                similarity
            )
        )

        # ----------------------------------------------------
        # Prototype verification threshold
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # This is a prototype threshold, NOT an accuracy claim.
        # We will calibrate this using genuine/impostor samples.
        #

        MATCH_THRESHOLD = 0.75

        speaker_match = similarity >= MATCH_THRESHOLD

        if speaker_match:

            speaker_verdict = "PROTECTED SPEAKER MATCH"

        else:

            speaker_verdict = "DIFFERENT SPEAKER"

        # ----------------------------------------------------
        # Store security event
        # ----------------------------------------------------

        event = {

            "speaker_id":
                speaker_id,

            "speaker_name":
                speaker["name"],

            "speaker_similarity":
                float(similarity),

            "speaker_similarity_percentage":
                float(similarity_percentage),

            "speaker_match":
                bool(speaker_match),

            "speaker_verdict":
                speaker_verdict,

            "timestamp":
                datetime.now(timezone.utc)

        }

        security_events_collection.insert_one(
            event
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {

            "success":
                True,

            "speaker_verification": {

                "speaker_id":
                    speaker_id,

                "speaker_name":
                    speaker["name"],

                "cosine_similarity":
                    round(float(similarity), 4),

                "similarity_indicator":
                    similarity_percentage,

                "match_threshold":
                    MATCH_THRESHOLD,

                "speaker_match":
                    speaker_match,

                "verdict":
                    speaker_verdict

            },

            "message":
                (
                    "Voice matches the protected "
                    "speaker profile."
                    if speaker_match
                    else
                    "Voice does not sufficiently "
                    "match the protected speaker profile."
                )

        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            f"[SpeakerVerification] Error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Speaker verification failed: {str(e)}"
        )

    finally:

        if temp_path and os.path.exists(temp_path):

            try:
                os.remove(temp_path)

            except Exception:
                pass