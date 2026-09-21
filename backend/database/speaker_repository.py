from datetime import datetime, timezone
from uuid import uuid4

from backend.database.mongodb import speakers_collection


class SpeakerRepository:

    @staticmethod
    def create_speaker(
        name: str,
        embedding,
        enrollment_audio_count: int = 1
    ):
        speaker_id = str(uuid4())

        document = {
            "speaker_id": speaker_id,
            "name": name,
            "embedding": embedding.tolist(),
            "embedding_dimension": len(embedding),
            "enrollment_audio_count": enrollment_audio_count,
            "created_at": datetime.now(timezone.utc),
            "status": "active"
        }

        speakers_collection.insert_one(document)

        return {
            "speaker_id": speaker_id,
            "name": name,
            "embedding_dimension": len(embedding),
            "enrollment_audio_count": enrollment_audio_count,
            "status": "active"
        }

    @staticmethod
    def get_speaker(speaker_id: str):
        return speakers_collection.find_one(
            {"speaker_id": speaker_id},
            {"_id": 0}
        )

    @staticmethod
    def list_speakers():
        return list(
            speakers_collection.find(
                {},
                {
                    "_id": 0,
                    "speaker_id": 1,
                    "name": 1,
                    "embedding_dimension": 1,
                    "enrollment_audio_count": 1,
                    "created_at": 1,
                    "status": 1
                }
            )
        )

    @staticmethod
    def delete_speaker(speaker_id: str):
        result = speakers_collection.delete_one(
            {"speaker_id": speaker_id}
        )

        return result.deleted_count > 0