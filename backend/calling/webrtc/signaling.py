from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict

router = APIRouter(prefix="/api/calling", tags=["Calling"])

# ============================================================
# ACTIVE CALL ROOMS
# ============================================================

rooms: Dict[str, Dict[str, WebSocket]] = {}


# ============================================================
# SAFE SEND
# ============================================================

async def safe_send(
    websocket: WebSocket,
    message: dict
) -> bool:

    try:
        await websocket.send_json(message)
        return True

    except Exception as e:

        print(
            f"[WebRTC] Failed to send message: {e}"
        )

        return False


# ============================================================
# WEBSOCKET SIGNALING
# ============================================================

@router.websocket("/ws/{room_id}/{peer_id}")
async def websocket_signaling(
    websocket: WebSocket,
    room_id: str,
    peer_id: str
):

    await websocket.accept()

    # --------------------------------------------------------
    # Create room
    # --------------------------------------------------------

    if room_id not in rooms:
        rooms[room_id] = {}

    room = rooms[room_id]

    # --------------------------------------------------------
    # Handle duplicate peer ID
    # --------------------------------------------------------

    old_socket = room.get(peer_id)

    if old_socket is not None:

        print(
            f"[WebRTC] Replacing existing connection: "
            f"{peer_id} | Room: {room_id}"
        )

        try:
            await old_socket.close(
                code=4001,
                reason="New connection replaced old connection"
            )
        except Exception:
            pass

    # --------------------------------------------------------
    # Register peer
    # --------------------------------------------------------

    room[peer_id] = websocket

    print(
        f"[WebRTC] Peer connected: "
        f"{peer_id} | Room: {room_id}"
    )

    print(
        f"[WebRTC] Active peers: "
        f"{list(room.keys())}"
    )

    # --------------------------------------------------------
    # Notify all existing peers
    # --------------------------------------------------------

    for other_peer_id, other_socket in list(room.items()):

        if other_peer_id == peer_id:
            continue

        await safe_send(
            other_socket,
            {
                "type": "peer_joined",
                "peer_id": peer_id
            }
        )

    # --------------------------------------------------------
    # Tell new peer about existing peers
    # --------------------------------------------------------

    for other_peer_id in list(room.keys()):

        if other_peer_id == peer_id:
            continue

        await safe_send(
            websocket,
            {
                "type": "peer_joined",
                "peer_id": other_peer_id
            }
        )

    # ========================================================
    # SIGNALING LOOP
    # ========================================================

    try:

        while True:

            message = await websocket.receive_json()

            message_type = message.get("type")

            target_peer_id = message.get(
                "target_peer_id"
            )

            print(
                f"[WebRTC] "
                f"{peer_id} -> {target_peer_id} "
                f"| {message_type}"
            )

            # ------------------------------------------------
            # No target
            # ------------------------------------------------

            if not target_peer_id:

                print(
                    f"[WebRTC] Message has no target: "
                    f"{message}"
                )

                continue

            # ------------------------------------------------
            # Find target
            # ------------------------------------------------

            target_socket = room.get(
                target_peer_id
            )

            if target_socket is None:

                print(
                    f"[WebRTC] Target unavailable: "
                    f"{target_peer_id}"
                )

                await safe_send(
                    websocket,
                    {
                        "type": "peer_unavailable",
                        "peer_id": target_peer_id
                    }
                )

                continue

            # ------------------------------------------------
            # Relay
            # ------------------------------------------------

            relay_message = {
                **message,
                "from_peer_id": peer_id
            }

            success = await safe_send(
                target_socket,
                relay_message
            )

            # ------------------------------------------------
            # Remove dead target
            # ------------------------------------------------

            if not success:

                if (
                    room.get(target_peer_id)
                    is target_socket
                ):

                    room.pop(
                        target_peer_id,
                        None
                    )

    # ========================================================
    # DISCONNECT
    # ========================================================

    except WebSocketDisconnect:

        print(
            f"[WebRTC] Peer disconnected: "
            f"{peer_id} | Room: {room_id}"
        )

    except Exception as e:

        print(
            f"[WebRTC] Signaling error: "
            f"{peer_id} | {e}"
        )

    finally:

        if room_id in rooms:

            room = rooms[room_id]

            # Only remove if this is still
            # the active socket for this peer.

            if room.get(peer_id) is websocket:

                room.pop(
                    peer_id,
                    None
                )

            # ------------------------------------------------
            # Notify remaining peers
            # ------------------------------------------------

            for other_peer_id, other_socket in list(
                room.items()
            ):

                await safe_send(
                    other_socket,
                    {
                        "type": "peer_left",
                        "peer_id": peer_id
                    }
                )

            # ------------------------------------------------
            # Remove empty room
            # ------------------------------------------------

            if not room:

                rooms.pop(
                    room_id,
                    None
                )

        print(
            f"[WebRTC] Cleanup complete: "
            f"{peer_id} | Room: {room_id}"
        )