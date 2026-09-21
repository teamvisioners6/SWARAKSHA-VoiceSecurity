package com.vigilvoice.mobile.calling.webrtc

import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import org.json.JSONObject

class SignalingClient(
    private val serverUrl: String,
    private val roomId: String,
    private val peerId: String,
    private val listener: Listener
) {

    interface Listener {

        fun onConnected()

        fun onMessage(
            message: JSONObject
        )

        fun onDisconnected()

        fun onError(
            error: String
        )
    }

    private val client =
        OkHttpClient()

    private var webSocket:
        WebSocket? = null


    // =========================================================
    // CONNECT
    // =========================================================

    fun connect() {

        try {

            val url =
                "$serverUrl/api/calling/ws/$roomId/$peerId"

            val request =
                Request.Builder()
                    .url(url)
                    .build()

            webSocket =
                client.newWebSocket(
                    request,
                    object : WebSocketListener() {

                        override fun onOpen(
                            webSocket: WebSocket,
                            response: okhttp3.Response
                        ) {

                            println(
                                "[SWARAKSHA] Signaling connected"
                            )

                            listener.onConnected()
                        }


                        override fun onMessage(
                            webSocket: WebSocket,
                            text: String
                        ) {

                            try {

                                val message =
                                    JSONObject(text)

                                println(
                                    "[SWARAKSHA] Signaling message: $message"
                                )

                                listener.onMessage(
                                    message
                                )

                            } catch (e: Exception) {

                                listener.onError(
                                    "Invalid signaling message: ${e.message}"
                                )
                            }
                        }


                        override fun onClosed(
                            webSocket: WebSocket,
                            code: Int,
                            reason: String
                        ) {

                            println(
                                "[SWARAKSHA] Signaling disconnected"
                            )

                            listener.onDisconnected()
                        }


                        override fun onFailure(
                            webSocket: WebSocket,
                            t: Throwable,
                            response: okhttp3.Response?
                        ) {

                            println(
                                "[SWARAKSHA] Signaling error: ${t.message}"
                            )

                            listener.onError(
                                t.message
                                    ?: "Unknown WebSocket error"
                            )
                        }
                    }
                )

        } catch (e: Exception) {

            listener.onError(
                "Signaling connection failed: ${e.message}"
            )
        }
    }


    // =========================================================
    // SEND GENERIC MESSAGE
    // =========================================================

    fun send(
        message: JSONObject
    ) {

        webSocket?.send(
            message.toString()
        )
    }


    // =========================================================
    // SEND OFFER
    // =========================================================

    fun sendOffer(
        targetPeerId: String,
        sdp: String
    ) {

        val message =
            JSONObject().apply {

                put(
                    "type",
                    "offer"
                )

                put(
                    "target_peer_id",
                    targetPeerId
                )

                put(
                    "sdp",
                    sdp
                )
            }

        send(message)
    }


    // =========================================================
    // SEND ANSWER
    // =========================================================

    fun sendAnswer(
        targetPeerId: String,
        sdp: String
    ) {

        val message =
            JSONObject().apply {

                put(
                    "type",
                    "answer"
                )

                put(
                    "target_peer_id",
                    targetPeerId
                )

                put(
                    "sdp",
                    sdp
                )
            }

        send(message)
    }


    // =========================================================
    // SEND ICE CANDIDATE
    // =========================================================

    fun sendIceCandidate(
        targetPeerId: String,
        candidate: String,
        sdpMid: String?,
        sdpMLineIndex: Int
    ) {

        val message =
            JSONObject().apply {

                put(
                    "type",
                    "ice_candidate"
                )

                put(
                    "target_peer_id",
                    targetPeerId
                )

                put(
                    "candidate",
                    candidate
                )

                put(
                    "sdp_mid",
                    sdpMid
                )

                put(
                    "sdp_m_line_index",
                    sdpMLineIndex
                )
            }

        send(message)
    }


    // =========================================================
    // DISCONNECT
    // =========================================================

    fun disconnect() {

        try {

            webSocket?.close(
                1000,
                "Call ended"
            )

        } catch (_: Exception) {
        }

        webSocket = null
    }
}