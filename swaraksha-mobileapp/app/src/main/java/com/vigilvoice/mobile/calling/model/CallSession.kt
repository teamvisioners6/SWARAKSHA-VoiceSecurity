package com.vigilvoice.mobile.calling.model

data class CallSession(
    val roomId: String = "",
    val localPeerId: String = "",
    val remotePeerId: String = "",
    val status: CallStatus = CallStatus.IDLE,
    val isMuted: Boolean = false,
    val isSpeakerOn: Boolean = true
)

enum class CallStatus {
    IDLE,
    CONNECTING,
    RINGING,
    CONNECTED,
    ENDED,
    ERROR
}