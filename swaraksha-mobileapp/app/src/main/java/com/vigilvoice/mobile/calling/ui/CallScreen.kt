package com.vigilvoice.mobile.calling.ui

import com.vigilvoice.mobile.SwarakshaWordmark
import android.util.Log
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Call
import androidx.compose.material.icons.filled.CallEnd
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.MicOff
import androidx.compose.material.icons.filled.VolumeOff
import androidx.compose.material.icons.filled.VolumeUp
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.vigilvoice.mobile.RetrofitClient
import com.vigilvoice.mobile.calling.analysis.LiveVoiceAnalyzer
import com.vigilvoice.mobile.calling.webrtc.SignalingClient
import com.vigilvoice.mobile.calling.webrtc.WebRTCManager
import org.json.JSONObject
import org.webrtc.IceCandidate
import org.webrtc.PeerConnection
import org.webrtc.SessionDescription

private const val TAG = "SWARAKSHA-CallScreen"

private const val CALLER = "CALLER"
private const val RECEIVER = "RECEIVER"

@Composable
fun CallScreen(
    roomId: String,
    peerId: String,
    onBack: () -> Unit
) {
    val context = LocalContext.current

    // =========================================================
    // SIGNALING HOLDER
    // =========================================================

    val signalingClientHolder =
        remember {
            mutableStateOf<SignalingClient?>(null)
        }

    // =========================================================
    // CALL STATE
    // =========================================================

    var callRole by remember {
        mutableStateOf<String?>(null)
    }

    var remotePeerId by remember {
        mutableStateOf<String?>(null)
    }

    var callConnected by remember {
        mutableStateOf(false)
    }

    var calling by remember {
        mutableStateOf(false)
    }

    var muted by remember {
        mutableStateOf(false)
    }

    var speakerOn by remember {
        mutableStateOf(true)
    }

    var remoteAudioAvailable by remember {
        mutableStateOf(false)
    }

    var monitoring by remember {
        mutableStateOf(false)
    }

    var signalingConnected by remember {
        mutableStateOf(false)
    }

    // =========================================================
    // AI STATE
    // =========================================================

    var liveRisk by remember {
        mutableStateOf<Double?>(null)
    }

    var liveVerdict by remember {
        mutableStateOf("WAITING")
    }

    var livePrediction by remember {
        mutableStateOf("WAITING")
    }

    var analyzing by remember {
        mutableStateOf(false)
    }

    var errorMessage by remember {
        mutableStateOf<String?>(null)
    }

    /*
     * =========================================================
     * PERSISTENT AI SPOOF STATE
     *
     * Once a strong SPOOF result is detected during this call,
     * do not allow a later weak REAL window to immediately hide
     * the warning.
     * =========================================================
     */

    var aiSpoofDetected by remember {
        mutableStateOf(false)
    }

    var strongestRisk by remember {
        mutableStateOf(0.0)
    }

    // =========================================================
    // LIVE VOICE ANALYZER
    // =========================================================

    val liveVoiceAnalyzer =
        remember {

            LiveVoiceAnalyzer(

                context = context,

                analyzeApi = { filePart ->

                    RetrofitClient.api.analyzeVoice(
                        filePart
                    )
                },

                listener =
                    object :
                        LiveVoiceAnalyzer.Listener {

                        override fun onAnalyzing() {

                            analyzing = true

                            Log.d(
                                TAG,
                                "LIVE AI ANALYSIS STARTED"
                            )
                        }

                        override fun onResult(
                            verdict: String,
                            riskScore: Double,
                            prediction: String
                        ) {

                            analyzing = false

                            val normalizedPrediction =
                                prediction
                                    .trim()
                                    .uppercase()

                            val normalizedVerdict =
                                verdict
                                    .trim()
                                    .uppercase()

                            val currentRisk =
                                riskScore.coerceIn(
                                    0.0,
                                    100.0
                                )

                            /*
                             * Keep track of the strongest
                             * evidence seen during this call.
                             */

                            if (
                                currentRisk >
                                strongestRisk
                            ) {

                                strongestRisk =
                                    currentRisk
                            }

                            /*
                             * A confirmed SPOOF result takes
                             * priority over a weak later window.
                             */

                            val spoofDetectedNow =
                                normalizedPrediction == "SPOOF" ||
                                normalizedPrediction == "AI SPOOF" ||
                                normalizedVerdict == "HIGH RISK" ||
                                currentRisk >= 70.0

                            if (
                                spoofDetectedNow
                            ) {

                                aiSpoofDetected =
                                    true

                                liveRisk =
                                    maxOf(
                                        currentRisk,
                                        strongestRisk
                                    )

                                liveVerdict =
                                    "AI SPOOF"

                                livePrediction =
                                    "SPOOF"

                                Log.w(
                                    TAG,
                                    "========================================"
                                )

                                Log.w(
                                    TAG,
                                    "🚨 AI VOICE DETECTED"
                                )

                                Log.w(
                                    TAG,
                                    "Risk = $currentRisk"
                                )

                                Log.w(
                                    TAG,
                                    "Prediction = $normalizedPrediction"
                                )

                                Log.w(
                                    TAG,
                                    "Persistent SPOOF state = TRUE"
                                )

                                Log.w(
                                    TAG,
                                    "========================================"
                                )

                                return
                            }

                            /*
                             * IMPORTANT:
                             *
                             * Once SPOOF has been confirmed,
                             * do not overwrite the security
                             * warning with a later weak REAL
                             * result.
                             */

                            if (
                                !aiSpoofDetected
                            ) {

                                liveRisk =
                                    currentRisk

                                liveVerdict =
                                    when {

                                        currentRisk >= 40.0 ->
                                            "SUSPICIOUS"

                                        else ->
                                            "REAL"
                                    }

                                livePrediction =
                                    normalizedPrediction
                            }

                            Log.d(
                                TAG,
                                "LIVE RESULT | " +
                                    "risk=$currentRisk | " +
                                    "verdict=$normalizedVerdict | " +
                                    "prediction=$normalizedPrediction | " +
                                    "spoofDetected=$aiSpoofDetected"
                            )
                        }

                        override fun onError(
                            error: String
                        ) {

                            analyzing = false

                            Log.e(
                                TAG,
                                "LIVE ANALYZER ERROR: $error"
                            )

                            errorMessage =
                                error
                        }
                    }
            )
        }

    // =========================================================
    // WEBRTC MANAGER
    // =========================================================

    val webRTCManager =
        remember {

            WebRTCManager(

                context = context,

                listener =
                    object :
                        WebRTCManager.Listener {

                        override fun onIceCandidate(
                            candidate: IceCandidate
                        ) {

                            val target =
                                remotePeerId

                            if (
                                target.isNullOrBlank()
                            ) {

                                Log.w(
                                    TAG,
                                    "ICE generated but remote peer is unknown"
                                )

                                return
                            }

                            signalingClientHolder
                                .value
                                ?.sendIceCandidate(

                                    targetPeerId =
                                        target,

                                    candidate =
                                        candidate.sdp,

                                    sdpMid =
                                        candidate.sdpMid,

                                    sdpMLineIndex =
                                        candidate.sdpMLineIndex
                                )
                        }

                        override fun onConnectionStateChanged(
                            state:
                                PeerConnection.PeerConnectionState
                        ) {

                            Log.d(
                                TAG,
                                "WEBRTC CONNECTION STATE = $state"
                            )

                            when (state) {

                                PeerConnection
                                    .PeerConnectionState
                                    .CONNECTED -> {

                                    callConnected =
                                        true

                                    calling =
                                        false

                                    Log.d(
                                        TAG,
                                        "CALL CONNECTED"
                                    )

                                    if (
                                        callRole ==
                                        RECEIVER
                                    ) {

                                        monitoring =
                                            true

                                        liveVoiceAnalyzer
                                            .start()

                                        Log.d(
                                            TAG,
                                            "REMOTE AI MONITORING STARTED"
                                        )
                                    }
                                }

                                PeerConnection
                                    .PeerConnectionState
                                    .DISCONNECTED,

                                PeerConnection
                                    .PeerConnectionState
                                    .FAILED,

                                PeerConnection
                                    .PeerConnectionState
                                    .CLOSED -> {

                                    callConnected =
                                        false

                                    monitoring =
                                        false

                                    remoteAudioAvailable =
                                        false

                                    liveVoiceAnalyzer
                                        .stop()

                                    liveRisk =
                                        null

                                    liveVerdict =
                                        "WAITING"

                                    livePrediction =
                                        "WAITING"

                                    analyzing =
                                        false

                                    aiSpoofDetected =
                                        false

                                    strongestRisk =
                                        0.0
                                }

                                else -> Unit
                            }
                        }

                        override fun onIceConnectionStateChanged(
                            state:
                                PeerConnection.IceConnectionState
                        ) {

                            Log.d(
                                TAG,
                                "ICE STATE = $state"
                            )
                        }

                        override fun onOfferCreated(
                            offer:
                                SessionDescription
                        ) {

                            val target =
                                remotePeerId

                            if (
                                target.isNullOrBlank()
                            ) {

                                errorMessage =
                                    "Remote device not found"

                                Log.e(
                                    TAG,
                                    "Cannot send offer: remote peer unknown"
                                )

                                return
                            }

                            Log.d(
                                TAG,
                                "SENDING OFFER TO $target"
                            )

                            signalingClientHolder
                                .value
                                ?.sendOffer(

                                    targetPeerId =
                                        target,

                                    sdp =
                                        offer.description
                                )
                        }

                        override fun onAnswerCreated(
                            answer:
                                SessionDescription
                        ) {

                            val target =
                                remotePeerId

                            if (
                                target.isNullOrBlank()
                            ) {

                                errorMessage =
                                    "Remote device not found"

                                return
                            }

                            Log.d(
                                TAG,
                                "SENDING ANSWER TO $target"
                            )

                            signalingClientHolder
                                .value
                                ?.sendAnswer(

                                    targetPeerId =
                                        target,

                                    sdp =
                                        answer.description
                                )
                        }

                        override fun onRemoteTrackReceived() {

                            Log.d(
                                TAG,
                                "REMOTE AUDIO TRACK RECEIVED"
                            )

                            remoteAudioAvailable =
                                true
                        }

                        override fun onError(
                            error: String
                        ) {

                            Log.e(
                                TAG,
                                "WEBRTC ERROR: $error"
                            )

                            errorMessage =
                                error
                        }
                    },

                liveVoiceAnalyzer =
                    liveVoiceAnalyzer
            )
        }

    // =========================================================
    // SIGNALING CLIENT
    // =========================================================

    val signalingClient =
        remember {

            SignalingClient(

                serverUrl =
                    "ws://192.168.21.210:8000",

                roomId =
                    roomId,

                peerId =
                    peerId,

                listener =
                    object :
                        SignalingClient.Listener {

                        override fun onConnected() {

                            signalingConnected =
                                true

                            Log.d(
                                TAG,
                                "SIGNALING CONNECTED"
                            )
                        }

                        override fun onMessage(
                            message: JSONObject
                        ) {

                            Log.d(
                                TAG,
                                "SIGNALING MESSAGE = $message"
                            )

                            try {

                                val type =
                                    message.optString(
                                        "type"
                                    )

                                val senderPeerId =
                                    message
                                        .optString(
                                            "peer_id"
                                        )
                                        .ifBlank {
                                            message.optString(
                                                "from_peer_id"
                                            )
                                        }
                                        .ifBlank {
                                            message.optString(
                                                "sender_peer_id"
                                            )
                                        }
                                        .ifBlank {
                                            message.optString(
                                                "source_peer_id"
                                            )
                                        }

                                when (type) {

                                    "peer_joined" -> {

                                        if (
                                            senderPeerId.isNotBlank() &&
                                            senderPeerId != peerId
                                        ) {

                                            remotePeerId =
                                                senderPeerId

                                            Log.d(
                                                TAG,
                                                "REMOTE PEER JOINED = $senderPeerId"
                                            )
                                        }
                                    }

                                    "offer" -> {

                                        if (
                                            senderPeerId.isNotBlank()
                                        ) {

                                            remotePeerId =
                                                senderPeerId
                                        }

                                        val sdp =
                                            message.optString(
                                                "sdp"
                                            )

                                        if (
                                            sdp.isBlank()
                                        ) {

                                            errorMessage =
                                                "Received empty offer"

                                            return
                                        }

                                        callRole =
                                            RECEIVER

                                        calling =
                                            false

                                        errorMessage =
                                            null

                                        Log.d(
                                            TAG,
                                            "INCOMING OFFER -> ROLE = RECEIVER"
                                        )

                                        webRTCManager
                                            .initialize()

                                        webRTCManager
                                            .createPeerConnection()

                                        val normalizedSdp =
                                            normalizeSdp(
                                                sdp
                                            )

                                        val description =
                                            SessionDescription(
                                                SessionDescription.Type.OFFER,
                                                normalizedSdp
                                            )

                                        webRTCManager
                                            .setRemoteDescription(
                                                description
                                            )

                                        webRTCManager
                                            .createAnswer()
                                    }

                                    "answer" -> {

                                        if (
                                            senderPeerId.isNotBlank()
                                        ) {

                                            remotePeerId =
                                                senderPeerId
                                        }

                                        val sdp =
                                            message.optString(
                                                "sdp"
                                            )

                                        if (
                                            sdp.isBlank()
                                        ) {

                                            errorMessage =
                                                "Received empty answer"

                                            return
                                        }

                                        Log.d(
                                            TAG,
                                            "INCOMING ANSWER"
                                        )

                                        val normalizedSdp =
                                            normalizeSdp(
                                                sdp
                                            )

                                        val description =
                                            SessionDescription(
                                                SessionDescription.Type.ANSWER,
                                                normalizedSdp
                                            )

                                        webRTCManager
                                            .setRemoteDescription(
                                                description
                                            )
                                    }

                                    "ice_candidate" -> {

                                        if (
                                            senderPeerId.isNotBlank() &&
                                            senderPeerId != peerId
                                        ) {

                                            remotePeerId =
                                                senderPeerId
                                        }

                                        val candidate =
                                            message.optString(
                                                "candidate"
                                            )

                                        if (
                                            candidate.isBlank()
                                        ) {

                                            return
                                        }

                                        val sdpMid =
                                            if (
                                                message.has(
                                                    "sdp_mid"
                                                ) &&
                                                !message.isNull(
                                                    "sdp_mid"
                                                )
                                            ) {

                                                message
                                                    .optString(
                                                        "sdp_mid"
                                                    )
                                                    .ifBlank {
                                                        null
                                                    }

                                            } else {

                                                null
                                            }

                                        val sdpMLineIndex =
                                            message.optInt(
                                                "sdp_m_line_index",
                                                0
                                            )

                                        webRTCManager
                                            .addIceCandidate(

                                                IceCandidate(
                                                    sdpMid,
                                                    sdpMLineIndex,
                                                    candidate
                                                )
                                            )
                                    }

                                    "peer_left" -> {

                                        Log.d(
                                            TAG,
                                            "REMOTE PEER LEFT"
                                        )

                                        callConnected =
                                            false

                                        calling =
                                            false

                                        monitoring =
                                            false

                                        remoteAudioAvailable =
                                            false

                                        liveVoiceAnalyzer
                                            .stop()

                                        liveRisk =
                                            null

                                        liveVerdict =
                                            "WAITING"

                                        livePrediction =
                                            "WAITING"

                                        analyzing =
                                            false

                                        aiSpoofDetected =
                                            false

                                        strongestRisk =
                                            0.0

                                        remotePeerId =
                                            null
                                    }

                                    "call_end" -> {

                                        Log.d(
                                            TAG,
                                            "REMOTE CALL ENDED"
                                        )

                                        callConnected =
                                            false

                                        calling =
                                            false

                                        monitoring =
                                            false

                                        remoteAudioAvailable =
                                            false

                                        liveVoiceAnalyzer
                                            .stop()

                                        liveRisk =
                                            null

                                        liveVerdict =
                                            "WAITING"

                                        livePrediction =
                                            "WAITING"

                                        analyzing =
                                            false

                                        aiSpoofDetected =
                                            false

                                        strongestRisk =
                                            0.0
                                    }
                                }

                            } catch (
                                e: Exception
                            ) {

                                Log.e(
                                    TAG,
                                    "SIGNALING MESSAGE ERROR",
                                    e
                                )

                                errorMessage =
                                    e.message
                                        ?: "Signaling message error"
                            }
                        }

                        override fun onDisconnected() {

                            signalingConnected =
                                false

                            Log.d(
                                TAG,
                                "SIGNALING DISCONNECTED"
                            )
                        }

                        override fun onError(
                            error: String
                        ) {

                            Log.e(
                                TAG,
                                "SIGNALING ERROR: $error"
                            )

                            errorMessage =
                                error
                        }
                    }
            )
        }

    // =========================================================
    // CONNECT SIGNALING
    // =========================================================

    LaunchedEffect(
        signalingClient
    ) {

        signalingClientHolder.value =
            signalingClient

        signalingClient.connect()
    }

    // =========================================================
    // START OUTGOING CALL
    // =========================================================

    fun startCall() {

        if (
            calling ||
            callConnected
        ) {

            return
        }

        if (
            !signalingConnected
        ) {

            errorMessage =
                "Signaling server is not connected"

            return
        }

        callRole =
            CALLER

        calling =
            true

        errorMessage =
            null

        liveRisk =
            null

        liveVerdict =
            "WAITING"

        livePrediction =
            "WAITING"

        aiSpoofDetected =
            false

        strongestRisk =
            0.0

        Log.d(
            TAG,
            "START CALL -> ROLE = CALLER"
        )

        if (
            !remotePeerId.isNullOrBlank()
        ) {

            webRTCManager
                .initialize()

            webRTCManager
                .createPeerConnection()

            webRTCManager
                .createOffer()

        } else {

            Log.d(
                TAG,
                "REMOTE PEER NOT AVAILABLE YET"
            )

            Log.d(
                TAG,
                "WAITING FOR PEER JOIN"
            )
        }
    }

    // =========================================================
    // CALLER WAITING FOR REMOTE PEER
    // =========================================================

    LaunchedEffect(
        callRole,
        remotePeerId,
        calling
    ) {

        if (
            callRole == CALLER &&
            calling &&
            !callConnected &&
            !remotePeerId.isNullOrBlank()
        ) {

            Log.d(
                TAG,
                "REMOTE PEER NOW AVAILABLE"
            )

            webRTCManager
                .initialize()

            webRTCManager
                .createPeerConnection()

            webRTCManager
                .createOffer()
        }
    }

    // =========================================================
    // STOP EVERYTHING
    // =========================================================

    fun stopEverything() {

        Log.d(
            TAG,
            "STOPPING CALL"
        )

        liveVoiceAnalyzer
            .stop()

        webRTCManager
            .close()

        signalingClient
            .disconnect()

        callConnected =
            false

        calling =
            false

        monitoring =
            false

        remoteAudioAvailable =
            false

        liveRisk =
            null

        liveVerdict =
            "WAITING"

        livePrediction =
            "WAITING"

        analyzing =
            false

        aiSpoofDetected =
            false

        strongestRisk =
            0.0

        callRole =
            null

        remotePeerId =
            null

        onBack()
    }

    // =========================================================
    // CLEANUP
    // =========================================================

    DisposableEffect(Unit) {

        onDispose {

            liveVoiceAnalyzer
                .stop()

            webRTCManager
                .close()

            signalingClient
                .disconnect()
        }
    }

    // =========================================================
    // UI
    // =========================================================

    Box(
        modifier =
            Modifier
                .fillMaxSize()
                .background(
                    Brush.verticalGradient(
                        colors = listOf(
                            Color(0xFFF7FBFF),
                            Color(0xFFF2FAF7)
                        )
                    )
                )
    ) {

        when (callRole) {

            RECEIVER -> {

                ReceiverSecurityPanel(

                    connected =
                        callConnected,

                    calling =
                        calling,

                    monitoring =
                        monitoring,

                    remoteAudioAvailable =
                        remoteAudioAvailable,

                    liveRisk =
                        liveRisk,

                    liveVerdict =
                        liveVerdict,

                    livePrediction =
                        livePrediction,

                    analyzing =
                        analyzing,

                    muted =
                        muted,

                    speakerOn =
                        speakerOn,

                    errorMessage =
                        errorMessage,

                    onMute = {

                        muted =
                            !muted

                        webRTCManager
                            .setMicrophoneEnabled(
                                !muted
                            )
                    },

                    onSpeaker = {

                        speakerOn =
                            !speakerOn
                    },

                    onEndCall = {

                        stopEverything()
                    }
                )
            }

            else -> {

                CallerCallPanel(

                    connected =
                        callConnected,

                    calling =
                        calling,

                    signalingConnected =
                        signalingConnected,

                    muted =
                        muted,

                    speakerOn =
                        speakerOn,

                    errorMessage =
                        errorMessage,

                    onStartCall = {

                        startCall()
                    },

                    onMute = {

                        muted =
                            !muted

                        webRTCManager
                            .setMicrophoneEnabled(
                                !muted
                            )
                    },

                    onSpeaker = {

                        speakerOn =
                            !speakerOn
                    },

                    onEndCall = {

                        stopEverything()
                    }
                )
            }
        }
    }
}


/* ============================================================
 * SDP NORMALIZATION
 * ============================================================
 */

private fun normalizeSdp(
    raw: String
): String {

    var sdp =
        raw.trim()

    if (
        sdp.startsWith("\"") &&
        sdp.endsWith("\"") &&
        sdp.length >= 2
    ) {

        sdp =
            sdp.substring(
                1,
                sdp.length - 1
            )
    }

    sdp =
        sdp
            .replace(
                "\\r\\n",
                "\n"
            )
            .replace(
                "\\n",
                "\n"
            )
            .replace(
                "\\r",
                "\n"
            )
            .replace(
                "\r\n",
                "\n"
            )
            .replace(
                "\r",
                "\n"
            )

    sdp =
        sdp
            .lines()
            .joinToString("\r\n")

    if (
        !sdp.endsWith("\r\n")
    ) {

        sdp += "\r\n"
    }

    return sdp
}


/* ============================================================
 * CALLER UI
 * ============================================================
 */

@Composable
private fun CallerCallPanel(
    connected: Boolean,
    calling: Boolean,
    signalingConnected: Boolean,
    muted: Boolean,
    speakerOn: Boolean,
    errorMessage: String?,
    onStartCall: () -> Unit,
    onMute: () -> Unit,
    onSpeaker: () -> Unit,
    onEndCall: () -> Unit
) {

    Column(
        modifier =
            Modifier
                .fillMaxSize()
                .padding(
                    horizontal = 24.dp,
                    vertical = 22.dp
                ),
        horizontalAlignment =
            Alignment.CenterHorizontally
    ) {

        Spacer(
            modifier =
                Modifier.height(18.dp)
        )

        Text(
            text = "SWARAKSHA",
            color =
                Color(0xFF1459A6),
            fontSize =
                30.sp,
            fontWeight =
                FontWeight.Bold
        )

        Spacer(
            modifier =
                Modifier.height(6.dp)
        )

        Text(
            text =
                when {

                    connected ->
                        "CALL CONNECTED"

                    calling ->
                        "OUTGOING CALL"

                    else ->
                        "READY TO CALL"
                },
            color =
                Color(0xFF64748B),
            fontSize =
                13.sp,
            fontWeight =
                FontWeight.Medium
        )

        Spacer(
            modifier =
                Modifier.height(55.dp)
        )

        Box(
            modifier =
                Modifier
                    .size(142.dp)
                    .clip(CircleShape)
                    .background(
                        Color(0xFFE2F5EE)
                    ),
            contentAlignment =
                Alignment.Center
        ) {

            Box(
                modifier =
                    Modifier
                        .size(104.dp)
                        .clip(CircleShape)
                        .background(
                            Color(0xFF16A978)
                        ),
                contentAlignment =
                    Alignment.Center
            ) {

                Icon(
                    imageVector =
                        Icons.Default.Call,

                    contentDescription =
                        null,

                    tint =
                        Color.White,

                    modifier =
                        Modifier.size(48.dp)
                )
            }
        }

        Spacer(
            modifier =
                Modifier.height(28.dp)
        )

        Text(
            text =
                when {

                    connected ->
                        "Protected voice call connected"

                    calling ->
                        "Connecting to the remote device..."

                    else ->
                        "Start a protected voice call"
                },

            color =
                Color(0xFF1459A6),

            fontSize =
                20.sp,

            fontWeight =
                FontWeight.SemiBold,

            textAlign =
                TextAlign.Center
        )

        Spacer(
            modifier =
                Modifier.height(10.dp)
        )

        Text(
            text =
                "Safer conversations with real-time voice security",

            color =
                Color(0xFF64748B),

            fontSize =
                12.sp,

            textAlign =
                TextAlign.Center
        )

        Spacer(
            modifier =
                Modifier.height(28.dp)
        )

        if (!signalingConnected) {

            Text(
                text =
                    "Connecting to secure call service...",

                color =
                    Color(0xFF64748B),

                fontSize =
                    12.sp
            )

            Spacer(
                modifier =
                    Modifier.height(18.dp)
            )
        }

        if (
            !connected &&
            !calling
        ) {

            Button(
                onClick =
                    onStartCall,

                modifier =
                    Modifier
                        .fillMaxWidth()
                        .height(58.dp),

                shape =
                    RoundedCornerShape(16.dp),

                colors =
                    ButtonDefaults
                        .buttonColors(
                            containerColor =
                                Color(0xFF1764B0)
                        )
            ) {

                Icon(
                    imageVector =
                        Icons.Default.Call,

                    contentDescription =
                        null
                )

                Spacer(
                    modifier =
                        Modifier.width(10.dp)
                )

                Text(
                    text =
                        "START CALL",

                    fontSize =
                        16.sp,

                    fontWeight =
                        FontWeight.Bold
                )
            }
        }

        if (connected) {

            Spacer(
                modifier =
                    Modifier.height(28.dp)
            )

            CallControls(
                muted =
                    muted,

                speakerOn =
                    speakerOn,

                onMute =
                    onMute,

                onSpeaker =
                    onSpeaker,

                onEndCall =
                    onEndCall
            )
        }

        Spacer(
            modifier =
                Modifier.weight(1f)
        )

        if (
            !connected &&
            !calling
        ) {

            Text(
                text =
                    "SWARAKSHA",

                color =
                    Color(0xFF16A978),

                fontSize =
                    12.sp,

                fontWeight =
                    FontWeight.Bold
            )

            Spacer(
                modifier =
                    Modifier.height(5.dp)
            )

            Text(
                text =
                    "Safer Conversations",

                color =
                    Color(0xFF64748B),

                fontSize =
                    12.sp
            )
        }

        errorMessage?.let {

            Spacer(
                modifier =
                    Modifier.height(18.dp)
            )

            Text(
                text =
                    it,

                color =
                    Color(0xFFD93025),

                fontSize =
                    12.sp,

                textAlign =
                    TextAlign.Center
            )
        }
    }
}


/* ============================================================
 * RECEIVER SECURITY PANEL
 * ============================================================
 */

@Composable
private fun ReceiverSecurityPanel(
    connected: Boolean,
    calling: Boolean,
    monitoring: Boolean,
    remoteAudioAvailable: Boolean,
    liveRisk: Double?,
    liveVerdict: String,
    livePrediction: String,
    analyzing: Boolean,
    muted: Boolean,
    speakerOn: Boolean,
    errorMessage: String?,
    onMute: () -> Unit,
    onSpeaker: () -> Unit,
    onEndCall: () -> Unit
) {

    val currentRisk =
        liveRisk?.coerceIn(
            0.0,
            100.0
        )

    val verdictColor =
        when (liveVerdict) {

            "AI SPOOF" ->
                Color(0xFFD93025)

            "SUSPICIOUS" ->
                Color(0xFFE58A00)

            "REAL" ->
                Color(0xFF159A68)

            else ->
                Color(0xFF64748B)
        }

    Column(
        modifier =
            Modifier
                .fillMaxSize()
                .padding(
                    horizontal = 22.dp,
                    vertical = 20.dp
                )
    ) {

        Text(
            text =
                "SWARAKSHA",

            color =
                Color(0xFF1459A6),

            fontSize =
                28.sp,

            fontWeight =
                FontWeight.Bold
        )

        Spacer(
            modifier =
                Modifier.height(5.dp)
        )

        Text(
            text =
                "PROTECTED CALL",

            color =
                Color(0xFF16A978),

            fontSize =
                12.sp,

            fontWeight =
                FontWeight.Bold
        )

        Spacer(
            modifier =
                Modifier.height(20.dp)
        )

        SecurityStatusCard(
            connected =
                connected,

            calling =
                calling,

            monitoring =
                monitoring,

            remoteAudioAvailable =
                remoteAudioAvailable
        )

        Spacer(
            modifier =
                Modifier.height(16.dp)
        )

        Box(
            modifier =
                Modifier
                    .fillMaxWidth()
                    .clip(
                        RoundedCornerShape(20.dp)
                    )
                    .background(
                        Color.White
                    )
                    .padding(20.dp)
        ) {

            Column {

                Text(
                    text =
                        "LIVE VOICE ANALYSIS",

                    color =
                        Color(0xFF1459A6),

                    fontSize =
                        12.sp,

                    fontWeight =
                        FontWeight.Bold
                )

                Spacer(
                    modifier =
                        Modifier.height(18.dp)
                )

                Row(
                    modifier =
                        Modifier.fillMaxWidth(),

                    horizontalArrangement =
                        Arrangement.SpaceBetween,

                    verticalAlignment =
                        Alignment.CenterVertically
                ) {

                    Column {

                        Text(
                            text =
                                "AI SPOOF RISK",

                            color =
                                Color(0xFF64748B),

                            fontSize =
                                11.sp
                        )

                        Spacer(
                            modifier =
                                Modifier.height(5.dp)
                        )

                        Text(
                            text =
                                currentRisk?.let {
                                    "${it.toInt()}%"
                                }
                                    ?: "--",

                            color =
                                if (
                                    currentRisk != null
                                ) {

                                    riskColor(
                                        currentRisk
                                    )

                                } else {

                                    Color(0xFF64748B)
                                },

                            fontSize =
                                34.sp,

                            fontWeight =
                                FontWeight.Bold
                        )
                    }

                    Column(
                        horizontalAlignment =
                            Alignment.End
                    ) {

                        Text(
                            text =
                                "VERDICT",

                            color =
                                Color(0xFF64748B),

                            fontSize =
                                11.sp
                        )

                        Spacer(
                            modifier =
                                Modifier.height(5.dp)
                        )

                        Text(
                            text =
                                liveVerdict,

                            color =
                                verdictColor,

                            fontSize =
                                16.sp,

                            fontWeight =
                                FontWeight.Bold
                        )
                    }
                }

                Spacer(
                    modifier =
                        Modifier.height(18.dp)
                )

                Box(
                    modifier =
                        Modifier
                            .fillMaxWidth()
                            .height(7.dp)
                            .clip(
                                RoundedCornerShape(
                                    10.dp
                                )
                            )
                            .background(
                                Color(0xFFE7EEF5)
                            )
                ) {

                    Box(
                        modifier =
                            Modifier
                                .fillMaxWidth(
                                    (
                                        (
                                            currentRisk
                                                ?: 0.0
                                        ) / 100.0
                                    ).toFloat()
                                )
                                .height(7.dp)
                                .clip(
                                    RoundedCornerShape(
                                        10.dp
                                    )
                                )
                                .background(
                                    if (
                                        currentRisk != null
                                    ) {

                                        riskColor(
                                            currentRisk
                                        )

                                    } else {

                                        Color(0xFFB8C4D1)
                                    }
                                )
                    )
                }

                Spacer(
                    modifier =
                        Modifier.height(14.dp)
                )

                /*
                 * STRONG AI WARNING
                 *
                 * This is the visible alert that was
                 * missing during your synthetic-voice test.
                 */

                if (
                    liveVerdict == "AI SPOOF"
                ) {

                    Box(
                        modifier =
                            Modifier
                                .fillMaxWidth()
                                .clip(
                                    RoundedCornerShape(
                                        16.dp
                                    )
                                )
                                .background(
                                    Color(0xFFFFE9E7)
                                )
                                .padding(16.dp)
                    ) {

                        Column {

                            Text(
                                text =
                                    "⚠ AI VOICE DETECTED",

                                color =
                                    Color(0xFFD93025),

                                fontSize =
                                    16.sp,

                                fontWeight =
                                    FontWeight.ExtraBold
                            )

                            Spacer(
                                modifier =
                                    Modifier.height(6.dp)
                            )

                            Text(
                                text =
                                    "The caller's voice shows strong synthetic-voice indicators.",

                                color =
                                    Color(0xFF7F1D1D),

                                fontSize =
                                    12.sp,

                                lineHeight =
                                    18.sp
                            )

                            Spacer(
                                modifier =
                                    Modifier.height(8.dp)
                            )

                            Text(
                                text =
                                    "SECURITY ACTION: BLOCK / VERIFY",

                                color =
                                    Color(0xFFD93025),

                                fontSize =
                                    11.sp,

                                fontWeight =
                                    FontWeight.Bold
                            )
                        }
                    }

                    Spacer(
                        modifier =
                            Modifier.height(14.dp)
                    )
                }

                Text(
                    text =
                        when {

                            !connected ->
                                "Waiting for call connection..."

                            !remoteAudioAvailable ->
                                "REMOTE AUDIO OFFLINE"

                            !monitoring ->
                                "AI MONITORING WAITING"

                            analyzing ->
                                "AI ANALYZING LATEST AUDIO..."

                            currentRisk == null ->
                                "AI MONITORING ACTIVE — WAITING FOR RESULT"

                            liveVerdict == "AI SPOOF" ->
                                "AI MONITORING ACTIVE — THREAT DETECTED"

                            else ->
                                "AI MONITORING ACTIVE"
                        },

                    color =
                        if (
                            liveVerdict ==
                            "AI SPOOF"
                        ) {

                            Color(0xFFD93025)

                        } else {

                            Color(0xFF64748B)
                        },

                    fontSize =
                        12.sp,

                    fontWeight =
                        if (
                            liveVerdict ==
                            "AI SPOOF"
                        ) {

                            FontWeight.Bold

                        } else {

                            FontWeight.Normal
                        }
                )

                if (
                    currentRisk != null &&
                    livePrediction.isNotBlank()
                ) {

                    Spacer(
                        modifier =
                            Modifier.height(7.dp)
                    )

                    Text(
                        text =
                            "MODEL: $livePrediction",

                        color =
                            Color(0xFF94A3B8),

                        fontSize =
                            10.sp
                    )
                }
            }
        }

        Spacer(
            modifier =
                Modifier.height(14.dp)
        )

        if (
            currentRisk != null
        ) {

            Box(
                modifier =
                    Modifier
                        .fillMaxWidth()
                        .clip(
                            RoundedCornerShape(
                                16.dp
                            )
                        )
                        .background(

                            when {

                                liveVerdict ==
                                    "AI SPOOF" ->
                                    Color(0xFFFFE9E7)

                                currentRisk >= 40.0 ->
                                    Color(0xFFFFF5E3)

                                else ->
                                    Color(0xFFEAF8F1)
                            }
                        )
                        .padding(16.dp)
            ) {

                Text(

                    text =

                        when {

                            liveVerdict ==
                                "AI SPOOF" ->

                                "High-risk synthetic voice detected. Sensitive actions should be blocked or independently verified."

                            currentRisk >= 40.0 ->

                                "Suspicious voice characteristics detected. Additional verification is recommended."

                            else ->

                                "No significant synthetic-voice indicators detected in the latest analyzed window."
                        },

                    color =
                        Color(0xFF475569),

                    fontSize =
                        12.sp,

                    lineHeight =
                        18.sp
                )
            }
        }

        errorMessage?.let {

            Spacer(
                modifier =
                    Modifier.height(10.dp)
            )

            Text(
                text =
                    it,

                color =
                    Color(0xFFD93025),

                fontSize =
                    11.sp,

                textAlign =
                    TextAlign.Center,

                modifier =
                    Modifier.fillMaxWidth()
            )
        }

        Spacer(
            modifier =
                Modifier.weight(1f)
        )

        CallControls(

            muted =
                muted,

            speakerOn =
                speakerOn,

            onMute =
                onMute,

            onSpeaker =
                onSpeaker,

            onEndCall =
                onEndCall
        )

        Spacer(
            modifier =
                Modifier.height(8.dp)
        )
    }
}


/* ============================================================
 * CALL CONTROLS
 * ============================================================
 */

@Composable
private fun CallControls(
    muted: Boolean,
    speakerOn: Boolean,
    onMute: () -> Unit,
    onSpeaker: () -> Unit,
    onEndCall: () -> Unit
) {

    Row(
        modifier =
            Modifier.fillMaxWidth(),

        horizontalArrangement =
            Arrangement.Center,

        verticalAlignment =
            Alignment.CenterVertically
    ) {

        IconButton(
            onClick =
                onMute
        ) {

            Icon(

                imageVector =

                    if (muted)
                        Icons.Default.MicOff
                    else
                        Icons.Default.Mic,

                contentDescription =
                    null,

                tint =
                    Color(0xFF1459A6)
            )
        }

        Spacer(
            modifier =
                Modifier.width(26.dp)
        )

        IconButton(
            onClick =
                onSpeaker
        ) {

            Icon(

                imageVector =

                    if (speakerOn)
                        Icons.Default.VolumeUp
                    else
                        Icons.Default.VolumeOff,

                contentDescription =
                    null,

                tint =
                    Color(0xFF1459A6)
            )
        }

        Spacer(
            modifier =
                Modifier.width(26.dp)
        )

        IconButton(
            onClick =
                onEndCall
        ) {

            Box(
                modifier =
                    Modifier
                        .size(48.dp)
                        .clip(CircleShape)
                        .background(
                            Color(0xFFE85B50)
                        ),

                contentAlignment =
                    Alignment.Center
            ) {

                Icon(

                    imageVector =
                        Icons.Default.CallEnd,

                    contentDescription =
                        null,

                    tint =
                        Color.White,

                    modifier =
                        Modifier.size(25.dp)
                )
            }
        }
    }
}


/* ============================================================
 * SECURITY STATUS CARD
 * ============================================================
 */

@Composable
private fun SecurityStatusCard(
    connected: Boolean,
    calling: Boolean,
    monitoring: Boolean,
    remoteAudioAvailable: Boolean
) {

    Box(

        modifier =
            Modifier
                .fillMaxWidth()
                .clip(
                    RoundedCornerShape(
                        18.dp
                    )
                )
                .background(
                    Color(0xFFEAF7F2)
                )
                .padding(16.dp)
    ) {

        Column {

            Text(

                text =
                    "SECURITY STATUS",

                color =
                    Color(0xFF1459A6),

                fontSize =
                    12.sp,

                fontWeight =
                    FontWeight.Bold
            )

            Spacer(
                modifier =
                    Modifier.height(12.dp)
            )

            SecurityRow(

                label =
                    "CALL",

                value =

                    when {

                        connected ->
                            "CONNECTED"

                        calling ->
                            "CONNECTING"

                        else ->
                            "WAITING"
                    },

                active =
                    connected
            )

            SecurityRow(

                label =
                    "REMOTE AUDIO",

                value =

                    if (
                        remoteAudioAvailable
                    ) {

                        "AVAILABLE"

                    } else {

                        "OFFLINE"
                    },

                active =
                    remoteAudioAvailable
            )

            SecurityRow(

                label =
                    "AI MONITORING",

                value =

                    if (
                        monitoring
                    ) {

                        "ACTIVE"

                    } else {

                        "WAITING"
                    },

                active =
                    monitoring
            )
        }
    }
}


/* ============================================================
 * SECURITY ROW
 * ============================================================
 */

@Composable
private fun SecurityRow(
    label: String,
    value: String,
    active: Boolean
) {

    Row(

        modifier =
            Modifier
                .fillMaxWidth()
                .padding(
                    vertical = 6.dp
                ),

        horizontalArrangement =
            Arrangement.SpaceBetween,

        verticalAlignment =
            Alignment.CenterVertically
    ) {

        Text(

            text =
                label,

            color =
                Color(0xFF64748B),

            fontSize =
                11.sp
        )

        Text(

            text =
                value,

            color =

                if (active) {

                    Color(0xFF159A68)

                } else {

                    Color(0xFF94A3B8)
                },

            fontSize =
                11.sp,

            fontWeight =
                FontWeight.Bold
        )
    }
}


/* ============================================================
 * RISK COLOR
 * ============================================================
 */

private fun riskColor(
    risk: Double
): Color {

    return when {

        risk >= 70.0 ->
            Color(0xFFD93025)

        risk >= 40.0 ->
            Color(0xFFE58A00)

        else ->
            Color(0xFF159A68)
    }
}