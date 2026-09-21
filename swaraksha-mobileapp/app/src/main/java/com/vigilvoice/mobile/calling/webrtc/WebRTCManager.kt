package com.vigilvoice.mobile.calling.webrtc

import android.content.Context
import android.util.Log

import com.vigilvoice.mobile.calling.analysis.LiveVoiceAnalyzer

import org.webrtc.AudioSource
import org.webrtc.AudioTrack
import org.webrtc.AudioTrackSink
import org.webrtc.IceCandidate
import org.webrtc.MediaConstraints
import org.webrtc.MediaStream
import org.webrtc.PeerConnection
import org.webrtc.PeerConnectionFactory
import org.webrtc.RtpReceiver
import org.webrtc.RtpTransceiver
import org.webrtc.SessionDescription
import org.webrtc.SoftwareVideoDecoderFactory
import org.webrtc.SoftwareVideoEncoderFactory

import java.nio.ByteBuffer


class WebRTCManager(
    private val context: Context,
    private val listener: Listener,
    private val liveVoiceAnalyzer: LiveVoiceAnalyzer? = null
) {


    // =========================================================
    // LISTENER
    // =========================================================

    interface Listener {

        fun onIceCandidate(
            candidate: IceCandidate
        )

        fun onConnectionStateChanged(
            state: PeerConnection.PeerConnectionState
        )

        fun onIceConnectionStateChanged(
            state: PeerConnection.IceConnectionState
        )

        fun onOfferCreated(
            offer: SessionDescription
        )

        fun onAnswerCreated(
            answer: SessionDescription
        )

        fun onRemoteTrackReceived()

        fun onError(
            error: String
        )
    }


    // =========================================================
    // CONSTANTS
    // =========================================================

    companion object {

        private const val TAG =
            "SWARAKSHA-WebRTC"

        private const val AUDIO_TRACK_ID =
            "swaraksha_audio_track"

        private const val AUDIO_STREAM_ID =
            "swaraksha_audio_stream"
    }


    // =========================================================
    // WEBRTC OBJECTS
    // =========================================================

    private var factory:
        PeerConnectionFactory? = null

    private var peerConnection:
        PeerConnection? = null

    private var audioSource:
        AudioSource? = null

    private var localAudioTrack:
        AudioTrack? = null

    private var remoteAudioTrack:
        AudioTrack? = null

    private var remoteAudioSink:
        AudioTrackSink? = null


    // =========================================================
    // STATE
    // =========================================================

    private var initialized =
        false

    private var remoteDescriptionSet =
        false

    private var answerCreationPending =
        false

    private var answerCreationInProgress =
        false


    // =========================================================
    // ICE QUEUE
    //
    // ICE can arrive before remote SDP.
    //
    // We hold it until the remote description has been set.
    // =========================================================

    private val pendingRemoteIceCandidates =
        mutableListOf<IceCandidate>()


    // =========================================================
    // INITIALIZE
    // =========================================================

    fun initialize() {

        if (initialized) {

            Log.d(
                TAG,
                "WebRTC already initialized"
            )

            return
        }


        try {

            Log.d(
                TAG,
                "Initializing WebRTC..."
            )


            PeerConnectionFactory.initialize(

                PeerConnectionFactory
                    .InitializationOptions
                    .builder(context)
                    .setEnableInternalTracer(false)
                    .createInitializationOptions()
            )


            val encoderFactory =
                SoftwareVideoEncoderFactory()


            val decoderFactory =
                SoftwareVideoDecoderFactory()


            factory =
                PeerConnectionFactory
                    .builder()
                    .setVideoEncoderFactory(
                        encoderFactory
                    )
                    .setVideoDecoderFactory(
                        decoderFactory
                    )
                    .createPeerConnectionFactory()


            createLocalAudioTrack()


            initialized =
                true


            Log.d(
                TAG,
                "WebRTC initialized successfully"
            )

        } catch (e: Exception) {

            Log.e(
                TAG,
                "WebRTC initialization failed",
                e
            )


            listener.onError(
                "WebRTC initialization failed: ${e.message}"
            )
        }
    }


    // =========================================================
    // LOCAL MICROPHONE
    // =========================================================

    private fun createLocalAudioTrack() {

        val pcFactory =
            factory
                ?: throw IllegalStateException(
                    "PeerConnectionFactory not initialized"
                )


        val constraints =
            MediaConstraints()


        audioSource =
            pcFactory.createAudioSource(
                constraints
            )


        localAudioTrack =
            pcFactory.createAudioTrack(
                AUDIO_TRACK_ID,
                audioSource
            )


        localAudioTrack?.setEnabled(
            true
        )


        Log.d(
            TAG,
            "Local microphone AudioTrack created"
        )
    }


    // =========================================================
    // CREATE PEER CONNECTION
    // =========================================================

    fun createPeerConnection(
        iceServers:
            List<PeerConnection.IceServer> =
                emptyList()
    ): PeerConnection? {


        if (!initialized) {

            initialize()
        }


        if (peerConnection != null) {

            Log.d(
                TAG,
                "PeerConnection already exists"
            )

            return peerConnection
        }


        val pcFactory =
            factory
                ?: run {

                    listener.onError(
                        "PeerConnectionFactory unavailable"
                    )

                    return null
                }


        try {

            val rtcConfig =
                PeerConnection.RTCConfiguration(
                    iceServers
                )


            rtcConfig.sdpSemantics =
                PeerConnection
                    .SdpSemantics
                    .UNIFIED_PLAN


            rtcConfig.continualGatheringPolicy =
                PeerConnection
                    .ContinualGatheringPolicy
                    .GATHER_CONTINUALLY


            peerConnection =
                pcFactory.createPeerConnection(

                    rtcConfig,

                    createPeerConnectionObserver()
                )


            if (peerConnection == null) {

                listener.onError(
                    "Failed to create PeerConnection"
                )

                return null
            }


            // -------------------------------------------------
            // ADD LOCAL MICROPHONE
            // -------------------------------------------------

            localAudioTrack?.let { track ->

                val sender =
                    peerConnection?.addTrack(

                        track,

                        listOf(
                            AUDIO_STREAM_ID
                        )
                    )


                if (sender != null) {

                    Log.d(
                        TAG,
                        "Local audio track added"
                    )

                } else {

                    Log.w(
                        TAG,
                        "Local audio track could not be added"
                    )
                }
            }


            Log.d(
                TAG,
                "PeerConnection created successfully"
            )


            return peerConnection

        } catch (e: Exception) {

            Log.e(
                TAG,
                "PeerConnection creation failed",
                e
            )


            listener.onError(
                "PeerConnection creation failed: ${e.message}"
            )


            return null
        }
    }


    // =========================================================
    // PEER CONNECTION OBSERVER
    // =========================================================

    private fun createPeerConnectionObserver():
        PeerConnection.Observer {

        return object :
            PeerConnection.Observer {


            override fun onSignalingChange(
                state:
                    PeerConnection.SignalingState
            ) {

                Log.d(
                    TAG,
                    "SIGNALING STATE = $state"
                )
            }


            override fun onIceConnectionChange(
                state:
                    PeerConnection.IceConnectionState
            ) {

                Log.d(
                    TAG,
                    "ICE CONNECTION STATE = $state"
                )


                listener.onIceConnectionStateChanged(
                    state
                )
            }


            override fun onIceConnectionReceivingChange(
                receiving: Boolean
            ) {

                Log.d(
                    TAG,
                    "ICE RECEIVING = $receiving"
                )
            }


            override fun onConnectionChange(
                state:
                    PeerConnection.PeerConnectionState
            ) {

                Log.d(
                    TAG,
                    "PEER CONNECTION STATE = $state"
                )


                listener.onConnectionStateChanged(
                    state
                )
            }


            override fun onIceGatheringChange(
                state:
                    PeerConnection.IceGatheringState
            ) {

                Log.d(
                    TAG,
                    "ICE GATHERING STATE = $state"
                )
            }


            override fun onIceCandidate(
                candidate:
                    IceCandidate
            ) {

                Log.d(
                    TAG,
                    "LOCAL ICE CANDIDATE GENERATED"
                )


                listener.onIceCandidate(
                    candidate
                )
            }


            override fun onIceCandidatesRemoved(
                candidates:
                    Array<IceCandidate>
            ) {

                Log.d(
                    TAG,
                    "ICE CANDIDATES REMOVED = ${candidates.size}"
                )
            }


            // =================================================
            // LEGACY REMOTE STREAM
            // =================================================

            override fun onAddStream(
                stream:
                    MediaStream
            ) {

                Log.d(
                    TAG,
                    "REMOTE MEDIA STREAM RECEIVED"
                )


                val audioTrack =
                    stream
                        .audioTracks
                        .firstOrNull()


                if (audioTrack != null) {

                    attachRemoteAudioTrack(
                        audioTrack
                    )
                }
            }


            override fun onRemoveStream(
                stream:
                    MediaStream
            ) {

                Log.d(
                    TAG,
                    "REMOTE MEDIA STREAM REMOVED"
                )


                detachRemoteAudioTrack()
            }


            // =================================================
            // DATA CHANNEL
            // =================================================

            override fun onDataChannel(
                dataChannel:
                    org.webrtc.DataChannel
            ) {

                Log.d(
                    TAG,
                    "DATA CHANNEL RECEIVED"
                )
            }


            // =================================================
            // RENEGOTIATION
            // =================================================

            override fun onRenegotiationNeeded() {

                Log.d(
                    TAG,
                    "RENEGOTIATION NEEDED"
                )
            }


            // =================================================
            // UNIFIED PLAN TRACK
            // =================================================

            override fun onTrack(
                transceiver:
                    RtpTransceiver
            ) {

                Log.d(
                    TAG,
                    "REMOTE TRANSCEIVER TRACK RECEIVED"
                )


                val track =
                    transceiver
                        .receiver
                        .track()


                if (track is AudioTrack) {

                    Log.d(
                        TAG,
                        "REMOTE AUDIO TRACK RECEIVED THROUGH onTrack"
                    )


                    attachRemoteAudioTrack(
                        track
                    )
                }
            }


            // =================================================
            // ADD TRACK
            // =================================================

            override fun onAddTrack(
                receiver:
                    RtpReceiver,

                mediaStreams:
                    Array<MediaStream>
            ) {

                Log.d(
                    TAG,
                    "REMOTE TRACK RECEIVED THROUGH onAddTrack"
                )


                val track =
                    receiver.track()


                if (track is AudioTrack) {

                    Log.d(
                        TAG,
                        "REMOTE AUDIO TRACK RECEIVED"
                    )


                    attachRemoteAudioTrack(
                        track
                    )
                }
            }
        }
    }


    // =========================================================
    // CREATE OFFER
    // =========================================================

    fun createOffer() {

        val pc =
            peerConnection
                ?: run {

                    listener.onError(
                        "PeerConnection not created"
                    )

                    return
                }


        // Reset negotiation state.

        remoteDescriptionSet =
            false

        answerCreationPending =
            false

        answerCreationInProgress =
            false


        val constraints =
            MediaConstraints().apply {

                mandatory.add(

                    MediaConstraints.KeyValuePair(
                        "OfferToReceiveAudio",
                        "true"
                    )
                )
            }


        Log.d(
            TAG,
            "CREATING OFFER"
        )


        pc.createOffer(

            object :
                SdpObserverAdapter() {


                override fun onCreateSuccess(
                    description:
                        SessionDescription
                ) {

                    Log.d(
                        TAG,
                        "OFFER CREATED"
                    )


                    pc.setLocalDescription(

                        SdpObserverAdapter(

                            onSuccess = {

                                Log.d(
                                    TAG,
                                    "LOCAL OFFER SET"
                                )
                            },

                            onFailure = { error ->

                                listener.onError(
                                    "Set local offer failed: $error"
                                )
                            }
                        ),

                        description
                    )


                    listener.onOfferCreated(
                        description
                    )
                }


                override fun onCreateFailure(
                    error:
                        String
                ) {

                    listener.onError(
                        "Offer creation failed: $error"
                    )
                }
            },

            constraints
        )
    }


    // =========================================================
    // CREATE ANSWER
    //
    // IMPORTANT FIX:
    //
    // If remote SDP has not finished being applied yet,
    // we DO NOT call pc.createAnswer().
    //
    // Instead we mark the request as pending.
    //
    // Once setRemoteDescription() succeeds,
    // createAnswer() is called automatically.
    // =========================================================

    fun createAnswer() {

        val pc =
            peerConnection
                ?: run {

                    listener.onError(
                        "PeerConnection not created"
                    )

                    return
                }


        val signalingState =
            pc.signalingState()


        Log.d(
            TAG,
            "CREATE ANSWER REQUESTED"
        )


        Log.d(
            TAG,
            "CURRENT SIGNALING STATE = $signalingState"
        )


        // -----------------------------------------------------
        // Correct state for createAnswer():
        //
        // HAVE_REMOTE_OFFER
        // HAVE_LOCAL_PRANSWER
        // -----------------------------------------------------

        if (
            signalingState !=
                PeerConnection.SignalingState
                    .HAVE_REMOTE_OFFER &&

            signalingState !=
                PeerConnection.SignalingState
                    .HAVE_LOCAL_PRANSWER
        ) {

            Log.d(
                TAG,
                "Remote offer not ready yet. Queueing answer."
            )


            answerCreationPending =
                true

            return
        }


        createAnswerNow(
            pc
        )
    }


    // =========================================================
    // ACTUAL ANSWER CREATION
    // =========================================================

    private fun createAnswerNow(
        pc:
            PeerConnection
    ) {

        if (answerCreationInProgress) {

            Log.d(
                TAG,
                "ANSWER CREATION ALREADY IN PROGRESS"
            )

            return
        }


        answerCreationInProgress =
            true

        answerCreationPending =
            false


        val constraints =
            MediaConstraints().apply {

                mandatory.add(

                    MediaConstraints.KeyValuePair(
                        "OfferToReceiveAudio",
                        "true"
                    )
                )
            }


        Log.d(
            TAG,
            "CREATING ANSWER NOW"
        )


        pc.createAnswer(

            object :
                SdpObserverAdapter() {


                override fun onCreateSuccess(
                    description:
                        SessionDescription
                ) {

                    Log.d(
                        TAG,
                        "ANSWER CREATED"
                    )


                    pc.setLocalDescription(

                        SdpObserverAdapter(

                            onSuccess = {

                                Log.d(
                                    TAG,
                                    "LOCAL ANSWER SET"
                                )


                                answerCreationInProgress =
                                    false
                            },

                            onFailure = { error ->

                                answerCreationInProgress =
                                    false

                                listener.onError(
                                    "Set local answer failed: $error"
                                )
                            }
                        ),

                        description
                    )


                    // -------------------------------------------------
                    // Notify CallScreen only after answer SDP exists.
                    // -------------------------------------------------

                    listener.onAnswerCreated(
                        description
                    )
                }


                override fun onCreateFailure(
                    error:
                        String
                ) {

                    answerCreationInProgress =
                        false


                    listener.onError(
                        "Answer creation failed: $error"
                    )
                }
            },

            constraints
        )
    }


    // =========================================================
    // SET REMOTE DESCRIPTION
    //
    // THIS IS THE MAIN FIX.
    //
    // createAnswer() is triggered ONLY after WebRTC confirms
    // the remote offer has been successfully installed.
    // =========================================================

    fun setRemoteDescription(
        description:
            SessionDescription
    ) {

        val pc =
            peerConnection
                ?: run {

                    listener.onError(
                        "PeerConnection not created"
                    )

                    return
                }


        Log.d(
            TAG,
            "SETTING REMOTE DESCRIPTION = ${description.type}"
        )


        pc.setRemoteDescription(

            SdpObserverAdapter(

                onSuccess = {

                    Log.d(
                        TAG,
                        "REMOTE DESCRIPTION SET SUCCESSFULLY"
                    )


                    if (
                        description.type ==
                            SessionDescription.Type.OFFER
                    ) {

                        remoteDescriptionSet =
                            true


                        Log.d(
                            TAG,
                            "REMOTE OFFER IS NOW READY"
                        )


                        // ---------------------------------------------
                        // FLUSH REMOTE ICE
                        // ---------------------------------------------

                        flushPendingRemoteIceCandidates(
                            pc
                        )


                        // ---------------------------------------------
                        // ANSWER WAS REQUESTED BEFORE OFFER FINISHED
                        // ---------------------------------------------

                        if (
                            answerCreationPending
                        ) {

                            Log.d(
                                TAG,
                                "PENDING ANSWER REQUEST FOUND"
                            )


                            val state =
                                pc.signalingState()


                            if (
                                state ==
                                    PeerConnection
                                        .SignalingState
                                        .HAVE_REMOTE_OFFER
                            ) {

                                createAnswerNow(
                                    pc
                                )

                            } else {

                                Log.w(
                                    TAG,
                                    "Remote offer set but signaling state is $state"
                                )
                            }
                        }
                    }


                    if (
                        description.type ==
                            SessionDescription.Type.ANSWER
                    ) {

                        remoteDescriptionSet =
                            true


                        flushPendingRemoteIceCandidates(
                            pc
                        )
                    }
                },

                onFailure = { error ->

                    Log.e(
                        TAG,
                        "REMOTE DESCRIPTION FAILED: $error"
                    )


                    listener.onError(
                        "Remote description failed: $error"
                    )
                }
            ),

            description
        )
    }


    // =========================================================
    // ADD ICE CANDIDATE
    //
    // ICE may arrive before the SDP.
    //
    // Queue it until remote description exists.
    // =========================================================

    fun addIceCandidate(
        candidate:
            IceCandidate
    ) {

        val pc =
            peerConnection
                ?: run {

                    Log.w(
                        TAG,
                        "Cannot add ICE: PeerConnection null"
                    )

                    return
                }


        if (!remoteDescriptionSet) {

            Log.d(
                TAG,
                "QUEUEING REMOTE ICE CANDIDATE"
            )


            synchronized(
                pendingRemoteIceCandidates
            ) {

                pendingRemoteIceCandidates.add(
                    candidate
                )
            }


            return
        }


        addIceCandidateNow(
            pc,
            candidate
        )
    }


    // =========================================================
    // ADD ICE NOW
    // =========================================================

    private fun addIceCandidateNow(
        pc:
            PeerConnection,

        candidate:
            IceCandidate
    ) {

        try {

            val added =
                pc.addIceCandidate(
                    candidate
                )


            if (added) {

                Log.d(
                    TAG,
                    "REMOTE ICE CANDIDATE ADDED"
                )

            } else {

                Log.w(
                    TAG,
                    "FAILED TO ADD REMOTE ICE CANDIDATE"
                )
            }

        } catch (e: Exception) {

            Log.e(
                TAG,
                "REMOTE ICE ERROR",
                e
            )
        }
    }


    // =========================================================
    // FLUSH QUEUED ICE
    // =========================================================

    private fun flushPendingRemoteIceCandidates(
        pc:
            PeerConnection
    ) {

        val queued =
            synchronized(
                pendingRemoteIceCandidates
            ) {

                val copy =
                    pendingRemoteIceCandidates.toList()

                pendingRemoteIceCandidates.clear()

                copy
            }


        if (queued.isEmpty()) {

            return
        }


        Log.d(
            TAG,
            "FLUSHING ${queued.size} QUEUED ICE CANDIDATES"
        )


        queued.forEach { candidate ->

            addIceCandidateNow(
                pc,
                candidate
            )
        }
    }


    // =========================================================
    // MICROPHONE
    // =========================================================

    fun setMicrophoneEnabled(
        enabled:
            Boolean
    ) {

        localAudioTrack
            ?.setEnabled(
                enabled
            )


        Log.d(
            TAG,
            "MICROPHONE ENABLED = $enabled"
        )
    }


    // =========================================================
    // GET PEER CONNECTION
    // =========================================================

    fun getPeerConnection():
        PeerConnection? {

        return peerConnection
    }


    // =========================================================
    // REMOTE AUDIO TRACK
    //
    // Remote caller
    //      ↓
    // WebRTC AudioTrack
    //      ↓
    // AudioTrackSink
    //      ↓
    // LiveVoiceAnalyzer
    //      ↓
    // FastAPI /analyze
    // =========================================================

    private fun attachRemoteAudioTrack(
        track:
            AudioTrack
    ) {

        if (
            remoteAudioTrack === track
        ) {

            Log.d(
                TAG,
                "REMOTE AUDIO TRACK ALREADY ATTACHED"
            )

            return
        }


        detachRemoteAudioTrack()


        remoteAudioTrack =
            track


        val sink =
            object :
                AudioTrackSink {


                override fun onData(
                    audioData:
                        ByteBuffer,

                    bitsPerSample:
                        Int,

                    sampleRate:
                        Int,

                    numberOfChannels:
                        Int,

                    numberOfFrames:
                        Int,

                    absoluteCaptureTimestampMs:
                        Long
                ) {

                    try {

                        val duplicate =
                            audioData.duplicate()


                        val bytes =
                            ByteArray(
                                duplicate.remaining()
                            )


                        duplicate.get(
                            bytes
                        )


                        Log.v(
                            TAG,
                            "REMOTE PCM: " +
                                "${bytes.size} bytes | " +
                                "${sampleRate}Hz | " +
                                "${numberOfChannels}ch | " +
                                "${bitsPerSample}bit | " +
                                "${numberOfFrames}frames"
                        )


                        // ---------------------------------------------
                        // SEND REMOTE AUDIO ONLY
                        // ---------------------------------------------

                        liveVoiceAnalyzer
                            ?.onPcmData(

                                data =
                                    bytes,

                                sampleRate =
                                    sampleRate,

                                channels =
                                    numberOfChannels,

                                bitsPerSample =
                                    bitsPerSample
                            )

                    } catch (e: Exception) {

                        Log.e(
                            TAG,
                            "REMOTE AUDIO PROCESSING ERROR",
                            e
                        )
                    }
                }
            }


        remoteAudioSink =
            sink


        try {

            track.addSink(
                sink
            )


            Log.d(
                TAG,
                "REMOTE AUDIO SINK ATTACHED"
            )


            listener.onRemoteTrackReceived()

        } catch (e: Exception) {

            Log.e(
                TAG,
                "REMOTE AUDIO SINK ERROR",
                e
            )


            remoteAudioTrack =
                null

            remoteAudioSink =
                null


            listener.onError(
                "Remote audio sink error: ${e.message}"
            )
        }
    }


    // =========================================================
    // DETACH REMOTE AUDIO
    // =========================================================

    private fun detachRemoteAudioTrack() {

        val track =
            remoteAudioTrack

        val sink =
            remoteAudioSink


        if (
            track != null &&
            sink != null
        ) {

            try {

                track.removeSink(
                    sink
                )


                Log.d(
                    TAG,
                    "REMOTE AUDIO SINK DETACHED"
                )

            } catch (e: Exception) {

                Log.w(
                    TAG,
                    "REMOTE AUDIO SINK REMOVAL FAILED",
                    e
                )
            }
        }


        remoteAudioTrack =
            null

        remoteAudioSink =
            null
    }


    // =========================================================
    // CLOSE
    // =========================================================

    fun close() {

        Log.d(
            TAG,
            "CLOSING WEBRTC"
        )


        detachRemoteAudioTrack()


        try {

            peerConnection
                ?.close()

        } catch (e: Exception) {

            Log.w(
                TAG,
                "PeerConnection close failed",
                e
            )
        }


        peerConnection =
            null


        try {

            localAudioTrack
                ?.setEnabled(
                    false
                )

        } catch (_: Exception) {
        }


        try {

            audioSource
                ?.dispose()

        } catch (e: Exception) {

            Log.w(
                TAG,
                "AudioSource dispose failed",
                e
            )
        }


        audioSource =
            null

        localAudioTrack =
            null


        synchronized(
            pendingRemoteIceCandidates
        ) {

            pendingRemoteIceCandidates.clear()
        }


        remoteDescriptionSet =
            false

        answerCreationPending =
            false

        answerCreationInProgress =
            false

        initialized =
            false


        Log.d(
            TAG,
            "WEBRTC CLOSED"
        )
    }


    // =========================================================
    // SDP OBSERVER ADAPTER
    // =========================================================

    private open class SdpObserverAdapter(
        private val onSuccess:
            (() -> Unit)? = null,

        private val onFailure:
            ((String) -> Unit)? = null

    ) : org.webrtc.SdpObserver {


        override fun onCreateSuccess(
            description:
                SessionDescription
        ) {
        }


        override fun onSetSuccess() {

            onSuccess?.invoke()
        }


        override fun onCreateFailure(
            error:
                String
        ) {

            onFailure?.invoke(
                error
            )
        }


        override fun onSetFailure(
            error:
                String
        ) {

            onFailure?.invoke(
                error
            )
        }
    }
}