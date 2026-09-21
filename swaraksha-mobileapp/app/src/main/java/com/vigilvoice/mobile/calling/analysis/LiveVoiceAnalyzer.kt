package com.vigilvoice.mobile.calling.analysis

import android.content.Context
import android.util.Log
import com.google.gson.JsonObject
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody
import retrofit2.Response
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

class LiveVoiceAnalyzer(
    private val context: Context,
    private val analyzeApi:
        suspend (MultipartBody.Part) -> Response<JsonObject>,
    private val listener: Listener
) {

    interface Listener {

        fun onAnalyzing()

        fun onResult(
            verdict: String,
            riskScore: Double,
            prediction: String
        )

        fun onError(
            error: String
        )
    }

    companion object {

        private const val TAG = "SWARAKSHA-Analyzer"

        // =====================================================
        // TARGET AUDIO FORMAT
        // =====================================================

        private const val TARGET_SAMPLE_RATE = 16000
        private const val TARGET_CHANNELS = 1
        private const val TARGET_BITS_PER_SAMPLE = 16
        private const val BYTES_PER_SAMPLE = 2

        // =====================================================
        // MODEL WINDOW
        //
        // Spectra-AASIST3 receives a 4-second window.
        // =====================================================

        private const val WINDOW_SECONDS = 4

        private const val WINDOW_BYTES =
            TARGET_SAMPLE_RATE *
                WINDOW_SECONDS *
                BYTES_PER_SAMPLE

        // =====================================================
        // ANALYSIS INTERVAL
        //
        // First result:
        //     after 4 seconds of audio
        //
        // Subsequent results:
        //     approximately every 1 second
        //
        // The model still sees the latest 4-second window.
        // =====================================================

        private const val ANALYSIS_INTERVAL_SECONDS = 1

        private const val ANALYSIS_INTERVAL_BYTES =
            TARGET_SAMPLE_RATE *
                ANALYSIS_INTERVAL_SECONDS *
                BYTES_PER_SAMPLE
    }

    // =========================================================
    // COROUTINES
    // =========================================================

    private val scope =
        CoroutineScope(
            SupervisorJob() +
                Dispatchers.IO
        )

    // =========================================================
    // STATE
    // =========================================================

    @Volatile
    private var running = false

    /*
     * TRUE while one backend inference is running.
     */
    @Volatile
    private var analysisInProgress = false

    /*
     * TRUE when fresh audio reached another analysis boundary
     * while the backend was still processing.
     *
     * We only need one pending request because the newest
     * rolling window contains the latest audio.
     */
    @Volatile
    private var analysisPending = false

    private var analysisJob: Job? = null

    // =========================================================
    // ROLLING PCM BUFFER
    //
    // Always stores the newest 4 seconds of 16-kHz mono PCM.
    // =========================================================

    private val pcmBuffer =
        ByteArrayOutputStream(
            WINDOW_BYTES
        )

    private val lock =
        Any()

    /*
     * Number of 16-kHz bytes received since the last
     * analysis boundary.
     */
    private var bytesSinceLastAnalysis = 0

    // Used only for readable diagnostic logging.
    private var lastLoggedBufferSize = 0

    // =========================================================
    // START
    // =========================================================

    fun start() {

        synchronized(lock) {

            if (running) {
                Log.d(TAG, "START ignored - analyzer already running")
                return
            }

            running = true

            pcmBuffer.reset()

            bytesSinceLastAnalysis = 0

            analysisPending = false

            analysisInProgress = false

            lastLoggedBufferSize = 0
        }

        Log.d(
            TAG,
            "ANALYZER STARTED | window=4s | interval=1s | target=16kHz mono"
        )
    }

    // =========================================================
    // RECEIVE WEBRTC PCM
    // =========================================================

    fun onPcmData(
        data: ByteArray,
        sampleRate: Int,
        channels: Int,
        bitsPerSample: Int
    ) {

        if (!running) {
            return
        }

        if (data.isEmpty()) {
            return
        }

        if (bitsPerSample != 16) {

            notifyError(
                "Unsupported WebRTC bit depth: $bitsPerSample"
            )

            return
        }

        if (sampleRate <= 0) {

            notifyError(
                "Invalid WebRTC sample rate: $sampleRate"
            )

            return
        }

        if (channels <= 0) {

            notifyError(
                "Invalid WebRTC channel count: $channels"
            )

            return
        }

        // =====================================================
        // STEP 1: DOWNMIX TO MONO
        // =====================================================

        val monoData =
            if (channels == 1) {

                data

            } else {

                downmixToMono(
                    data = data,
                    channels = channels
                )
            }

        if (monoData.isEmpty()) {
            return
        }

        // =====================================================
        // STEP 2: RESAMPLE TO 16 kHz
        // =====================================================

        val resampledData =
            if (sampleRate == TARGET_SAMPLE_RATE) {

                monoData

            } else {

                resamplePcm16(
                    pcmData = monoData,
                    inputSampleRate = sampleRate,
                    outputSampleRate = TARGET_SAMPLE_RATE
                )
            }

        if (resampledData.isEmpty()) {
            return
        }

        // =====================================================
        // STEP 3: ADD TO ROLLING BUFFER
        // =====================================================

        var snapshotToAnalyze: ByteArray? = null

        synchronized(lock) {

            if (!running) {
                return
            }

            appendToRollingBuffer(
                resampledData
            )

            bytesSinceLastAnalysis +=
                resampledData.size

            // -------------------------------------------------
            // Diagnostic buffer logging.
            //
            // We log roughly every 1 second instead of every
            // WebRTC audio callback.
            // -------------------------------------------------

            val currentBufferSize =
                pcmBuffer.size()

            if (
                currentBufferSize >=
                    lastLoggedBufferSize +
                    ANALYSIS_INTERVAL_BYTES
            ) {

                lastLoggedBufferSize =
                    currentBufferSize.coerceAtMost(
                        WINDOW_BYTES
                    )

                val bufferedSeconds =
                    currentBufferSize.toDouble() /
                        (
                            TARGET_SAMPLE_RATE *
                                BYTES_PER_SAMPLE
                        )

                Log.d(
                    TAG,
                    "BUFFER = %.2fs / 4.00s | bytes=%d"
                        .format(
                            bufferedSeconds,
                            currentBufferSize
                        )
                )
            }

            // -------------------------------------------------
            // Need a complete 4-second window.
            // -------------------------------------------------

            if (
                pcmBuffer.size() >= WINDOW_BYTES &&
                bytesSinceLastAnalysis >=
                    ANALYSIS_INTERVAL_BYTES
            ) {

                bytesSinceLastAnalysis = 0

                // -------------------------------------------------
                // Backend already processing:
                //
                // Don't start another request.
                // Mark that a newer window is waiting.
                // -------------------------------------------------

                if (analysisInProgress) {

                    analysisPending = true

                    Log.d(
                        TAG,
                        "ANALYSIS BUSY -> newest window marked PENDING"
                    )

                } else {

                    /*
                     * IMPORTANT:
                     *
                     * Reserve the analysis state HERE before
                     * leaving the synchronized block.
                     *
                     * This fixes the previous race/state bug.
                     */
                    analysisInProgress = true

                    analysisPending = false

                    snapshotToAnalyze =
                        pcmBuffer
                            .toByteArray()
                            .copyOf(
                                WINDOW_BYTES
                            )

                    Log.d(
                        TAG,
                        "4-SECOND WINDOW READY -> STARTING ANALYSIS"
                    )
                }
            }
        }

        // =====================================================
        // START ANALYSIS OUTSIDE AUDIO LOCK
        // =====================================================

        if (
            snapshotToAnalyze != null
        ) {

            launchAnalysis(
                snapshotToAnalyze
            )
        }
    }

    // =========================================================
    // ROLLING BUFFER
    // =========================================================

    private fun appendToRollingBuffer(
        data: ByteArray
    ) {

        if (data.isEmpty()) {
            return
        }

        // -----------------------------------------------------
        // Incoming chunk itself is larger than 4 seconds.
        // Keep only its newest 4 seconds.
        // -----------------------------------------------------

        if (data.size >= WINDOW_BYTES) {

            pcmBuffer.reset()

            pcmBuffer.write(
                data,
                data.size - WINDOW_BYTES,
                WINDOW_BYTES
            )

            return
        }

        // -----------------------------------------------------
        // Current buffer + new data.
        // -----------------------------------------------------

        val newSize =
            pcmBuffer.size() +
                data.size

        if (newSize <= WINDOW_BYTES) {

            pcmBuffer.write(
                data
            )

            return
        }

        // -----------------------------------------------------
        // Remove oldest samples.
        // -----------------------------------------------------

        val existing =
            pcmBuffer.toByteArray()

        val combined =
            ByteArray(
                existing.size +
                    data.size
            )

        System.arraycopy(
            existing,
            0,
            combined,
            0,
            existing.size
        )

        System.arraycopy(
            data,
            0,
            combined,
            existing.size,
            data.size
        )

        val start =
            combined.size -
                WINDOW_BYTES

        pcmBuffer.reset()

        pcmBuffer.write(
            combined,
            start,
            WINDOW_BYTES
        )
    }

    // =========================================================
    // LAUNCH BACKEND ANALYSIS
    //
    // IMPORTANT:
    //
    // analysisInProgress is already reserved before this
    // function is called.
    //
    // This function MUST NOT re-check and reset that state.
    // =========================================================

    private fun launchAnalysis(
        pcmData: ByteArray
    ) {

        if (!running) {

            synchronized(lock) {
                analysisInProgress = false
            }

            return
        }

        Log.d(
            TAG,
            "ANALYSIS START | PCM=${pcmData.size} bytes | " +
                "duration=%.2fs"
                    .format(
                        pcmData.size.toDouble() /
                            (
                                TARGET_SAMPLE_RATE *
                                    BYTES_PER_SAMPLE
                            )
                    )
        )

        analysisJob =
            scope.launch {

                notifyAnalyzing()

                try {

                    // =========================================
                    // CREATE WAV
                    // =========================================

                    val wavData =
                        createWav(
                            pcmData =
                                pcmData,

                            sampleRate =
                                TARGET_SAMPLE_RATE,

                            channels =
                                TARGET_CHANNELS,

                            bitsPerSample =
                                TARGET_BITS_PER_SAMPLE
                        )

                    Log.d(
                        TAG,
                        "WAV CREATED | size=${wavData.size} bytes"
                    )

                    // =========================================
                    // SEND TO FASTAPI
                    // =========================================

                    analyzeChunk(
                        wavData
                    )

                } catch (e: Exception) {

                    Log.e(
                        TAG,
                        "ANALYSIS EXCEPTION",
                        e
                    )

                    notifyError(
                        "Chunk analysis error: ${e.message}"
                    )

                } finally {

                    continuePendingAnalysis()
                }
            }
    }

    // =========================================================
    // CONTINUE PENDING ANALYSIS
    //
    // This is the important fix.
    //
    // We first release the current analysis state.
    // Then, if a newer window is pending, we reserve the new
    // analysis and launch it directly.
    //
    // We NEVER set analysisInProgress=true and then call a
    // function that refuses to start because it is already true.
    // =========================================================

    private fun continuePendingAnalysis() {

        var nextSnapshot: ByteArray? = null

        synchronized(lock) {

            analysisInProgress = false

            if (
                running &&
                analysisPending &&
                pcmBuffer.size() >= WINDOW_BYTES
            ) {

                analysisPending = false

                /*
                 * Reserve the next analysis BEFORE leaving
                 * the lock.
                 */
                analysisInProgress = true

                nextSnapshot =
                    pcmBuffer
                        .toByteArray()
                        .copyOf(
                            WINDOW_BYTES
                        )

                Log.d(
                    TAG,
                    "PENDING WINDOW FOUND -> STARTING NEXT ANALYSIS"
                )
            }
        }

        if (
            nextSnapshot != null
        ) {

            launchAnalysis(
                nextSnapshot
            )
        } else {

            Log.d(
                TAG,
                "ANALYSIS FINISHED | no pending window"
            )
        }
    }

    // =========================================================
    // FASTAPI
    // =========================================================

    private suspend fun analyzeChunk(
        wavData: ByteArray
    ) {

        try {

            Log.d(
                TAG,
                "POST /analyze -> sending remote_live_chunk.wav"
            )

            val requestBody =
                wavData.toRequestBody(
                    "audio/wav".toMediaType()
                )

            val filePart =
                MultipartBody.Part.createFormData(
                    "file",
                    "remote_live_chunk.wav",
                    requestBody
                )

            val response =
                analyzeApi(
                    filePart
                )

            Log.d(
                TAG,
                "BACKEND RESPONSE | HTTP ${response.code()}"
            )

            if (!response.isSuccessful) {

                Log.e(
                    TAG,
                    "BACKEND HTTP ERROR = ${response.code()}"
                )

                notifyError(
                    "Backend HTTP error: ${response.code()}"
                )

                return
            }

            val json =
                response.body()

            if (json == null) {

                Log.e(
                    TAG,
                    "BACKEND RESPONSE EMPTY"
                )

                notifyError(
                    "Empty backend response"
                )

                return
            }

            Log.d(
                TAG,
                "BACKEND RESPONSE RECEIVED"
            )

            Log.d(
                TAG,
                "BACKEND JSON = $json"
            )

            val result =
                parseResponse(
                    json
                )

            Log.d(
                TAG,
                "AI RESULT | verdict=${result.verdict} | " +
                    "risk=${result.riskScore} | " +
                    "prediction=${result.prediction}"
            )

            notifyResult(
                verdict =
                    result.verdict,

                riskScore =
                    result.riskScore,

                prediction =
                    result.prediction
            )

        } catch (e: Exception) {

            Log.e(
                TAG,
                "BACKEND CONNECTION ERROR",
                e
            )

            notifyError(
                "Backend connection error: ${e.message}"
            )
        }
    }

    // =========================================================
    // RESPONSE MODEL
    // =========================================================

    private data class AnalysisResult(
        val verdict: String,
        val riskScore: Double,
        val prediction: String
    )

    // =========================================================
    // PARSE BACKEND RESPONSE
    // =========================================================

    private fun parseResponse(
        json: JsonObject
    ): AnalysisResult {

        var verdict =
            "UNKNOWN"

        var riskScore =
            0.0

        var prediction =
            "UNKNOWN"

        // =====================================================
        // FINAL RESULT
        // =====================================================

        if (
            json.has("final_result") &&
            json
                .get("final_result")
                .isJsonObject
        ) {

            val finalResult =
                json.getAsJsonObject(
                    "final_result"
                )

            if (
                finalResult.has("verdict") &&
                !finalResult
                    .get("verdict")
                    .isJsonNull
            ) {

                verdict =
                    finalResult
                        .get("verdict")
                        .asString
            }

            if (
                finalResult.has("risk_score") &&
                !finalResult
                    .get("risk_score")
                    .isJsonNull
            ) {

                riskScore =
                    finalResult
                        .get("risk_score")
                        .asDouble

                // Backend may return 0.0 - 1.0.
                if (
                    riskScore <= 1.0
                ) {

                    riskScore *= 100.0
                }
            }
        }

        // =====================================================
        // VOICE AI
        // =====================================================

        if (
            json.has("voice_ai") &&
            json
                .get("voice_ai")
                .isJsonObject
        ) {

            val voiceAi =
                json.getAsJsonObject(
                    "voice_ai"
                )

            if (
                voiceAi.has("prediction") &&
                !voiceAi
                    .get("prediction")
                    .isJsonNull
            ) {

                prediction =
                    voiceAi
                        .get("prediction")
                        .asString
            }
        }

        return AnalysisResult(
            verdict =
                verdict,

            riskScore =
                riskScore,

            prediction =
                prediction
        )
    }

    // =========================================================
    // MULTI-CHANNEL -> MONO
    // =========================================================

    private fun downmixToMono(
        data: ByteArray,
        channels: Int
    ): ByteArray {

        if (channels <= 1) {
            return data
        }

        val bytesPerFrame =
            channels *
                BYTES_PER_SAMPLE

        if (
            data.size <
            bytesPerFrame
        ) {

            return ByteArray(0)
        }

        val frameCount =
            data.size /
                bytesPerFrame

        val output =
            ByteBuffer
                .allocate(
                    frameCount *
                        BYTES_PER_SAMPLE
                )
                .order(
                    ByteOrder.LITTLE_ENDIAN
                )

        var inputOffset =
            0

        repeat(frameCount) {

            var sum =
                0L

            repeat(channels) {

                if (
                    inputOffset + 1 <
                    data.size
                ) {

                    val sample =
                        ByteBuffer
                            .wrap(
                                data,
                                inputOffset,
                                2
                            )
                            .order(
                                ByteOrder.LITTLE_ENDIAN
                            )
                            .short
                            .toInt()

                    sum += sample
                }

                inputOffset +=
                    BYTES_PER_SAMPLE
            }

            val mono =
                (
                    sum /
                        channels
                )
                    .coerceIn(
                        -32768L,
                        32767L
                    )
                    .toShort()

            output.putShort(
                mono
            )
        }

        return output.array()
    }

    // =========================================================
    // PCM16 RESAMPLER
    // =========================================================

    private fun resamplePcm16(
        pcmData: ByteArray,
        inputSampleRate: Int,
        outputSampleRate: Int
    ): ByteArray {

        if (
            inputSampleRate <= 0 ||
            outputSampleRate <= 0
        ) {

            return ByteArray(0)
        }

        if (
            inputSampleRate ==
            outputSampleRate
        ) {

            return pcmData
        }

        val inputSampleCount =
            pcmData.size /
                BYTES_PER_SAMPLE

        if (inputSampleCount < 2) {
            return ByteArray(0)
        }

        val outputSampleCount =
            (
                inputSampleCount.toDouble() *
                    outputSampleRate.toDouble() /
                    inputSampleRate.toDouble()
            )
                .roundToInt()

        if (outputSampleCount <= 0) {
            return ByteArray(0)
        }

        val input =
            ShortArray(
                inputSampleCount
            )

        val inputBuffer =
            ByteBuffer
                .wrap(pcmData)
                .order(
                    ByteOrder.LITTLE_ENDIAN
                )

        for (i in input.indices) {

            input[i] =
                inputBuffer.short
        }

        val output =
            ByteBuffer
                .allocate(
                    outputSampleCount *
                        BYTES_PER_SAMPLE
                )
                .order(
                    ByteOrder.LITTLE_ENDIAN
                )

        val ratio =
            inputSampleRate.toDouble() /
                outputSampleRate.toDouble()

        for (
            outputIndex in
            0 until outputSampleCount
        ) {

            val sourcePosition =
                outputIndex *
                    ratio

            val index =
                sourcePosition
                    .toInt()

            val fraction =
                sourcePosition -
                    index

            val sample1 =
                input[
                    index.coerceIn(
                        0,
                        input.lastIndex
                    )
                ]
                    .toDouble()

            val sample2 =
                input[
                    (index + 1)
                        .coerceIn(
                            0,
                            input.lastIndex
                        )
                ]
                    .toDouble()

            val interpolated =
                sample1 +
                    (
                        sample2 -
                            sample1
                    ) *
                    fraction

            output.putShort(
                interpolated
                    .roundToInt()
                    .coerceIn(
                        -32768,
                        32767
                    )
                    .toShort()
            )
        }

        return output.array()
    }

    // =========================================================
    // WAV
    // =========================================================

    private fun createWav(
        pcmData: ByteArray,
        sampleRate: Int,
        channels: Int,
        bitsPerSample: Int
    ): ByteArray {

        val byteRate =
            sampleRate *
                channels *
                bitsPerSample /
                8

        val blockAlign =
            channels *
                bitsPerSample /
                8

        val output =
            ByteArrayOutputStream(
                44 +
                    pcmData.size
            )

        output.write(
            "RIFF".toByteArray()
        )

        output.write(
            intToLittleEndian(
                36 +
                    pcmData.size
            )
        )

        output.write(
            "WAVE".toByteArray()
        )

        output.write(
            "fmt ".toByteArray()
        )

        output.write(
            intToLittleEndian(
                16
            )
        )

        // PCM format
        output.write(
            shortToLittleEndian(
                1
            )
        )

        output.write(
            shortToLittleEndian(
                channels
            )
        )

        output.write(
            intToLittleEndian(
                sampleRate
            )
        )

        output.write(
            intToLittleEndian(
                byteRate
            )
        )

        output.write(
            shortToLittleEndian(
                blockAlign
            )
        )

        output.write(
            shortToLittleEndian(
                bitsPerSample
            )
        )

        output.write(
            "data".toByteArray()
        )

        output.write(
            intToLittleEndian(
                pcmData.size
            )
        )

        output.write(
            pcmData
        )

        return output.toByteArray()
    }

    // =========================================================
    // LITTLE ENDIAN HELPERS
    // =========================================================

    private fun intToLittleEndian(
        value: Int
    ): ByteArray {

        return ByteBuffer
            .allocate(4)
            .order(
                ByteOrder.LITTLE_ENDIAN
            )
            .putInt(value)
            .array()
    }

    private fun shortToLittleEndian(
        value: Int
    ): ByteArray {

        return ByteBuffer
            .allocate(2)
            .order(
                ByteOrder.LITTLE_ENDIAN
            )
            .putShort(
                value.toShort()
            )
            .array()
    }

    // =========================================================
    // CALLBACKS
    // =========================================================

    private suspend fun notifyAnalyzing() {

        withContext(
            Dispatchers.Main
        ) {

            listener.onAnalyzing()
        }
    }

    private suspend fun notifyResult(
        verdict: String,
        riskScore: Double,
        prediction: String
    ) {

        withContext(
            Dispatchers.Main
        ) {

            listener.onResult(
                verdict,
                riskScore,
                prediction
            )
        }
    }

    private fun notifyError(
        error: String
    ) {

        Log.e(
            TAG,
            "ANALYZER ERROR = $error"
        )

        scope.launch {

            withContext(
                Dispatchers.Main
            ) {

                listener.onError(
                    error
                )
            }
        }
    }

    // =========================================================
    // STOP
    // =========================================================

    fun stop() {

        synchronized(lock) {

            running = false

            pcmBuffer.reset()

            bytesSinceLastAnalysis = 0

            analysisPending = false

            analysisInProgress = false

            lastLoggedBufferSize = 0
        }

        analysisJob?.cancel()

        analysisJob = null

        Log.d(
            TAG,
            "ANALYZER STOPPED"
        )
    }

    // =========================================================
    // DESTROY
    // =========================================================

    fun destroy() {

        stop()

        scope.cancel()

        Log.d(
            TAG,
            "ANALYZER DESTROYED"
        )
    }
}