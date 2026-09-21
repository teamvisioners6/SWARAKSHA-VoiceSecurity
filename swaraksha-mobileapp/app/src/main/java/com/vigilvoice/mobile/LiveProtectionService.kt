package com.vigilvoice.mobile

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.graphics.PixelFormat
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.IBinder
import android.provider.Settings
import android.view.Gravity
import android.view.WindowManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView

import androidx.core.app.NotificationCompat

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody

import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale


class LiveProtectionService : Service() {

    companion object {

        private const val CHANNEL_ID =
            "vigilvoice_live_protection"

        private const val NOTIFICATION_ID =
            9001

        private const val SAMPLE_RATE =
            16000

        private const val CHANNELS =
            AudioFormat.CHANNEL_IN_MONO

        private const val ENCODING =
            AudioFormat.ENCODING_PCM_16BIT

        private const val CHUNK_SECONDS =
            6

        private const val CHUNK_SAMPLES =
            SAMPLE_RATE * CHUNK_SECONDS
    }


    // =========================================================
    // COROUTINE
    // =========================================================

    private val serviceScope =
        CoroutineScope(
            Dispatchers.IO +
                    SupervisorJob()
        )

    private var processingJob: Job? =
        null


    // =========================================================
    // AUDIO
    // =========================================================

    private var audioRecord:
            AudioRecord? = null


    // =========================================================
    // OVERLAY
    // =========================================================

    private var windowManager:
            WindowManager? = null

    private var overlayView:
            LinearLayout? = null

    private var aiText:
            TextView? = null

    private var scamText:
            TextView? = null

    private var verdictText:
            TextView? = null

    private var statusText:
            TextView? = null


    // =========================================================
    // STOP FLAG
    // =========================================================

    @Volatile
    private var isStopping =
        false


    // =========================================================
    // CREATE
    // =========================================================

    override fun onCreate() {

        super.onCreate()

        isStopping = false

        createNotificationChannel()

        startForeground(
            NOTIFICATION_ID,
            createNotification()
        )

        if (
            Build.VERSION.SDK_INT >=
            Build.VERSION_CODES.M &&
            !Settings.canDrawOverlays(this)
        ) {

            stopServiceCompletely()

            return
        }

        showOverlay()

        startLiveMonitoring()
    }


    // =========================================================
    // START COMMAND
    // =========================================================

    override fun onStartCommand(
        intent: Intent?,
        flags: Int,
        startId: Int
    ): Int {

        return START_NOT_STICKY
    }


    // =========================================================
    // START MONITORING
    // =========================================================

    private fun startLiveMonitoring() {

        if (
            processingJob?.isActive == true
        ) {
            return
        }

        isStopping = false

        processingJob =
            serviceScope.launch {

                try {

                    val minimumBuffer =
                        AudioRecord.getMinBufferSize(
                            SAMPLE_RATE,
                            CHANNELS,
                            ENCODING
                        )

                    val bufferSize =
                        maxOf(
                            minimumBuffer,
                            CHUNK_SAMPLES * 2
                        )


                    audioRecord =
                        AudioRecord(
                            MediaRecorder.AudioSource.MIC,
                            SAMPLE_RATE,
                            CHANNELS,
                            ENCODING,
                            bufferSize
                        )


                    if (
                        audioRecord?.state !=
                        AudioRecord.STATE_INITIALIZED
                    ) {

                        updateStatus(
                            "MICROPHONE ERROR"
                        )

                        stopServiceCompletely()

                        return@launch
                    }


                    audioRecord?.startRecording()

                    updateStatus(
                        "LISTENING..."
                    )


                    val samples =
                        ShortArray(
                            CHUNK_SAMPLES
                        )


                    while (
                        isActive &&
                        !isStopping
                    ) {

                        var totalRead =
                            0


                        updateStatus(
                            "LISTENING..."
                        )


                        while (
                            totalRead <
                            CHUNK_SAMPLES &&
                            isActive &&
                            !isStopping
                        ) {

                            val read =
                                audioRecord?.read(
                                    samples,
                                    totalRead,
                                    CHUNK_SAMPLES -
                                            totalRead
                                ) ?: -1


                            if (
                                read <= 0
                            ) {
                                break
                            }


                            totalRead += read
                        }


                        // -------------------------------------
                        // STOP CHECK
                        // -------------------------------------

                        if (
                            isStopping ||
                            !isActive
                        ) {
                            break
                        }


                        if (
                            totalRead <
                            SAMPLE_RATE
                        ) {

                            continue
                        }


                        updateStatus(
                            "ANALYZING VOICE..."
                        )


                        val audioFile =
                            createWavFile(
                                samples,
                                totalRead
                            )


                        try {

                            if (
                                !isStopping
                            ) {

                                analyzeChunk(
                                    audioFile
                                )
                            }

                        } catch (
                            e: Exception
                        ) {

                            if (
                                !isStopping
                            ) {

                                updateStatus(
                                    "ANALYSIS ERROR"
                                )
                            }

                        } finally {

                            try {
                                audioFile.delete()
                            } catch (
                                _: Exception
                            ) {
                            }
                        }


                        if (
                            isStopping
                        ) {
                            break
                        }


                        delay(300)
                    }


                } catch (
                    e: Exception
                ) {

                    if (
                        !isStopping
                    ) {

                        updateStatus(
                            "LIVE PROTECTION ERROR"
                        )
                    }

                } finally {

                    releaseAudio()
                }
            }
    }


    // =========================================================
    // ANALYZE CHUNK
    // =========================================================

    private suspend fun analyzeChunk(
        audioFile: File
    ) {

        if (isStopping) {
            return
        }


        val requestBody =
            audioFile.asRequestBody(
                "audio/wav".toMediaType()
            )


        val multipart =
            MultipartBody.Part.createFormData(
                "file",
                audioFile.name,
                requestBody
            )


        val response =
            RetrofitClient
                .api
                .analyzeVoice(
                    multipart
                )


        if (isStopping) {
            return
        }


        if (
            !response.isSuccessful
        ) {

            updateStatus(
                "SERVER ERROR ${response.code()}"
            )

            return
        }


        val body =
            response.body()


        if (body == null) {

            updateStatus(
                "EMPTY SERVER RESPONSE"
            )

            return
        }


        // =====================================================
        // VOICE AI
        // =====================================================

        val voiceAI =
            body.getAsJsonObject(
                "voice_ai"
            )


        val aiProbability =
            voiceAI
                ?.get("ai_probability")
                ?.asDouble
                ?: 0.0


        // =====================================================
        // SCAM
        // =====================================================

        val scamAnalysis =
            body.getAsJsonObject(
                "scam_analysis"
            )


        val scamRisk =
            scamAnalysis
                ?.get("risk_score")
                ?.asDouble
                ?: 0.0


        // =====================================================
        // FINAL RESULT
        // =====================================================

        val finalResult =
            body.getAsJsonObject(
                "final_result"
            )


        val verdict =
            finalResult
                ?.get("verdict")
                ?.asString
                ?: "UNKNOWN"


        if (!isStopping) {

            updateOverlay(
                aiProbability,
                scamRisk,
                verdict
            )
        }
    }


    // =========================================================
    // CREATE WAV
    // =========================================================

    private fun createWavFile(
        samples: ShortArray,
        sampleCount: Int
    ): File {

        val file =
            File(
                cacheDir,
                "vigilvoice_live.wav"
            )


        val dataSize =
            sampleCount * 2


        val fileSize =
            36 + dataSize


        FileOutputStream(
            file
        ).use { output ->


            // RIFF

            output.write(
                byteArrayOf(
                    'R'.code.toByte(),
                    'I'.code.toByte(),
                    'F'.code.toByte(),
                    'F'.code.toByte()
                )
            )


            writeIntLE(
                output,
                fileSize
            )


            // WAVE

            output.write(
                byteArrayOf(
                    'W'.code.toByte(),
                    'A'.code.toByte(),
                    'V'.code.toByte(),
                    'E'.code.toByte()
                )
            )


            // fmt

            output.write(
                byteArrayOf(
                    'f'.code.toByte(),
                    'm'.code.toByte(),
                    't'.code.toByte(),
                    ' '.code.toByte()
                )
            )


            writeIntLE(
                output,
                16
            )


            writeShortLE(
                output,
                1
            )


            writeShortLE(
                output,
                1
            )


            writeIntLE(
                output,
                SAMPLE_RATE
            )


            writeIntLE(
                output,
                SAMPLE_RATE * 2
            )


            writeShortLE(
                output,
                2
            )


            writeShortLE(
                output,
                16
            )


            // data

            output.write(
                byteArrayOf(
                    'd'.code.toByte(),
                    'a'.code.toByte(),
                    't'.code.toByte(),
                    'a'.code.toByte()
                )
            )


            writeIntLE(
                output,
                dataSize
            )


            for (
            i in 0 until sampleCount
            ) {

                writeShortLE(
                    output,
                    samples[i].toInt()
                )
            }
        }


        return file
    }


    // =========================================================
    // WRITE INT
    // =========================================================

    private fun writeIntLE(
        output: FileOutputStream,
        value: Int
    ) {

        val buffer =
            ByteBuffer
                .allocate(4)
                .order(
                    ByteOrder.LITTLE_ENDIAN
                )


        buffer.putInt(
            value
        )


        output.write(
            buffer.array()
        )
    }


    // =========================================================
    // WRITE SHORT
    // =========================================================

    private fun writeShortLE(
        output: FileOutputStream,
        value: Int
    ) {

        val buffer =
            ByteBuffer
                .allocate(2)
                .order(
                    ByteOrder.LITTLE_ENDIAN
                )


        buffer.putShort(
            value.toShort()
        )


        output.write(
            buffer.array()
        )
    }


    // =========================================================
    // SHOW OVERLAY
    // =========================================================

    private fun showOverlay() {

        windowManager =
            getSystemService(
                Context.WINDOW_SERVICE
            ) as WindowManager


        overlayView =
            LinearLayout(this).apply {

                orientation =
                    LinearLayout.VERTICAL

                setPadding(
                    28,
                    22,
                    28,
                    22
                )

                setBackgroundColor(
                    Color.rgb(
                        15,
                        23,
                        42
                    )
                )
            }


        // -----------------------------------------------------
        // TITLE
        // -----------------------------------------------------

        val title =
            TextView(this).apply {

                text =
                    "🛡 VIGILVOICE  ● LIVE"

                textSize =
                    16f

                setTextColor(
                    Color.rgb(
                        0,
                        230,
                        118
                    )
                )

                setTypeface(
                    null,
                    android.graphics.Typeface.BOLD
                )
            }


        // -----------------------------------------------------
        // STATUS
        // -----------------------------------------------------

        statusText =
            TextView(this).apply {

                text =
                    "Starting protection..."

                textSize =
                    11f

                setTextColor(
                    Color.LTGRAY
                )
            }


        // -----------------------------------------------------
        // AI
        // -----------------------------------------------------

        aiText =
            TextView(this).apply {

                text =
                    "AI VOICE     --"

                textSize =
                    15f

                setTextColor(
                    Color.WHITE
                )

                setPadding(
                    0,
                    15,
                    0,
                    4
                )
            }


        // -----------------------------------------------------
        // SCAM
        // -----------------------------------------------------

        scamText =
            TextView(this).apply {

                text =
                    "SCAM RISK    --"

                textSize =
                    15f

                setTextColor(
                    Color.WHITE
                )

                setPadding(
                    0,
                    4,
                    0,
                    4
                )
            }


        // -----------------------------------------------------
        // VERDICT
        // -----------------------------------------------------

        verdictText =
            TextView(this).apply {

                text =
                    "● PROTECTING"

                textSize =
                    17f

                setTypeface(
                    null,
                    android.graphics.Typeface.BOLD
                )

                setTextColor(
                    Color.rgb(
                        0,
                        230,
                        118
                    )
                )

                setPadding(
                    0,
                    8,
                    0,
                    12
                )
            }


        // =====================================================
        // STOP BUTTON
        // =====================================================

        val stopButton =
            Button(this).apply {

                text =
                    "STOP PROTECTION"

                isAllCaps =
                    false

                setOnClickListener {

                    // IMPORTANT:
                    // Stop everything immediately.

                    stopServiceCompletely()
                }
            }


        // =====================================================
        // ADD VIEWS
        // =====================================================

        overlayView?.addView(
            title
        )

        overlayView?.addView(
            statusText
        )

        overlayView?.addView(
            aiText
        )

        overlayView?.addView(
            scamText
        )

        overlayView?.addView(
            verdictText
        )

        overlayView?.addView(
            stopButton
        )


        // =====================================================
        // WINDOW TYPE
        // =====================================================

        val layoutType =

            if (
                Build.VERSION.SDK_INT >=
                Build.VERSION_CODES.O
            ) {

                android.view.WindowManager
                    .LayoutParams
                    .TYPE_APPLICATION_OVERLAY

            } else {

                @Suppress("DEPRECATION")

                android.view.WindowManager
                    .LayoutParams
                    .TYPE_PHONE
            }


        val params =
            WindowManager.LayoutParams(

                650,

                WindowManager
                    .LayoutParams
                    .WRAP_CONTENT,

                layoutType,

                WindowManager
                    .LayoutParams
                    .FLAG_NOT_FOCUSABLE,

                PixelFormat.TRANSLUCENT
            )


        params.gravity =
            Gravity.TOP or
                    Gravity.CENTER_HORIZONTAL


        params.y =
            100


        try {

            windowManager?.addView(
                overlayView,
                params
            )

        } catch (
            _: Exception
        ) {
        }
    }


    // =========================================================
    // UPDATE OVERLAY
    // =========================================================

    private fun updateOverlay(
        aiProbability: Double,
        scamRisk: Double,
        verdict: String
    ) {

        if (isStopping) {
            return
        }


        serviceScope.launch(
            Dispatchers.Main
        ) {

            if (isStopping) {
                return@launch
            }


            aiText?.text =
                String.format(
                    Locale.US,
                    "AI VOICE     %.1f%%",
                    aiProbability
                )


            scamText?.text =
                String.format(
                    Locale.US,
                    "SCAM RISK    %.1f%%",
                    scamRisk
                )


            verdictText?.text =
                when {

                    verdict.contains(
                        "HIGH RISK",
                        ignoreCase = true
                    ) -> {

                        "🔴 HIGH RISK"
                    }


                    verdict.contains(
                        "SPOOF",
                        ignoreCase = true
                    ) -> {

                        "🔴 AI VOICE DETECTED"
                    }


                    verdict.contains(
                        "SUSPICIOUS",
                        ignoreCase = true
                    ) -> {

                        "🟠 SUSPICIOUS"
                    }


                    else -> {

                        "🟢 VOICE APPEARS REAL"
                    }
                }


            verdictText?.setTextColor(

                when {

                    verdict.contains(
                        "HIGH RISK",
                        ignoreCase = true
                    ) ||
                            verdict.contains(
                                "SPOOF",
                                ignoreCase = true
                            ) -> {

                        Color.rgb(
                            255,
                            82,
                            82
                        )
                    }


                    verdict.contains(
                        "SUSPICIOUS",
                        ignoreCase = true
                    ) -> {

                        Color.rgb(
                            255,
                            179,
                            0
                        )
                    }


                    else -> {

                        Color.rgb(
                            0,
                            230,
                            118
                        )
                    }
                }
            )


            statusText?.text =
                "Analysis updated • LIVE"
        }
    }


    // =========================================================
    // UPDATE STATUS
    // =========================================================

    private fun updateStatus(
        message: String
    ) {

        if (isStopping) {
            return
        }


        serviceScope.launch(
            Dispatchers.Main
        ) {

            if (!isStopping) {

                statusText?.text =
                    message
            }
        }
    }


    // =========================================================
    // RELEASE AUDIO
    // =========================================================

    private fun releaseAudio() {

        try {

            audioRecord?.stop()

        } catch (
            _: Exception
        ) {
        }


        try {

            audioRecord?.release()

        } catch (
            _: Exception
        ) {
        }


        audioRecord =
            null
    }


    // =========================================================
    // REMOVE OVERLAY
    // =========================================================

    private fun removeOverlay() {

        try {

            overlayView?.let {

                windowManager?.removeView(
                    it
                )
            }

        } catch (
            _: Exception
        ) {
        }


        overlayView =
            null

        windowManager =
            null

        aiText =
            null

        scamText =
            null

        verdictText =
            null

        statusText =
            null
    }


    // =========================================================
    // COMPLETE STOP
    // =========================================================

    private fun stopServiceCompletely() {

        if (isStopping) {
            return
        }


        isStopping =
            true


        // -----------------------------------------------------
        // STOP COROUTINE
        // -----------------------------------------------------

        processingJob?.cancel()

        processingJob =
            null


        // -----------------------------------------------------
        // STOP AUDIO
        // -----------------------------------------------------

        releaseAudio()


        // -----------------------------------------------------
        // REMOVE OVERLAY
        // -----------------------------------------------------

        removeOverlay()


        // -----------------------------------------------------
        // REMOVE FOREGROUND SERVICE
        // -----------------------------------------------------

        if (
            Build.VERSION.SDK_INT >=
            Build.VERSION_CODES.N
        ) {

            stopForeground(
                STOP_FOREGROUND_REMOVE
            )

        } else {

            @Suppress("DEPRECATION")

            stopForeground(
                true
            )
        }


        // -----------------------------------------------------
        // STOP SERVICE
        // -----------------------------------------------------

        stopSelf()
    }


    // =========================================================
    // NOTIFICATION CHANNEL
    // =========================================================

    private fun createNotificationChannel() {

        if (
            Build.VERSION.SDK_INT >=
            Build.VERSION_CODES.O
        ) {

            val channel =
                NotificationChannel(
                    CHANNEL_ID,
                    "VIGILVOICE Live Protection",
                    NotificationManager
                        .IMPORTANCE_LOW
                )


            channel.description =
                "Live voice protection is active"


            val manager =
                getSystemService(
                    NotificationManager::class.java
                )


            manager.createNotificationChannel(
                channel
            )
        }
    }


    // =========================================================
    // NOTIFICATION
    // =========================================================

    private fun createNotification():
            Notification {

        return NotificationCompat
            .Builder(
                this,
                CHANNEL_ID
            )

            .setContentTitle(
                "VIGILVOICE Protection Active"
            )

            .setContentText(
                "Monitoring permitted microphone audio"
            )

            .setSmallIcon(
                android.R.drawable
                    .ic_btn_speak_now
            )

            .setOngoing(true)

            .setCategory(
                NotificationCompat
                    .CATEGORY_SERVICE
            )

            .build()
    }


    // =========================================================
    // DESTROY
    // =========================================================

    override fun onDestroy() {

        isStopping =
            true


        processingJob?.cancel()

        processingJob =
            null


        releaseAudio()

        removeOverlay()


        try {

            if (
                Build.VERSION.SDK_INT >=
                Build.VERSION_CODES.N
            ) {

                stopForeground(
                    STOP_FOREGROUND_REMOVE
                )
            }

        } catch (
            _: Exception
        ) {
        }


        serviceScope.cancel()


        super.onDestroy()
    }


    // =========================================================
    // BIND
    // =========================================================

    override fun onBind(
        intent: Intent?
    ): IBinder? {

        return null
    }
}