package com.vigilvoice.mobile

import com.vigilvoice.mobile.calling.ui.CallScreen

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.media.MediaRecorder
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.provider.Settings
import android.util.Log

import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll

import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue

import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

import androidx.core.content.ContextCompat

import com.google.gson.JsonArray

import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody

import java.io.File
import java.io.FileOutputStream
import java.util.Locale
import kotlin.math.abs
import kotlin.math.sin


// ============================================================
// SWARAKSHA COLOUR PALETTE
// ============================================================

private val NavyBlue = Color(0xFF1257A6)
private val BrightBlue = Color(0xFF1688E8)
private val DeepBlue = Color(0xFF0B3D78)

private val Green = Color(0xFF16A673)
private val DarkGreen = Color(0xFF087653)
private val LightGreen = Color(0xFFE7F8F1)

private val Orange = Color(0xFFFF8A16)
private val DarkOrange = Color(0xFFE86D00)
private val LightOrange = Color(0xFFFFF0DE)

private val LightBlue = Color(0xFFEAF5FF)

private val Background = Color(0xFFF7FBFF)

private val TextDark = Color(0xFF173B63)
private val TextGrey = Color(0xFF58708D)

private val DangerRed = Color(0xFFD92D20)
private val DangerLight = Color(0xFFFFE9E7)


// ============================================================
// MAIN ACTIVITY
// ============================================================

class MainActivity : ComponentActivity() {

    override fun onCreate(
        savedInstanceState: Bundle?
    ) {

        super.onCreate(
            savedInstanceState
        )

        setContent {

            MaterialTheme {

                SwarakshaApp()
            }
        }
    }
}


// ============================================================
// MAIN APP
// ============================================================

@Composable
fun SwarakshaApp() {

    val context =
        LocalContext.current

    val scope =
        rememberCoroutineScope()


    // ========================================================
    // DEVICE ROLE / PEER ID
    // ========================================================

    val preferences =
        remember {

            context.getSharedPreferences(
                "swaraksha_config",
                Context.MODE_PRIVATE
            )
        }


    var peerId by remember {

        mutableStateOf(

            preferences.getString(
                "peer_id",
                ""
            ) ?: ""
        )
    }


    var showCallScreen by remember {

        mutableStateOf(false)
    }


    // ========================================================
    // RECORDING
    // ========================================================

    var isRecording by remember {
        mutableStateOf(false)
    }

    var isAnalyzing by remember {
        mutableStateOf(false)
    }

    var recordingTime by remember {
        mutableIntStateOf(0)
    }

    var amplitude by remember {
        mutableFloatStateOf(0f)
    }

    var recordedFile by remember {
        mutableStateOf<File?>(null)
    }

    var selectedFileName by remember {
        mutableStateOf("")
    }

    var statusMessage by remember {

        mutableStateOf(
            "System ready to protect you."
        )
    }


    // ========================================================
    // AI RESULTS
    // ========================================================

    var verdict by remember {
        mutableStateOf("")
    }

    var riskScore by remember {
        mutableFloatStateOf(0f)
    }

    var aiProbability by remember {
        mutableFloatStateOf(0f)
    }

    var realProbability by remember {
        mutableFloatStateOf(0f)
    }

    var scamRisk by remember {
        mutableFloatStateOf(0f)
    }

    var modelName by remember {
        mutableStateOf("Spectra-AASIST3")
    }

    var transcript by remember {
        mutableStateOf("")
    }

    var durationSeconds by remember {
        mutableFloatStateOf(0f)
    }

    var segmentCount by remember {
        mutableIntStateOf(0)
    }

    var reasons by remember {
        mutableStateOf<List<String>>(emptyList())
    }

    var scamVerdict by remember {
        mutableStateOf("")
    }

    var securityDecision by remember {
        mutableStateOf("")
    }

    var securityActionType by remember {
        mutableStateOf("")
    }

    var securityRecommendation by remember {
        mutableStateOf("")
    }

    var speakerVerdict by remember {
        mutableStateOf("")
    }

    var speakerSimilarity by remember {
        mutableFloatStateOf(0f)
    }

    var impersonationStatus by remember {
        mutableStateOf("")
    }


    var recorder by remember {
        mutableStateOf<MediaRecorder?>(null)
    }


    // ========================================================
    // CLEAR RESULTS
    // ========================================================

    fun clearResults() {

        verdict = ""

        riskScore = 0f

        aiProbability = 0f

        realProbability = 0f

        scamRisk = 0f

        transcript = ""

        durationSeconds = 0f

        segmentCount = 0

        reasons = emptyList()

        scamVerdict = ""

        securityDecision = ""

        securityActionType = ""

        securityRecommendation = ""

        speakerVerdict = ""

        speakerSimilarity = 0f

        impersonationStatus = ""

        statusMessage =
            "System ready to protect you."
    }


    // ========================================================
    // MICROPHONE PERMISSION
    // ========================================================

    val microphonePermissionLauncher =
        rememberLauncherForActivityResult(

            ActivityResultContracts.RequestPermission()

        ) { granted ->

            if (granted) {

                statusMessage =
                    "Microphone permission granted."

            } else {

                statusMessage =
                    "Microphone permission required."
            }
        }


    // ========================================================
    // START LIVE PROTECTION
    // ========================================================

    fun startLiveProtection() {

        if (
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {

            microphonePermissionLauncher.launch(
                Manifest.permission.RECORD_AUDIO
            )

            return
        }


        if (!Settings.canDrawOverlays(context)) {

            statusMessage =
                "Allow overlay permission to enable live protection."

            val intent =
                Intent(
                    Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                    Uri.parse(
                        "package:${context.packageName}"
                    )
                )

            context.startActivity(intent)

            return
        }


        val intent =
            Intent(
                context,
                LiveProtectionService::class.java
            )


        ContextCompat.startForegroundService(
            context,
            intent
        )


        statusMessage =
            "Live protection is running."
    }


    // ========================================================
    // STOP LIVE PROTECTION
    // ========================================================

    fun stopLiveProtection() {

        try {

            val intent =
                Intent(
                    context,
                    LiveProtectionService::class.java
                )

            context.stopService(
                intent
            )

        } catch (e: Exception) {

            Log.e(
                "SWARAKSHA",
                "Unable to stop live service",
                e
            )
        }


        statusMessage =
            "Live protection stopped."
    }


    // ========================================================
    // FILE PICKER
    // ========================================================

    val filePickerLauncher =
        rememberLauncherForActivityResult(

            ActivityResultContracts.GetContent()

        ) { uri: Uri? ->

            if (uri == null) {
                return@rememberLauncherForActivityResult
            }


            scope.launch {

                try {

                    val fileName =
                        getFileName(
                            context,
                            uri
                        )


                    val extension =
                        fileName
                            .substringAfterLast(
                                ".",
                                "wav"
                            )


                    val outputFile =
                        File(
                            context.cacheDir,
                            "selected_audio.$extension"
                        )


                    context.contentResolver
                        .openInputStream(uri)
                        ?.use { input ->

                            FileOutputStream(
                                outputFile
                            ).use { output ->

                                input.copyTo(
                                    output
                                )
                            }
                        }


                    recordedFile =
                        outputFile


                    selectedFileName =
                        fileName


                    clearResults()


                    selectedFileName =
                        fileName


                    Log.d(
                        "SWARAKSHA_FILE",
                        "========================================"
                    )

                    Log.d(
                        "SWARAKSHA_FILE",
                        "FILE SELECTED"
                    )

                    Log.d(
                        "SWARAKSHA_FILE",
                        "Original name = $fileName"
                    )

                    Log.d(
                        "SWARAKSHA_FILE",
                        "Extension = $extension"
                    )

                    Log.d(
                        "SWARAKSHA_FILE",
                        "Android cache path = ${outputFile.absolutePath}"
                    )

                    Log.d(
                        "SWARAKSHA_FILE",
                        "Copied file size = ${outputFile.length()} bytes"
                    )

                    Log.d(
                        "SWARAKSHA_FILE",
                        "========================================"
                    )


                    statusMessage =
                        "Audio selected. Ready for AI analysis."

                } catch (e: Exception) {

                    Log.e(
                        "SWARAKSHA_FILE",
                        "File selection failed",
                        e
                    )


                    statusMessage =
                        "Unable to select audio: ${e.message}"
                }
            }
        }


    // ========================================================
    // START RECORDING
    // ========================================================

    fun startRecording() {

        if (
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {

            microphonePermissionLauncher.launch(
                Manifest.permission.RECORD_AUDIO
            )

            return
        }


        try {

            val file =
                File(
                    context.cacheDir,
                    "swaraksha_recording.m4a"
                )


            val mediaRecorder =
                MediaRecorder()


            mediaRecorder.setAudioSource(
                MediaRecorder.AudioSource.MIC
            )


            mediaRecorder.setOutputFormat(
                MediaRecorder.OutputFormat.MPEG_4
            )


            mediaRecorder.setAudioEncoder(
                MediaRecorder.AudioEncoder.AAC
            )


            mediaRecorder.setAudioEncodingBitRate(
                128000
            )


            mediaRecorder.setAudioSamplingRate(
                44100
            )


            mediaRecorder.setOutputFile(
                file.absolutePath
            )


            mediaRecorder.prepare()


            mediaRecorder.start()


            recorder =
                mediaRecorder


            recordedFile =
                file


            isRecording =
                true


            recordingTime =
                0


            selectedFileName =
                "swaraksha_recording.m4a"


            clearResults()


            selectedFileName =
                "swaraksha_recording.m4a"


            statusMessage =
                "Recording your voice..."


            // =================================================
            // TIMER
            // =================================================

            scope.launch {

                while (isRecording) {

                    delay(1000)

                    if (isRecording) {

                        recordingTime++
                    }
                }
            }


            // =================================================
            // AMPLITUDE
            // =================================================

            scope.launch {

                while (isRecording) {

                    try {

                        val value =
                            mediaRecorder.maxAmplitude


                        amplitude =
                            (
                                value / 32767f
                            ).coerceIn(
                                0f,
                                1f
                            )

                    } catch (_: Exception) {
                    }


                    delay(100)
                }
            }

        } catch (e: Exception) {

            Log.e(
                "SWARAKSHA",
                "Recording failed",
                e
            )


            statusMessage =
                "Recording failed: ${e.message}"


            isRecording =
                false
        }
    }


    // ========================================================
    // STOP RECORDING
    // ========================================================

    fun stopRecording() {

        try {

            recorder?.stop()

        } catch (_: Exception) {
        }


        try {

            recorder?.release()

        } catch (_: Exception) {
        }


        recorder =
            null


        isRecording =
            false


        amplitude =
            0f


        statusMessage =
            "Recording complete. Ready for AI analysis."
    }


    // ========================================================
    // ANALYZE VOICE
    // ========================================================

    fun analyzeVoice() {

        val file =
            recordedFile


        // =====================================================
        // FILE VALIDATION
        // =====================================================

        if (
            file == null ||
            !file.exists()
        ) {

            statusMessage =
                "Please record or upload an audio file first."

            return
        }


        if (file.length() <= 0L) {

            statusMessage =
                "The selected audio file is empty."

            return
        }


        Log.d(
            "SWARAKSHA_FILE",
            "========================================"
        )


        Log.d(
            "SWARAKSHA_FILE",
            "STARTING AUDIO ANALYSIS"
        )


        Log.d(
            "SWARAKSHA_FILE",
            "File name = ${file.name}"
        )


        Log.d(
            "SWARAKSHA_FILE",
            "File path = ${file.absolutePath}"
        )


        Log.d(
            "SWARAKSHA_FILE",
            "File size = ${file.length()} bytes"
        )


        Log.d(
            "SWARAKSHA_FILE",
            "Selected display name = $selectedFileName"
        )


        Log.d(
            "SWARAKSHA_FILE",
            "========================================"
        )


        scope.launch {

            isAnalyzing =
                true


            statusMessage =
                "Analyzing voice with Spectra-AASIST3..."


            try {

                // =================================================
                // READ FILE
                // =================================================

                val fileBytes =
                    file.readBytes()


                Log.d(
                    "SWARAKSHA_FILE",
                    "Bytes loaded into memory = ${fileBytes.size}"
                )


                // =================================================
                // REQUEST BODY
                // =================================================

                val requestBody =
                    fileBytes.toRequestBody(
                        "application/octet-stream"
                            .toMediaType()
                    )


                // =================================================
                // MULTIPART
                // =================================================

                val multipart =
                    MultipartBody.Part.createFormData(
                        "file",
                        file.name,
                        requestBody
                    )


                Log.d(
                    "SWARAKSHA_FILE",
                    "UPLOAD FILE NAME = ${file.name}"
                )


                Log.d(
                    "SWARAKSHA_FILE",
                    "UPLOAD FILE SIZE = ${fileBytes.size} bytes"
                )


                // =================================================
                // API CALL
                // =================================================

                Log.d(
                    "SWARAKSHA_API",
                    "Sending audio to backend..."
                )


                val response =
                    RetrofitClient
                        .api
                        .analyzeVoice(
                            multipart
                        )


                // =================================================
                // HTTP RESPONSE
                // =================================================

                Log.d(
                    "SWARAKSHA_API",
                    "HTTP CODE = ${response.code()}"
                )


                Log.d(
                    "SWARAKSHA_API",
                    "HTTP SUCCESS = ${response.isSuccessful}"
                )


                // =================================================
                // HTTP ERROR
                // =================================================

                if (!response.isSuccessful) {

                    statusMessage =
                        "Server error: HTTP ${response.code()}"


                    Log.e(
                        "SWARAKSHA_API",
                        "Backend HTTP error: ${response.code()}"
                    )


                    return@launch
                }


                // =================================================
                // JSON BODY
                // =================================================

                val json =
                    response.body()


                if (json == null) {

                    statusMessage =
                        "Server returned an empty response."


                    Log.e(
                        "SWARAKSHA_API",
                        "Response body is NULL"
                    )


                    return@launch
                }


                // =================================================
                // FULL RESPONSE
                // =================================================

                Log.d(
                    "SWARAKSHA_API",
                    "========================================"
                )


                Log.d(
                    "SWARAKSHA_API",
                    "FULL BACKEND RESPONSE:"
                )


                Log.d(
                    "SWARAKSHA_API",
                    json.toString()
                )


                Log.d(
                    "SWARAKSHA_API",
                    "========================================"
                )


                // =================================================
                // FINAL RESULT
                // =================================================

                val finalResult =
                    json.getAsJsonObject(
                        "final_result"
                    )


                if (finalResult == null) {

                    statusMessage =
                        "Invalid server response: final_result missing."


                    Log.e(
                        "SWARAKSHA_API",
                        "final_result missing"
                    )


                    return@launch
                }


                verdict =
                    finalResult
                        .get("verdict")
                        ?.asString
                        ?: "UNKNOWN"


                riskScore =
                    finalResult
                        .get("risk_score")
                        ?.asFloat
                        ?: 0f


                // =================================================
                // VOICE AI
                // =================================================

                val voiceAI =
                    json.getAsJsonObject(
                        "voice_ai"
                    )


                if (voiceAI == null) {

                    statusMessage =
                        "Invalid server response: voice_ai missing."


                    Log.e(
                        "SWARAKSHA_API",
                        "voice_ai missing"
                    )


                    return@launch
                }


                // =================================================
                // AI PROBABILITY
                // =================================================

                val aiElement =
                    voiceAI.get(
                        "ai_probability"
                    )


                if (
                    aiElement == null ||
                    aiElement.isJsonNull
                ) {

                    statusMessage =
                        "AI probability missing from server."


                    Log.e(
                        "SWARAKSHA_API",
                        "ai_probability missing: $voiceAI"
                    )


                    return@launch
                }


                aiProbability =
                    aiElement.asFloat


                // =================================================
                // REAL PROBABILITY
                // =================================================

                val realElement =
                    voiceAI.get(
                        "real_probability"
                    )


                if (
                    realElement == null ||
                    realElement.isJsonNull
                ) {

                    statusMessage =
                        "Real probability missing from server."


                    Log.e(
                        "SWARAKSHA_API",
                        "real_probability missing: $voiceAI"
                    )


                    return@launch
                }


                realProbability =
                    realElement.asFloat


                // =================================================
                // MODEL
                // =================================================

                modelName =
                    voiceAI
                        .get(
                            "model"
                        )
                        ?.asString
                        ?: "Spectra-AASIST3"


                // =================================================
                // TRANSCRIPT
                // =================================================

                val transcriptObject =
                    json.getAsJsonObject(
                        "transcript"
                    )


                transcript =
                    transcriptObject
                        ?.get("text")
                        ?.asString
                        ?: ""


                // =================================================
                // SCAM ANALYSIS
                // =================================================

                val scamAnalysis =
                    json.getAsJsonObject(
                        "scam_analysis"
                    )


                scamRisk =
                    scamAnalysis
                        ?.get("risk_score")
                        ?.asFloat
                        ?: 0f


                scamVerdict =
                    scamAnalysis
                        ?.get("verdict")
                        ?.asString
                        ?: ""


                // =================================================
                // RISK ANALYSIS
                // =================================================

                val riskAnalysis =
                    json.getAsJsonObject(
                        "risk_analysis"
                    )


                val reasonArray =
                    riskAnalysis
                        ?.getAsJsonArray(
                            "reasons"
                        )


                reasons =
                    jsonArrayToList(
                        reasonArray
                    )


                // =================================================
                // AUDIO INFO
                // =================================================

                durationSeconds =
                    voiceAI
                        .get(
                            "duration"
                        )
                        ?.asFloat
                        ?: 0f


                segmentCount =
                    voiceAI
                        .get(
                            "segments"
                        )
                        ?.asInt
                        ?: 0


                // =================================================
                // SECURITY ACTION
                // =================================================

                val securityAction =
                    json.getAsJsonObject(
                        "security_action"
                    )


                securityDecision =
                    securityAction
                        ?.get("decision")
                        ?.asString
                        ?: ""


                securityActionType =
                    securityAction
                        ?.get("action_type")
                        ?.asString
                        ?: ""


                securityRecommendation =
                    securityAction
                        ?.get("recommendation")
                        ?.asString
                        ?: ""


                // =================================================
                // SPEAKER VERIFICATION
                // =================================================

                val speakerVerification =
                    json.getAsJsonObject(
                        "speaker_verification"
                    )


                speakerVerdict =
                    speakerVerification
                        ?.get("verdict")
                        ?.asString
                        ?: ""


                speakerSimilarity =
                    speakerVerification
                        ?.get("similarity_indicator")
                        ?.asFloat
                        ?: 0f


                // =================================================
                // IMPERSONATION DETECTION
                // =================================================

                val impersonationDetection =
                    json.getAsJsonObject(
                        "impersonation_detection"
                    )


                impersonationStatus =
                    impersonationDetection
                        ?.get("status")
                        ?.asString
                        ?: ""


                // =================================================
                // DEBUG
                // =================================================

                Log.d(
                    "SWARAKSHA_API",
                    "VERDICT = $verdict"
                )


                Log.d(
                    "SWARAKSHA_API",
                    "OVERALL RISK = $riskScore"
                )


                Log.d(
                    "SWARAKSHA_API",
                    "AI PROBABILITY = $aiProbability"
                )


                Log.d(
                    "SWARAKSHA_API",
                    "REAL PROBABILITY = $realProbability"
                )


                Log.d(
                    "SWARAKSHA_API",
                    "SCAM RISK = $scamRisk"
                )


                Log.d(
                    "SWARAKSHA_API",
                    "MODEL = $modelName"
                )


                Log.d(
                    "SWARAKSHA_API",
                    "DURATION = $durationSeconds"
                )


                Log.d(
                    "SWARAKSHA_API",
                    "SEGMENTS = $segmentCount"
                )


                Log.d(
                    "SWARAKSHA_API",
                    "SECURITY DECISION = $securityDecision"
                )


                Log.d(
                    "SWARAKSHA_API",
                    "SPEAKER VERDICT = $speakerVerdict"
                )


                Log.d(
                    "SWARAKSHA_API",
                    "IMPERSONATION = $impersonationStatus"
                )


                Log.d(
                    "SWARAKSHA_API",
                    "========================================"
                )


                // =================================================
                // PROBABILITY VALIDATION
                // =================================================

                if (
                    aiProbability < 0f ||
                    aiProbability > 100f ||
                    realProbability < 0f ||
                    realProbability > 100f
                ) {

                    statusMessage =
                        "Invalid probability received from server."


                    Log.e(
                        "SWARAKSHA_API",
                        "INVALID PROBABILITY"
                    )


                    return@launch
                }


                statusMessage =
                    "Analysis completed successfully."

            } catch (e: Exception) {

                Log.e(
                    "SWARAKSHA_API",
                    "Analysis failed",
                    e
                )


                statusMessage =
                    "Analysis failed: ${e.message}"

            } finally {

                isAnalyzing =
                    false
            }
        }
    }


    // ========================================================
    // CALL SCREEN
    // ========================================================

    if (showCallScreen) {

        if (peerId.isEmpty()) {

            DeviceRoleScreen(

                onPhone1 = {

                    preferences
                        .edit()
                        .putString(
                            "peer_id",
                            "phone1"
                        )
                        .apply()


                    peerId =
                        "phone1"
                },


                onPhone2 = {

                    preferences
                        .edit()
                        .putString(
                            "peer_id",
                            "phone2"
                        )
                        .apply()


                    peerId =
                        "phone2"
                },


                onBack = {

                    showCallScreen =
                        false
                }
            )

        } else {

            CallScreen(

                roomId =
                    "demo-room",

                peerId =
                    peerId,

                onBack = {

                    showCallScreen =
                        false
                }
            )
        }


        return
    }


    // ========================================================
    // MAIN HOME UI
    // ========================================================

    Surface(

        modifier =
            Modifier.fillMaxSize(),

        color =
            Background
    ) {

        Column(

            modifier =
                Modifier
                    .fillMaxSize()
                    .verticalScroll(
                        rememberScrollState()
                    )
                    .padding(
                        horizontal = 18.dp,
                        vertical = 14.dp
                    )
        ) {

            HeaderSection()


            Spacer(
                modifier =
                    Modifier.height(18.dp)
            )


            // =================================================
            // SECURE CALL
            // =================================================

            Button(

                onClick = {

                    showCallScreen =
                        true
                },

                modifier =
                    Modifier
                        .fillMaxWidth()
                        .height(56.dp),

                shape =
                    RoundedCornerShape(16.dp),

                colors =
                    ButtonDefaults.buttonColors(
                        containerColor =
                            NavyBlue
                    )
            ) {

                Text(
                    "OPEN SWARAKSHA CALL",

                    fontWeight =
                        FontWeight.ExtraBold
                )
            }


            Spacer(
                modifier =
                    Modifier.height(14.dp)
            )


            // =================================================
            // DEVICE ROLE INFO
            // =================================================

            if (peerId.isNotEmpty()) {

                Card(

                    modifier =
                        Modifier.fillMaxWidth(),

                    shape =
                        RoundedCornerShape(16.dp),

                    colors =
                        CardDefaults.cardColors(
                            containerColor =
                                LightBlue
                        )
                ) {

                    Row(

                        modifier =
                            Modifier
                                .fillMaxWidth()
                                .padding(14.dp),

                        verticalAlignment =
                            Alignment.CenterVertically,

                        horizontalArrangement =
                            Arrangement.SpaceBetween
                    ) {

                        Column {

                            Text(
                                "CALL DEVICE",

                                fontSize =
                                    10.sp,

                                fontWeight =
                                    FontWeight.Bold,

                                color =
                                    TextGrey
                            )

                            Text(
                                peerId.uppercase(),

                                fontSize =
                                    16.sp,

                                fontWeight =
                                    FontWeight.ExtraBold,

                                color =
                                    NavyBlue
                            )
                        }


                        OutlinedButton(

                            onClick = {

                                preferences
                                    .edit()
                                    .remove(
                                        "peer_id"
                                    )
                                    .apply()


                                peerId =
                                    ""
                            }
                        ) {

                            Text(
                                "CHANGE"
                            )
                        }
                    }
                }


                Spacer(
                    modifier =
                        Modifier.height(14.dp)
                )
            }


            ReadyCard(
                statusMessage =
                    statusMessage
            )


            Spacer(
                modifier =
                    Modifier.height(14.dp)
            )


            WaveformCard(

                isRecording =
                    isRecording,

                amplitude =
                    amplitude
            )


            Spacer(
                modifier =
                    Modifier.height(14.dp)
            )


            LiveProtectionButtons(

                onStart = {
                    startLiveProtection()
                },

                onStop = {
                    stopLiveProtection()
                }
            )


            Spacer(
                modifier =
                    Modifier.height(14.dp)
            )


            ActionCard(

                title =
                    if (isRecording)
                        "STOP VOICE RECORDING"
                    else
                        "START VOICE RECORDING",

                subtitle =
                    if (isRecording)
                        "Recording ${formatTime(recordingTime)}"
                    else
                        "Tap to record your voice",

                iconText =
                    if (isRecording)
                        "■"
                    else
                        "●",

                background =
                    LightGreen,

                borderColor =
                    Green,

                iconBackground =
                    Green,

                titleColor =
                    DarkGreen,

                onClick = {

                    if (isRecording) {

                        stopRecording()

                    } else {

                        startRecording()
                    }
                }
            )


            Spacer(
                modifier =
                    Modifier.height(12.dp)
            )


            ActionCard(

                title =
                    "UPLOAD VOICE RECORDING",

                subtitle =
                    if (
                        selectedFileName.isNotEmpty()
                    ) {

                        selectedFileName

                    } else {

                        "Choose an audio file from your device"
                    },

                iconText =
                    "↑",

                background =
                    LightBlue,

                borderColor =
                    BrightBlue,

                iconBackground =
                    BrightBlue,

                titleColor =
                    NavyBlue,

                onClick = {

                    filePickerLauncher.launch(
                        "audio/*"
                    )
                }
            )


            Spacer(
                modifier =
                    Modifier.height(12.dp)
            )


            ActionCard(

                title =
                    "ANALYZE WITH AI",

                subtitle =
                    "Detect deepfake, spoofed and scam voices",

                iconText =
                    "⌕",

                background =
                    LightOrange,

                borderColor =
                    Orange,

                iconBackground =
                    Orange,

                titleColor =
                    DarkOrange,

                enabled =
                    !isAnalyzing,

                onClick = {

                    analyzeVoice()
                }
            )


            Spacer(
                modifier =
                    Modifier.height(14.dp)
            )


            AnimatedVisibility(
                visible =
                    isAnalyzing
            ) {

                AnalyzingCard()
            }


            AnimatedVisibility(
                visible =
                    verdict.isNotEmpty()
            ) {

                Column {

                    Spacer(
                        modifier =
                            Modifier.height(4.dp)
                    )


                    ResultCard(

                        verdict =
                            verdict,

                        riskScore =
                            riskScore,

                        aiProbability =
                            aiProbability,

                        realProbability =
                            realProbability,

                        scamRisk =
                            scamRisk,

                        modelName =
                            modelName,

                        transcript =
                            transcript,

                        durationSeconds =
                            durationSeconds,

                        segmentCount =
                            segmentCount,

                        reasons =
                            reasons,

                        scamVerdict =
                            scamVerdict,

                        securityDecision =
                            securityDecision,

                        securityActionType =
                            securityActionType,

                        securityRecommendation =
                            securityRecommendation,

                        speakerVerdict =
                            speakerVerdict,

                        speakerSimilarity =
                            speakerSimilarity,

                        impersonationStatus =
                            impersonationStatus
                    )


                    Spacer(
                        modifier =
                            Modifier.height(18.dp)
                    )


                    OutlinedButton(

                        onClick = {

                            clearResults()

                            recordedFile =
                                null

                            selectedFileName =
                                ""
                        },

                        modifier =
                            Modifier
                                .fillMaxWidth()
                                .height(52.dp),

                        shape =
                            RoundedCornerShape(16.dp)
                    ) {

                        Text(
                            "×",

                            fontSize =
                                22.sp,

                            fontWeight =
                                FontWeight.Bold
                        )


                        Spacer(
                            modifier =
                                Modifier.width(8.dp)
                        )


                        Text(
                            "CLEAR RESULTS",

                            fontWeight =
                                FontWeight.Bold
                        )
                    }
                }
            }


            Spacer(
                modifier =
                    Modifier.height(20.dp)
            )
        }
    }
}


// ============================================================
// DEVICE ROLE SCREEN
// ============================================================

@Composable
fun DeviceRoleScreen(

    onPhone1: () -> Unit,

    onPhone2: () -> Unit,

    onBack: () -> Unit
) {

    Surface(

        modifier =
            Modifier.fillMaxSize(),

        color =
            Background
    ) {

        Column(

            modifier =
                Modifier
                    .fillMaxSize()
                    .padding(28.dp),

            horizontalAlignment =
                Alignment.CenterHorizontally,

            verticalArrangement =
                Arrangement.Center
        ) {

            Text(

                text =
                    "SWARAKSHA",

                fontSize =
                    32.sp,

                fontWeight =
                    FontWeight.ExtraBold,

                color =
                    NavyBlue
            )


            Spacer(
                modifier =
                    Modifier.height(8.dp)
            )


            Text(

                text =
                    "SECURE CALL SETUP",

                fontSize =
                    12.sp,

                fontWeight =
                    FontWeight.Bold,

                letterSpacing =
                    1.5.sp,

                color =
                    TextGrey
            )


            Spacer(
                modifier =
                    Modifier.height(12.dp)
            )


            Text(

                text =
                    "Select the role for this device",

                fontSize =
                    13.sp,

                color =
                    TextGrey
            )


            Spacer(
                modifier =
                    Modifier.height(35.dp)
            )


            // =================================================
            // PHONE 1
            // =================================================

            Button(

                onClick =
                    onPhone1,

                modifier =
                    Modifier
                        .fillMaxWidth()
                        .height(62.dp),

                shape =
                    RoundedCornerShape(18.dp),

                colors =
                    ButtonDefaults.buttonColors(
                        containerColor =
                            NavyBlue
                    )
            ) {

                Text(

                    text =
                        "PHONE 1",

                    fontSize =
                        14.sp,

                    fontWeight =
                        FontWeight.ExtraBold
                )
            }


            Spacer(
                modifier =
                    Modifier.height(14.dp)
            )


            // =================================================
            // PHONE 2
            // =================================================

            Button(

                onClick =
                    onPhone2,

                modifier =
                    Modifier
                        .fillMaxWidth()
                        .height(62.dp),

                shape =
                    RoundedCornerShape(18.dp),

                colors =
                    ButtonDefaults.buttonColors(
                        containerColor =
                            BrightBlue
                    )
            ) {

                Text(

                    text =
                        "PHONE 2",

                    fontSize =
                        14.sp,

                    fontWeight =
                        FontWeight.ExtraBold
                )
            }


            Spacer(
                modifier =
                    Modifier.height(28.dp)
            )


            // =================================================
            // INFO CARD
            // =================================================

            Card(

                modifier =
                    Modifier.fillMaxWidth(),

                shape =
                    RoundedCornerShape(18.dp),

                colors =
                    CardDefaults.cardColors(
                        containerColor =
                            LightBlue
                    )
            ) {

                Column(

                    modifier =
                        Modifier.padding(16.dp),

                    horizontalAlignment =
                        Alignment.CenterHorizontally
                ) {

                    Text(

                        text =
                            "DEVICE ROLE",

                        fontSize =
                            11.sp,

                        fontWeight =
                            FontWeight.ExtraBold,

                        color =
                            NavyBlue
                    )


                    Spacer(
                        modifier =
                            Modifier.height(5.dp)
                    )


                    Text(

                        text =
                            "Select once. SWARAKSHA will remember this device.",

                        fontSize =
                            11.sp,

                        color =
                            TextGrey,

                        textAlign =
                            TextAlign.Center
                    )
                }
            }


            Spacer(
                modifier =
                    Modifier.height(18.dp)
            )


            // =================================================
            // BACK
            // =================================================

            OutlinedButton(

                onClick =
                    onBack,

                modifier =
                    Modifier.fillMaxWidth(),

                shape =
                    RoundedCornerShape(16.dp)
            ) {

                Text(

                    text =
                        "BACK",

                    fontSize =
                        11.sp,

                    fontWeight =
                        FontWeight.Bold
                )
            }
        }
    }
}


// ============================================================
// HEADER
// ============================================================

@Composable
fun HeaderSection() {

    Box(

        modifier =
            Modifier
                .fillMaxWidth()
                .clip(
                    RoundedCornerShape(24.dp)
                )
                .background(
                    Brush.horizontalGradient(
                        listOf(
                            LightBlue,
                            Color.White,
                            LightGreen
                        )
                    )
                )
                .padding(
                    horizontal = 18.dp,
                    vertical = 20.dp
                )
    ) {

        Row(
            verticalAlignment =
                Alignment.CenterVertically
        ) {

            Box(

                modifier =
                    Modifier
                        .size(72.dp)
                        .clip(CircleShape)
                        .background(
                            Brush.linearGradient(
                                listOf(
                                    NavyBlue,
                                    Green
                                )
                            )
                        ),

                contentAlignment =
                    Alignment.Center
            ) {

                Text(
                    "🛡",

                    fontSize =
                        38.sp,

                    color =
                        Color.White
                )
            }


            Spacer(
                modifier =
                    Modifier.width(14.dp)
            )


            Column(
                modifier =
                    Modifier.weight(1f)
            ) {

                Row {

                    Text(
                        "SWA",

                        fontSize =
                            28.sp,

                        fontWeight =
                            FontWeight.ExtraBold,

                        color =
                            NavyBlue
                    )


                    Text(
                        "RAKSHA",

                        fontSize =
                            28.sp,

                        fontWeight =
                            FontWeight.ExtraBold,

                        color =
                            Green
                    )
                }


                Text(
                    "AI VOICE PROTECTION",

                    fontSize =
                        11.sp,

                    letterSpacing =
                        2.sp,

                    fontWeight =
                        FontWeight.Bold,

                    color =
                        NavyBlue
                )


                Spacer(
                    modifier =
                        Modifier.height(4.dp)
                )


                Text(
                    "Detect AI-generated and spoofed voices",

                    fontSize =
                        12.sp,

                    color =
                        TextGrey
                )
            }
        }


        Box(

            modifier =
                Modifier
                    .size(16.dp)
                    .clip(CircleShape)
                    .background(Orange)
                    .align(Alignment.TopEnd)
        )
    }
}


// ============================================================
// READY CARD
// ============================================================

@Composable
fun ReadyCard(
    statusMessage: String
) {

    Card(

        modifier =
            Modifier
                .fillMaxWidth()
                .shadow(
                    4.dp,
                    RoundedCornerShape(22.dp)
                ),

        shape =
            RoundedCornerShape(22.dp),

        colors =
            CardDefaults.cardColors(
                containerColor =
                    LightGreen
            )
    ) {

        Row(

            modifier =
                Modifier
                    .fillMaxWidth()
                    .padding(20.dp),

            verticalAlignment =
                Alignment.CenterVertically
        ) {

            Box(

                modifier =
                    Modifier
                        .size(48.dp)
                        .clip(CircleShape)
                        .background(Green),

                contentAlignment =
                    Alignment.Center
            ) {

                Text(
                    "✓",

                    fontSize =
                        28.sp,

                    fontWeight =
                        FontWeight.Bold,

                    color =
                        Color.White
                )
            }


            Spacer(
                modifier =
                    Modifier.width(16.dp)
            )


            Column(
                modifier =
                    Modifier.weight(1f)
            ) {

                Text(
                    "READY",

                    fontSize =
                        25.sp,

                    fontWeight =
                        FontWeight.ExtraBold,

                    color =
                        DarkGreen
                )


                Text(
                    statusMessage,

                    fontSize =
                        13.sp,

                    color =
                        TextGrey
                )
            }


            Column(
                horizontalAlignment =
                    Alignment.End
            ) {

                Text(
                    "🛡",

                    fontSize =
                        30.sp
                )


                Text(
                    "Real-time Detection",

                    fontSize =
                        11.sp,

                    color =
                        NavyBlue
                )


                Text(
                    "Safer Conversations",

                    fontSize =
                        11.sp,

                    color =
                        NavyBlue
                )
            }
        }
    }
}


// ============================================================
// WAVEFORM
// ============================================================

@Composable
fun WaveformCard(
    isRecording: Boolean,
    amplitude: Float
) {

    val infiniteTransition =
        rememberInfiniteTransition(
            label =
                "wave"
        )


    val animation by
    infiniteTransition.animateFloat(

        initialValue =
            0f,

        targetValue =
            1f,

        animationSpec =
            infiniteRepeatable(

                animation =
                    tween(
                        durationMillis =
                            1400,

                        easing =
                            FastOutSlowInEasing
                    ),

                repeatMode =
                    RepeatMode.Reverse
            ),

        label =
            "waveAnimation"
    )


    Card(

        modifier =
            Modifier
                .fillMaxWidth()
                .height(250.dp),

        shape =
            RoundedCornerShape(22.dp),

        colors =
            CardDefaults.cardColors(
                containerColor =
                    Color.White
            )
    ) {

        Column(

            modifier =
                Modifier
                    .fillMaxSize()
                    .padding(16.dp),

            horizontalAlignment =
                Alignment.CenterHorizontally,

            verticalArrangement =
                Arrangement.Center
        ) {

            Row(

                modifier =
                    Modifier
                        .fillMaxWidth()
                        .height(120.dp),

                horizontalArrangement =
                    Arrangement.SpaceEvenly,

                verticalAlignment =
                    Alignment.CenterVertically
            ) {

                repeat(29) { index ->

                    val wave =
                        sin(
                            index * 0.7 +
                                animation * 4
                        )


                    val baseHeight =
                        18f +
                            abs(wave) * 35f


                    val extraHeight =
                        if (isRecording)
                            amplitude * 45f
                        else
                            0f


                    val barHeight =
                        baseHeight +
                            extraHeight


                    val barColor =
                        when {

                            index < 10 ->
                                BrightBlue

                            index < 20 ->
                                Green

                            else ->
                                Orange
                        }


                    Box(

                        modifier =
                            Modifier
                                .width(6.dp)
                                .height(
                                    barHeight.dp
                                )
                                .clip(
                                    RoundedCornerShape(
                                        10.dp
                                    )
                                )
                                .background(
                                    barColor
                                )
                    )
                }
            }


            Text(

                if (isRecording)
                    "LISTENING..."
                else
                    "READY TO LISTEN",

                fontSize =
                    21.sp,

                fontWeight =
                    FontWeight.ExtraBold,

                color =
                    NavyBlue
            )


            Spacer(
                modifier =
                    Modifier.height(5.dp)
            )


            Text(

                if (isRecording)
                    "SWARAKSHA is monitoring your voice"
                else
                    "Record or upload audio for analysis",

                fontSize =
                    13.sp,

                color =
                    TextGrey
            )
        }
    }
}


// ============================================================
// LIVE PROTECTION BUTTONS
// ============================================================

@Composable
fun LiveProtectionButtons(
    onStart: () -> Unit,
    onStop: () -> Unit
) {

    Row(

        modifier =
            Modifier.fillMaxWidth(),

        horizontalArrangement =
            Arrangement.spacedBy(12.dp)
    ) {

        Button(

            onClick =
                onStart,

            modifier =
                Modifier
                    .weight(1f)
                    .height(112.dp),

            shape =
                RoundedCornerShape(20.dp),

            colors =
                ButtonDefaults.buttonColors(
                    containerColor =
                        BrightBlue
                )
        ) {

            Column(
                horizontalAlignment =
                    Alignment.Start
            ) {

                Row(
                    verticalAlignment =
                        Alignment.CenterVertically
                ) {

                    Text(
                        "🛡",

                        fontSize =
                            24.sp
                    )


                    Spacer(
                        modifier =
                            Modifier.width(8.dp)
                    )


                    Text(
                        "START LIVE",

                        fontWeight =
                            FontWeight.ExtraBold,

                        fontSize =
                            15.sp
                    )
                }


                Spacer(
                    modifier =
                        Modifier.height(6.dp)
                )


                Text(
                    "Real-time AI monitoring",

                    fontSize =
                        11.sp
                )
            }
        }


        OutlinedButton(

            onClick =
                onStop,

            modifier =
                Modifier
                    .weight(1f)
                    .height(112.dp),

            shape =
                RoundedCornerShape(20.dp),

            border =
                BorderStroke(
                    2.dp,
                    Orange
                ),

            colors =
                ButtonDefaults.outlinedButtonColors(
                    contentColor =
                        DarkOrange
                )
        ) {

            Column(
                horizontalAlignment =
                    Alignment.Start
            ) {

                Row(
                    verticalAlignment =
                        Alignment.CenterVertically
                ) {

                    Text(
                        "■",

                        fontSize =
                            24.sp
                    )


                    Spacer(
                        modifier =
                            Modifier.width(8.dp)
                    )


                    Text(
                        "STOP LIVE",

                        fontWeight =
                            FontWeight.ExtraBold,

                        fontSize =
                            15.sp
                    )
                }


                Spacer(
                    modifier =
                        Modifier.height(6.dp)
                )


                Text(
                    "Stop background monitoring",

                    fontSize =
                        11.sp
                )
            }
        }
    }
}


// ============================================================
// ACTION CARD
// ============================================================

@Composable
fun ActionCard(

    title: String,

    subtitle: String,

    iconText: String,

    background: Color,

    borderColor: Color,

    iconBackground: Color,

    titleColor: Color,

    enabled: Boolean = true,

    onClick: () -> Unit
) {

    Card(

        modifier =
            Modifier
                .fillMaxWidth()
                .height(94.dp)
                .border(
                    width =
                        1.5.dp,

                    color =
                        borderColor,

                    shape =
                        RoundedCornerShape(20.dp)
                )
                .clickable(

                    enabled =
                        enabled,

                    onClick =
                        onClick
                ),

        shape =
            RoundedCornerShape(20.dp),

        colors =
            CardDefaults.cardColors(
                containerColor =
                    background
            )
    ) {

        Row(

            modifier =
                Modifier
                    .fillMaxSize()
                    .padding(
                        horizontal =
                            14.dp
                    ),

            verticalAlignment =
                Alignment.CenterVertically
        ) {

            Box(

                modifier =
                    Modifier
                        .size(58.dp)
                        .clip(CircleShape)
                        .background(
                            iconBackground
                        ),

                contentAlignment =
                    Alignment.Center
            ) {

                Text(

                    text =
                        iconText,

                    fontSize =
                        28.sp,

                    color =
                        Color.White,

                    fontWeight =
                        FontWeight.Bold
                )
            }


            Spacer(
                modifier =
                    Modifier.width(15.dp)
            )


            Column(

                modifier =
                    Modifier.weight(1f)
            ) {

                Text(

                    title,

                    fontSize =
                        16.sp,

                    fontWeight =
                        FontWeight.ExtraBold,

                    color =
                        titleColor
                )


                Spacer(
                    modifier =
                        Modifier.height(3.dp)
                )


                Text(

                    subtitle,

                    fontSize =
                        12.sp,

                    color =
                        TextGrey
                )
            }


            Text(

                "›",

                fontSize =
                    34.sp,

                fontWeight =
                    FontWeight.Bold,

                color =
                    titleColor
            )
        }
    }
}


// ============================================================
// ANALYZING CARD
// ============================================================

@Composable
fun AnalyzingCard() {

    Card(

        modifier =
            Modifier.fillMaxWidth(),

        shape =
            RoundedCornerShape(20.dp),

        colors =
            CardDefaults.cardColors(
                containerColor =
                    LightBlue
            )
    ) {

        Row(

            modifier =
                Modifier
                    .fillMaxWidth()
                    .padding(20.dp),

            verticalAlignment =
                Alignment.CenterVertically
        ) {

            CircularProgressIndicator(

                modifier =
                    Modifier.size(38.dp),

                color =
                    BrightBlue,

                strokeWidth =
                    4.dp
            )


            Spacer(
                modifier =
                    Modifier.width(16.dp)
            )


            Column {

                Text(

                    "ANALYZING VOICE",

                    fontWeight =
                        FontWeight.ExtraBold,

                    color =
                        NavyBlue
                )


                Text(

                    "Spectra-AASIST3 + Risk Engine",

                    fontSize =
                        12.sp,

                    color =
                        TextGrey
                )
            }
        }
    }
}


// ============================================================
// RESULT CARD
// ============================================================

@Composable
fun ResultCard(

    verdict: String,

    riskScore: Float,

    aiProbability: Float,

    realProbability: Float,

    scamRisk: Float,

    modelName: String,

    transcript: String,

    durationSeconds: Float,

    segmentCount: Int,

    reasons: List<String>,

    scamVerdict: String,

    securityDecision: String,

    securityActionType: String,

    securityRecommendation: String,

    speakerVerdict: String,

    speakerSimilarity: Float,

    impersonationStatus: String
) {

    val isHighRisk =
        verdict.contains(
            "HIGH",
            ignoreCase = true
        ) ||
            verdict.contains(
                "SPOOF",
                ignoreCase = true
            )


    val isSuspicious =
        verdict.contains(
            "SUSPICIOUS",
            ignoreCase = true
        )


    val resultColor =
        when {

            isHighRisk ->
                DangerRed

            isSuspicious ->
                Orange

            else ->
                Green
        }


    val resultBackground =
        when {

            isHighRisk ->
                DangerLight

            isSuspicious ->
                LightOrange

            else ->
                LightGreen
        }


    val securityColor =
        when (
            securityDecision.uppercase()
        ) {

            "BLOCK" ->
                DangerRed

            "VERIFY" ->
                Orange

            else ->
                Green
        }


    Card(

        modifier =
            Modifier.fillMaxWidth(),

        shape =
            RoundedCornerShape(24.dp),

        colors =
            CardDefaults.cardColors(
                containerColor =
                    resultBackground
            )
    ) {

        Column(

            modifier =
                Modifier.padding(20.dp)
        ) {

            // =================================================
            // RESULT HEADER
            // =================================================

            Row(
                verticalAlignment =
                    Alignment.CenterVertically
            ) {

                Box(

                    modifier =
                        Modifier
                            .size(56.dp)
                            .clip(CircleShape)
                            .background(
                                resultColor
                            ),

                    contentAlignment =
                        Alignment.Center
                ) {

                    Text(

                        if (isHighRisk)
                            "!"
                        else
                            "✓",

                        fontSize =
                            28.sp,

                        fontWeight =
                            FontWeight.ExtraBold,

                        color =
                            Color.White
                    )
                }


                Spacer(
                    modifier =
                        Modifier.width(14.dp)
                )


                Column {

                    Text(

                        "ANALYSIS RESULT",

                        fontSize =
                            12.sp,

                        fontWeight =
                            FontWeight.Bold,

                        color =
                            TextGrey
                    )


                    Text(

                        verdict,

                        fontSize =
                            23.sp,

                        fontWeight =
                            FontWeight.ExtraBold,

                        color =
                            resultColor
                    )
                }
            }


            Spacer(
                modifier =
                    Modifier.height(18.dp)
            )


            HorizontalDivider()


            Spacer(
                modifier =
                    Modifier.height(16.dp)
            )


            ResultMetric(

                label =
                    "Overall Risk",

                value =
                    "${fixed2(riskScore)}%",

                valueColor =
                    resultColor
            )


            Spacer(
                modifier =
                    Modifier.height(10.dp)
            )


            ResultMetric(

                label =
                    "AI Voice Probability",

                value =
                    "${fixed2(aiProbability)}%",

                valueColor =

                    when {

                        aiProbability >= 70f ->
                            DangerRed

                        aiProbability >= 40f ->
                            Orange

                        else ->
                            Green
                    }
            )


            Spacer(
                modifier =
                    Modifier.height(10.dp)
            )


            ResultMetric(

                label =
                    "Real Voice Probability",

                value =
                    "${fixed2(realProbability)}%",

                valueColor =

                    when {

                        realProbability >= 70f ->
                            Green

                        else ->
                            Orange
                    }
            )


            Spacer(
                modifier =
                    Modifier.height(10.dp)
            )


            ResultMetric(

                label =
                    "Scam Risk",

                value =
                    "${fixed2(scamRisk)}%",

                valueColor =

                    if (scamRisk >= 40f)
                        Orange
                    else
                        Green
            )


            Spacer(
                modifier =
                    Modifier.height(16.dp)
            )


            // =================================================
            // SECURITY ACTION
            // =================================================

            if (
                securityDecision.isNotBlank()
            ) {

                Card(

                    modifier =
                        Modifier.fillMaxWidth(),

                    shape =
                        RoundedCornerShape(16.dp),

                    colors =
                        CardDefaults.cardColors(
                            containerColor =
                                Color.White
                        ),

                    border =
                        BorderStroke(
                            1.5.dp,
                            securityColor
                        )
                ) {

                    Column(

                        modifier =
                            Modifier.padding(14.dp)
                    ) {

                        Text(

                            "SECURITY ACTION",

                            fontSize =
                                11.sp,

                            fontWeight =
                                FontWeight.ExtraBold,

                            color =
                                TextGrey
                        )


                        Spacer(
                            modifier =
                                Modifier.height(4.dp)
                        )


                        Text(

                            securityDecision,

                            fontSize =
                                22.sp,

                            fontWeight =
                                FontWeight.ExtraBold,

                            color =
                                securityColor
                        )


                        if (
                            securityActionType.isNotBlank()
                        ) {

                            Spacer(
                                modifier =
                                    Modifier.height(3.dp)
                            )


                            Text(

                                securityActionType,

                                fontSize =
                                    12.sp,

                                fontWeight =
                                    FontWeight.Bold,

                                color =
                                    NavyBlue
                            )
                        }


                        if (
                            securityRecommendation.isNotBlank()
                        ) {

                            Spacer(
                                modifier =
                                    Modifier.height(7.dp)
                            )


                            Text(

                                securityRecommendation,

                                fontSize =
                                    12.sp,

                                color =
                                    TextGrey
                            )
                        }
                    }
                }
            }


            // =================================================
            // SPEAKER VERIFICATION
            // =================================================

            if (
                speakerVerdict.isNotBlank() ||
                impersonationStatus.isNotBlank()
            ) {

                Spacer(
                    modifier =
                        Modifier.height(16.dp)
                )


                Text(

                    "SPEAKER VERIFICATION",

                    fontSize =
                        12.sp,

                    fontWeight =
                        FontWeight.ExtraBold,

                    color =
                        NavyBlue
                )


                Spacer(
                    modifier =
                        Modifier.height(8.dp)
                )


                if (
                    speakerVerdict.isNotBlank()
                ) {

                    ResultMetric(

                        label =
                            "Speaker Result",

                        value =
                            speakerVerdict,

                        valueColor =

                            if (
                                speakerVerdict.contains(
                                    "SAME",
                                    ignoreCase = true
                                )
                            ) {

                                Green

                            } else {

                                DangerRed
                            }
                    )
                }


                if (
                    speakerSimilarity > 0f
                ) {

                    Spacer(
                        modifier =
                            Modifier.height(7.dp)
                    )


                    ResultMetric(

                        label =
                            "Similarity Indicator",

                        value =
                            "${fixed2(speakerSimilarity)}%",

                        valueColor =
                            NavyBlue
                    )
                }


                if (
                    impersonationStatus.isNotBlank()
                ) {

                    Spacer(
                        modifier =
                            Modifier.height(7.dp)
                    )


                    Text(

                        impersonationStatus,

                        fontSize =
                            12.sp,

                        fontWeight =
                            FontWeight.Bold,

                        color =

                            if (
                                impersonationStatus.contains(
                                    "SPOOF",
                                    ignoreCase = true
                                )
                            ) {

                                DangerRed

                            } else {

                                TextDark
                            }
                    )
                }
            }


            Spacer(
                modifier =
                    Modifier.height(16.dp)
            )


            // =================================================
            // MODEL
            // =================================================

            Text(

                "MODEL",

                fontSize =
                    11.sp,

                fontWeight =
                    FontWeight.Bold,

                color =
                    TextGrey
            )


            Text(

                modelName,

                fontSize =
                    14.sp,

                fontWeight =
                    FontWeight.Bold,

                color =
                    NavyBlue
            )


            Spacer(
                modifier =
                    Modifier.height(14.dp)
            )


            // =================================================
            // AUDIO INFO
            // =================================================

            Row(

                modifier =
                    Modifier.fillMaxWidth(),

                horizontalArrangement =
                    Arrangement.SpaceBetween
            ) {

                Text(

                    "Duration: ${
                        fixed2(
                            durationSeconds
                        )
                    } sec",

                    fontSize =
                        12.sp,

                    color =
                        TextGrey
                )


                Text(

                    "Segments: $segmentCount",

                    fontSize =
                        12.sp,

                    color =
                        TextGrey
                )
            }


            // =================================================
            // REASONS
            // =================================================

            if (reasons.isNotEmpty()) {

                Spacer(
                    modifier =
                        Modifier.height(18.dp)
                )


                Text(

                    "DETECTION REASONS",

                    fontSize =
                        12.sp,

                    fontWeight =
                        FontWeight.ExtraBold,

                    color =
                        NavyBlue
                )


                Spacer(
                    modifier =
                        Modifier.height(8.dp)
                )


                reasons.forEach { reason ->

                    Row(

                        modifier =
                            Modifier.padding(
                                vertical =
                                    3.dp
                            ),

                        verticalAlignment =
                            Alignment.Top
                    ) {

                        Text(

                            "•",

                            color =
                                Orange,

                            fontWeight =
                                FontWeight.Bold
                        )


                        Spacer(
                            modifier =
                                Modifier.width(7.dp)
                        )


                        Text(

                            reason,

                            fontSize =
                                12.sp,

                            color =
                                TextGrey
                        )
                    }
                }
            }


            // =================================================
            // TRANSCRIPT
            // =================================================

            if (transcript.isNotBlank()) {

                Spacer(
                    modifier =
                        Modifier.height(18.dp)
                )


                Text(

                    "TRANSCRIPT",

                    fontSize =
                        12.sp,

                    fontWeight =
                        FontWeight.ExtraBold,

                    color =
                        NavyBlue
                )


                Spacer(
                    modifier =
                        Modifier.height(6.dp)
                )


                Card(

                    shape =
                        RoundedCornerShape(14.dp),

                    colors =
                        CardDefaults.cardColors(
                            containerColor =
                                Color.White
                        )
                ) {

                    Text(

                        transcript,

                        modifier =
                            Modifier.padding(12.dp),

                        fontSize =
                            12.sp,

                        color =
                            TextGrey
                    )
                }
            }


            // =================================================
            // SCAM VERDICT
            // =================================================

            if (scamVerdict.isNotBlank()) {

                Spacer(
                    modifier =
                        Modifier.height(16.dp)
                )


                Text(

                    "SCAM ANALYSIS",

                    fontSize =
                        12.sp,

                    fontWeight =
                        FontWeight.ExtraBold,

                    color =
                        NavyBlue
                )


                Text(

                    scamVerdict,

                    fontSize =
                        13.sp,

                    fontWeight =
                        FontWeight.Bold,

                    color =
                        TextDark
                )
            }
        }
    }
}


// ============================================================
// RESULT METRIC
// ============================================================

@Composable
fun ResultMetric(

    label: String,

    value: String,

    valueColor: Color
) {

    Row(

        modifier =
            Modifier.fillMaxWidth(),

        horizontalArrangement =
            Arrangement.SpaceBetween,

        verticalAlignment =
            Alignment.CenterVertically
    ) {

        Text(

            label,

            fontSize =
                13.sp,

            color =
                TextGrey
        )


        Text(

            value,

            fontSize =
                17.sp,

            fontWeight =
                FontWeight.ExtraBold,

            color =
                valueColor
        )
    }
}


// ============================================================
// JSON ARRAY
// ============================================================

fun jsonArrayToList(
    array: JsonArray?
): List<String> {

    if (array == null) {

        return emptyList()
    }


    val result =
        mutableListOf<String>()


    for (element in array) {

        try {

            result.add(
                element.asString
            )

        } catch (_: Exception) {
        }
    }


    return result
}


// ============================================================
// FILE NAME
// ============================================================

fun getFileName(

    context: Context,

    uri: Uri

): String {

    var result: String? =
        null


    context.contentResolver
        .query(
            uri,
            null,
            null,
            null,
            null
        )
        ?.use { cursor ->

            val nameIndex =
                cursor.getColumnIndex(
                    OpenableColumns.DISPLAY_NAME
                )


            if (
                nameIndex >= 0 &&
                cursor.moveToFirst()
            ) {

                result =
                    cursor.getString(
                        nameIndex
                    )
            }
        }


    return result
        ?: "selected_audio.wav"
}


// ============================================================
// TIME FORMAT
// ============================================================

fun formatTime(
    seconds: Int
): String {

    val minutes =
        seconds / 60


    val remaining =
        seconds % 60


    return String.format(
        Locale.US,
        "%02d:%02d",
        minutes,
        remaining
    )
}


// ============================================================
// FLOAT FORMAT
// ============================================================

fun fixed2(
    value: Float
): String {

    return String.format(
        Locale.US,
        "%.2f",
        value
    )
}