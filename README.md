# SWARAKSHA – AI-Powered Real-Time Voice Security

<p align="center">
  <b>AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks</b>
</p>

<p align="center">
  Smart India Hackathon 2026 · Problem Statement ID: SIH26104
</p>

---

## Overview

**SWARAKSHA** is an AI-powered real-time voice security system designed to detect and respond to voice-cloning and synthetic-speech impersonation attacks during voice conversations.

Instead of treating voice security as a simple binary "real or fake" classification, SWARAKSHA combines multiple security layers:

- Synthetic voice detection
- Speaker verification
- Scam and social-engineering analysis
- Dynamic risk assessment
- Real-time security alerts
- Action-aware security decisions
- Adaptive verification for suspicious interactions

The prototype uses a controlled **WebRTC-based voice calling environment** so that the live audio stream can be accessed and analyzed with user consent.

---

## Problem Statement

### SIH26104

**AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks**

Modern voice-cloning technologies can generate highly realistic speech that may be used for impersonation, social engineering, financial fraud, and unauthorized disclosure of sensitive information.

Traditional voice communication systems generally do not provide an integrated AI security layer capable of analyzing live speech, evaluating impersonation risk, and triggering appropriate verification before sensitive actions are performed.

SWARAKSHA addresses this problem through a real-time, API-driven voice security architecture.

---

## Key Idea

SWARAKSHA follows four security questions:

```text
1. WHO is speaking?
        ↓
   Speaker Verification

2. HOW is the voice being delivered?
        ↓
   Synthetic / Replay Analysis

3. CAN the caller prove their identity?
        ↓
   Adaptive Verification

4. WHAT should happen next?
        ↓
   Action-Aware Security Gate
