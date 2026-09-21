class RiskEngine:

    def __init__(self):
        print("VIGILVOICE Risk Engine initialized")

    def analyze(
        self,
        voice_result,
        scam_result=None,
        action_context=None
    ):

        # ====================================================
        # AASIST3
        # ====================================================

        spoof_score = float(
            voice_result.get(
                "spoof_probability",
                0.0
            )
        )

        spoof_score = max(
            0.0,
            min(1.0, spoof_score)
        )

        # ====================================================
        # CALIBRATED VOICE VERDICT
        # ====================================================

        if spoof_score >= 0.70:

            voice_verdict = "AI SPOOF"

        elif spoof_score >= 0.40:

            voice_verdict = "SUSPICIOUS"

        else:

            voice_verdict = "REAL"

        # ====================================================
        # SCAM SCORE
        # ====================================================

        scam_score = 0.0

        if scam_result:

            scam_score = float(
                scam_result.get(
                    "scam_risk_score",
                    0.0
                )
            )

        scam_score = max(
            0.0,
            min(100.0, scam_score)
        )

        # ====================================================
        # FINAL RISK
        # ====================================================

        voice_risk = (
            spoof_score * 100.0
        )

        combined_risk = (
            voice_risk * 0.60
            +
            scam_score * 0.40
        )

        combined_risk = max(
            0.0,
            min(100.0, combined_risk)
        )

        # ====================================================
        # IMPORTANT SAFETY RULE
        #
        # Strong AI spoof evidence must not be downgraded
        # simply because the transcript contains no scam words.
        # ====================================================

        if spoof_score >= 0.70:

            final_risk = max(
                combined_risk,
                voice_risk
            )

            final_verdict = "HIGH RISK"

        elif (
            spoof_score >= 0.40
            or
            scam_score >= 40
        ):

            final_risk = combined_risk

            final_verdict = "SUSPICIOUS"

        else:

            final_risk = combined_risk

            final_verdict = "LOW RISK"

        # ====================================================
        # CONFIDENCE
        # ====================================================

        if spoof_score >= 0.70:

            confidence = "HIGH"

        elif spoof_score >= 0.40:

            confidence = "MEDIUM"

        elif scam_score >= 70:

            confidence = "HIGH"

        else:

            confidence = "HIGH"

        # ====================================================
        # REASONS
        # ====================================================

        reasons = []

        if voice_verdict == "AI SPOOF":

            reasons.append(
                "Strong synthetic voice characteristics detected."
            )

            reasons.append(
                f"AASIST3 spoof score is "
                f"{spoof_score * 100:.1f}%."
            )

        elif voice_verdict == "SUSPICIOUS":

            reasons.append(
                "Voice contains characteristics requiring verification."
            )

        else:

            reasons.append(
                "Voice characteristics are consistent with genuine speech."
            )

        # ====================================================
        # SCAM EVIDENCE
        # ====================================================

        if scam_result:

            categories = scam_result.get(
                "detected_categories",
                []
            )

            if categories:

                if isinstance(
                    categories,
                    dict
                ):

                    category_names = list(
                        categories.keys()
                    )

                else:

                    category_names = categories

                reasons.append(
                    "Potential scam indicators detected: "
                    +
                    ", ".join(
                        category_names
                    )
                )

        if scam_score >= 70:

            reasons.append(
                "Conversation contains strong scam indicators."
            )

        elif scam_score >= 40:

            reasons.append(
                "Conversation contains suspicious scam indicators."
            )

        # ====================================================
        # ACTION-AWARE SECURITY GATE
        # ====================================================

        if action_context is None:
            action_context = {}

        action_type = str(
            action_context.get(
                "action_type",
                "NORMAL_CONVERSATION"
            )
        ).upper()

        transaction_value = float(
            action_context.get(
                "transaction_value",
                0.0
            )
        )

        urgency = bool(
            action_context.get(
                "urgency",
                False
            )
        )

        # Sensitive actions require stronger verification.

        sensitive_actions = {
            "FUND_TRANSFER",
            "PAYMENT",
            "OTP_REQUEST",
            "PASSWORD_RESET",
            "CREDENTIAL_REQUEST",
            "ACCOUNT_ACCESS",
            "PRIVILEGED_ACCESS",
            "CONFIDENTIAL_DISCLOSURE"
        }

        is_sensitive_action = (
            action_type in sensitive_actions
        )

        # ====================================================
        # ACTION DECISION
        # ====================================================

        if spoof_score >= 0.70:

            action_decision = "BLOCK"

            action_reason = (
                "AI-generated voice detected. "
                "High-risk action is blocked pending verification."
            )

        elif (
            final_risk >= 70
            and is_sensitive_action
        ):

            action_decision = "BLOCK"

            action_reason = (
                "High risk detected during a sensitive action."
            )

        elif (
            final_risk >= 40
            or
            is_sensitive_action
            or
            urgency
            or
            transaction_value > 0
        ):

            action_decision = "VERIFY"

            action_reason = (
                "Additional caller verification is required "
                "before proceeding."
            )

        else:

            action_decision = "ALLOW"

            action_reason = (
                "No elevated security condition detected."
            )

        # ====================================================
        # ACTION RECOMMENDATION
        # ====================================================

        if action_decision == "BLOCK":

            recommendation = (
                "Do not approve the requested action. "
                "Initiate independent verification."
            )

        elif action_decision == "VERIFY":

            recommendation = (
                "Verify the caller using an independent "
                "trusted channel or active challenge."
            )

        else:

            recommendation = (
                "Conversation may continue under normal protection."
            )

        # ====================================================
        # RETURN
        # ====================================================

        return {

            "voice_verdict":
                voice_verdict,

            "voice_risk_score":
                round(
                    voice_risk,
                    2
                ),

            "final_verdict":
                final_verdict,

            "risk_score":
                round(
                    final_risk,
                    2
                ),

            "confidence":
                confidence,

            "reasons":
                reasons,

            # =================================================
            # ACTION-AWARE SECURITY
            # =================================================

            "security_action": {

                "decision":
                    action_decision,

                "action_type":
                    action_type,

                "sensitive_action":
                    is_sensitive_action,

                "transaction_value":
                    transaction_value,

                "urgency":
                    urgency,

                "reason":
                    action_reason,

                "recommendation":
                    recommendation

            },

            "voice": {

                "spoof_probability":
                    round(
                        spoof_score,
                        4
                    ),

                "ai_probability":
                    round(
                        spoof_score * 100,
                        2
                    ),

                "real_probability":
                    round(
                        (1.0 - spoof_score) * 100,
                        2
                    ),

                "model":
                    "Spectra-AASIST3"

            },

            "scam": {

                "risk_score":
                    round(
                        scam_score,
                        2
                    )

            },

            "calibration": {

                "real_below":
                    0.40,

                "suspicious_from":
                    0.40,

                "spoof_from":
                    0.70,

                "source":
                    "ASVspoof2019 LA Dev balanced calibration subset"

            }

        }