import re


class ScamDetector:

    def __init__(self):

        self.high_risk_patterns = {

            "emergency": [
                "accident",
                "kidnapped",
                "kidnap",
                "hospital",
                "police",
                "arrest",
                "emergency",
                "danger",
                "help me",
            ],

            "money": [
                "send money",
                "transfer money",
                "pay money",
                "payment",
                "upi",
                "qr code",
                "bank account",
                "account number",
                "otp",
                "pin",
                "refund",
            ],

            "urgency": [
                "immediately",
                "right now",
                "urgent",
                "quickly",
                "within one hour",
                "don't tell anyone",
                "do not tell anyone",
            ],

            "impersonation": [
                "i am your son",
                "i am your daughter",
                "i am your friend",
                "i am your brother",
                "i am your sister",
                "this is your son",
                "this is your daughter",
                "police officer",
                "bank officer",
            ],
        }

    def analyze(self, text):

        text = text.lower().strip()

        detected = {}
        total_matches = 0

        for category, patterns in self.high_risk_patterns.items():

            matches = []

            for pattern in patterns:

                if pattern in text:
                    matches.append(pattern)

            if matches:

                detected[category] = matches
                total_matches += len(matches)

        # Simple risk calculation
        risk_score = min(
            100,
            total_matches * 12
        )

        # Extra combination boost
        if (
            "money" in detected
            and "urgency" in detected
        ):
            risk_score += 20

        if (
            "emergency" in detected
            and "money" in detected
        ):
            risk_score += 20

        if (
            "impersonation" in detected
            and "money" in detected
        ):
            risk_score += 20

        risk_score = min(
            100,
            risk_score
        )

        if risk_score >= 70:
            verdict = "HIGH RISK"

        elif risk_score >= 40:
            verdict = "SUSPICIOUS"

        else:
            verdict = "LOW RISK"

        return {

            "text": text,

            "scam_risk_score": risk_score,

            "verdict": verdict,

            "detected_categories": detected,

            "total_matches": total_matches,
        }