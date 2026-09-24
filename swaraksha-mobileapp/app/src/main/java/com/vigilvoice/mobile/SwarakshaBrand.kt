package com.vigilvoice.mobile

import androidx.compose.foundation.layout.Row
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.sp
import androidx.compose.material3.Text

/**
 * SWARAKSHA official wordmark.
 *
 * SWA    -> Saffron
 * RAKSHA -> Green
 *
 * No separate logo/icon is used.
 */
@Composable
fun SwarakshaWordmark(
    fontSize: TextUnit = 30.sp
) {
    Row(
        verticalAlignment = Alignment.CenterVertically
    ) {

        Text(
            text = "SWA",
            fontSize = fontSize,
            fontWeight = FontWeight.ExtraBold,
            color = Color(0xFFFF9933),
            letterSpacing = 0.2.sp
        )

        Text(
            text = "RAKSHA",
            fontSize = fontSize,
            fontWeight = FontWeight.ExtraBold,
            color = Color(0xFF138808),
            letterSpacing = 0.2.sp
        )
    }
}