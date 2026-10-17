package com.guitartab.model

import kotlinx.serialization.Serializable

@Serializable
data class NoteEvent(
    val pitch: Int,
    val startSec: Double,
    val endSec: Double,
    val velocity: Int,
)

@Serializable
data class Meta(
    val durationSec: Double,
    val noteCount: Int,
    val tuning: String,
)

@Serializable
data class TranscriptionResponse(
    val tab: String,
    val midiBase64: String,
    val quality: String,
    val qualityReason: String? = null,
    val meta: Meta,
)

@Serializable
data class ErrorResponse(val error: String)
