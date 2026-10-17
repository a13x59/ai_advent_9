package com.guitartab

import com.guitartab.audio.AudioFormat
import com.guitartab.audio.FfmpegConverter
import com.guitartab.guitar.FretAssignment
import com.guitartab.guitar.TabRenderer
import com.guitartab.midi.MidiParser
import com.guitartab.model.Meta
import com.guitartab.model.NoteEvent
import com.guitartab.model.TranscriptionResponse
import com.guitartab.quality.QualityGate
import com.guitartab.worker.BasicPitchClient
import java.util.Base64

/** Оркестрирует полный конвейер: аудио -> MIDI -> раскладка -> таб -> оценка качества. */
class TranscriptionService(
    private val ffmpeg: FfmpegConverter,
    private val worker: BasicPitchClient,
    private val qualityGate: QualityGate,
) {
    suspend fun transcribe(input: ByteArray, format: AudioFormat): TranscriptionResponse {
        val wav = ffmpeg.toWav(input, format)
        val midi = worker.predict(wav)
        val notes = MidiParser.parse(midi)

        val columns = groupColumns(notes)
        val assignments = FretAssignment.assign(columns)
        val tab = TabRenderer.render(columns, assignments)

        val duration = notes.maxOfOrNull { it.endSec } ?: 0.0
        val meta = Meta(durationSec = round1(duration), noteCount = notes.size, tuning = "EADGBE")
        val (quality, reason) = qualityGate.evaluate(tab, notes, meta)

        return TranscriptionResponse(
            tab = tab,
            midiBase64 = Base64.getEncoder().encodeToString(midi),
            quality = quality,
            qualityReason = reason,
            meta = meta,
        )
    }

    /** Группирует ноты в колонки (одновременно звучащие — в одну колонку). */
    private fun groupColumns(notes: List<NoteEvent>, epsilonSec: Double = 0.05): List<FretAssignment.Column> {
        val sorted = notes.sortedWith(compareBy({ it.startSec }, { it.pitch }))
        val columns = mutableListOf<MutableList<NoteEvent>>()
        for (n in sorted) {
            val last = columns.lastOrNull()
            if (last != null && (n.startSec - last.first().startSec) <= epsilonSec) {
                last.add(n)
            } else {
                columns.add(mutableListOf(n))
            }
        }
        return columns.map { FretAssignment.Column(it.sortedBy { n -> n.pitch }) }
    }

    private fun round1(v: Double): Double = Math.round(v * 10.0) / 10.0
}
