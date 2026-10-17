package com.guitartab.midi

import com.guitartab.model.NoteEvent
import java.io.ByteArrayInputStream
import javax.sound.midi.MetaMessage
import javax.sound.midi.MidiSystem
import javax.sound.midi.Sequence
import javax.sound.midi.ShortMessage

/** Парсит MIDI (из Basic Pitch) в список нот с временем в секундах. */
object MidiParser {

    private data class RawNote(val pitch: Int, val startTick: Long, val endTick: Long, val velocity: Int)

    fun parse(bytes: ByteArray): List<NoteEvent> {
        val sequence: Sequence = MidiSystem.getSequence(ByteArrayInputStream(bytes))
        val division = sequence.resolution
        if (division <= 0) return emptyList() // SMPTE-тайминг в MVP не поддерживаем

        val tempoMap = mutableListOf(0L to 500_000) // tick -> µs на четверть (default 120 BPM)
        val active = mutableMapOf<Int, RawNote>()
        val notes = mutableListOf<RawNote>()

        for (track in sequence.tracks) {
            for (i in 0 until track.size()) {
                val ev = track[i]
                when (val msg = ev.message) {
                    is MetaMessage -> {
                        if (msg.type == 0x51 && msg.data.size >= 3) {
                            val mspq = ((msg.data[0].toInt() and 0xFF) shl 16) or
                                ((msg.data[1].toInt() and 0xFF) shl 8) or
                                (msg.data[2].toInt() and 0xFF)
                            if (mspq > 0) tempoMap.add(ev.tick to mspq)
                        }
                    }
                    is ShortMessage -> {
                        val pitch = msg.data1
                        val key = (msg.channel shl 8) or pitch
                        when (msg.command) {
                            ShortMessage.NOTE_ON -> {
                                if (msg.data2 > 0) {
                                    active[key] = RawNote(pitch, ev.tick, ev.tick, msg.data2)
                                } else {
                                    active.remove(key)?.let { notes.add(it.copy(endTick = ev.tick)) }
                                }
                            }
                            ShortMessage.NOTE_OFF -> {
                                active.remove(key)?.let { notes.add(it.copy(endTick = ev.tick)) }
                            }
                        }
                    }
                }
            }
        }

        val tickToSeconds = buildTickToSeconds(division, tempoMap)
        return notes
            .filter { it.endTick > it.startTick }
            .sortedBy { it.startTick }
            .map { n ->
                NoteEvent(
                    pitch = n.pitch,
                    startSec = tickToSeconds(n.startTick),
                    endSec = tickToSeconds(n.endTick),
                    velocity = n.velocity,
                )
            }
    }

    private fun buildTickToSeconds(division: Int, tempoEvents: List<Pair<Long, Int>>): (Long) -> Double {
        val sorted = tempoEvents.sortedBy { it.first }
        return { tick ->
            var seconds = 0.0
            var cursor = 0L
            var mspq = 500_000
            for ((t, m) in sorted) {
                if (t > tick) break
                if (t > cursor) {
                    seconds += (t - cursor) * mspq / 1_000_000.0 / division
                    cursor = t
                }
                mspq = m
            }
            seconds += (tick - cursor) * mspq / 1_000_000.0 / division
            seconds
        }
    }
}
