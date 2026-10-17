package com.guitartab.guitar

import com.guitartab.guitar.FretAssignment.Column

/** Рендерит ASCII-табулатуру из колонок и назначенных струн. */
object TabRenderer {

    fun render(columns: List<Column>, assignments: List<IntArray>): String {
        val n = GuitarTuning.strings.size
        val sb = StringBuilder()
        for (s in 0 until n) {
            sb.append(GuitarTuning.strings[s].name).append('|')
            for (ci in columns.indices) {
                if (ci > 0) sb.append('-')
                val fret = fretAt(columns[ci], assignments[ci], s)
                sb.append(if (fret >= 0) fret.toString().padStart(2) else "--")
            }
            sb.append('|').append('\n')
        }
        return sb.toString().trimEnd('\n')
    }

    private fun fretAt(col: Column, assignment: IntArray, stringIndex: Int): Int {
        for (i in col.notes.indices) {
            if (assignment[i] == stringIndex) {
                return col.notes[i].pitch - GuitarTuning.strings[stringIndex].openMidi
            }
        }
        return -1
    }
}
