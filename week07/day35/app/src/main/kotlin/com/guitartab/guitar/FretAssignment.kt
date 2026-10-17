package com.guitartab.guitar

import com.guitartab.model.NoteEvent
import kotlin.math.abs

/**
 * Раскладка нот по струнам/ладам в стандартном строе.
 * Витерби по колонкам (аккорд = колонка) с минимизацией движения левой руки.
 */
object FretAssignment {
    private const val STRING_WEIGHT = 2.0
    private const val UNMATCHED_PENALTY = 2.0
    private const val MAX_STATES = 2000

    class Column(val notes: List<NoteEvent>)

    /** Для каждой колонки возвращает IntArray: на позиции i (нота по возрастанию высоты) — индекс струны. */
    fun assign(columns: List<Column>): List<IntArray> {
        if (columns.isEmpty()) return emptyList()
        val statesPerCol = columns.map { buildStates(it) }
        if (statesPerCol.any { it.isEmpty() }) {
            return greedyAssign(columns)
        }

        val n = columns.size
        val dp = Array(n) { DoubleArray(statesPerCol[it].size) { Double.POSITIVE_INFINITY } }
        val back = Array(n) { IntArray(statesPerCol[it].size) { -1 } }

        for (j in statesPerCol[0].indices) {
            dp[0][j] = withinCost(columns[0], statesPerCol[0][j])
        }
        for (i in 1 until n) {
            for (j in statesPerCol[i].indices) {
                var bestCost = Double.POSITIVE_INFINITY
                var bestPrev = -1
                for (p in statesPerCol[i - 1].indices) {
                    val cost = dp[i - 1][p] +
                        transitionCost(columns[i - 1], statesPerCol[i - 1][p], columns[i], statesPerCol[i][j]) +
                        withinCost(columns[i], statesPerCol[i][j])
                    if (cost < bestCost) {
                        bestCost = cost
                        bestPrev = p
                    }
                }
                dp[i][j] = bestCost
                back[i][j] = bestPrev
            }
        }

        val chosen = IntArray(n)
        var best = 0
        for (j in 1 until dp[n - 1].size) {
            if (dp[n - 1][j] < dp[n - 1][best]) best = j
        }
        chosen[n - 1] = best
        for (i in n - 1 downTo 1) {
            best = back[i][best]
            chosen[i - 1] = best
        }
        return columns.indices.map { i -> statesPerCol[i][chosen[i]] }
    }

    /** Все инъективные назначения струн нотам колонки (ноты отсортированы по высоте). */
    private fun buildStates(col: Column): List<IntArray> {
        val k = col.notes.size
        if (k > 6) return emptyList()
        val result = mutableListOf<IntArray>()

        fun rec(i: Int, used: BooleanArray, cur: IntArray) {
            if (result.size >= MAX_STATES) return
            if (i == k) {
                result.add(cur.copyOf())
                return
            }
            val pitch = col.notes[i].pitch
            for (s in 0 until 6) {
                if (used[s]) continue
                val fret = pitch - GuitarTuning.strings[s].openMidi
                if (fret in 0..GuitarTuning.MAX_FRET) {
                    used[s] = true
                    cur[i] = s
                    rec(i + 1, used, cur)
                    used[s] = false
                }
            }
        }
        rec(0, BooleanArray(6), IntArray(k))
        return result
    }

    private fun withinCost(col: Column, strings: IntArray): Double {
        var cost = 0.0
        for (i in col.notes.indices) {
            val fret = col.notes[i].pitch - GuitarTuning.strings[strings[i]].openMidi
            cost += fret * 0.01 // лёгкое предпочтение низких позиций
        }
        return cost
    }

    private fun transitionCost(c1: Column, s1: IntArray, c2: Column, s2: IntArray): Double {
        val m = minOf(c1.notes.size, c2.notes.size)
        var cost = 0.0
        for (i in 0 until m) {
            val f1 = c1.notes[i].pitch - GuitarTuning.strings[s1[i]].openMidi
            val f2 = c2.notes[i].pitch - GuitarTuning.strings[s2[i]].openMidi
            cost += abs(f1 - f2) + STRING_WEIGHT * abs(s1[i] - s2[i])
        }
        cost += UNMATCHED_PENALTY * abs(c1.notes.size - c2.notes.size)
        return cost
    }

    /** Запасной жадный вариант, если у какой-то ноты нет валидной позиции. */
    private fun greedyAssign(columns: List<Column>): List<IntArray> {
        var prevAvgFret = 0.0
        return columns.map { col ->
            val used = BooleanArray(6)
            val assignment = IntArray(col.notes.size) { -1 }
            for (i in col.notes.indices) {
                val pitch = col.notes[i].pitch
                val cands = GuitarTuning.candidates(pitch).filter { !used[it.first] }
                val best = cands.minByOrNull { abs(it.second - prevAvgFret) }
                if (best != null) {
                    used[best.first] = true
                    assignment[i] = best.first
                }
            }
            val frets = col.notes.indices.mapNotNull { i ->
                if (assignment[i] >= 0) col.notes[i].pitch - GuitarTuning.strings[assignment[i]].openMidi else null
            }
            if (frets.isNotEmpty()) prevAvgFret = frets.average()
            assignment
        }
    }
}
