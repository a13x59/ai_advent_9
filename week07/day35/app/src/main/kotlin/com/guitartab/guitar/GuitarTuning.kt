package com.guitartab.guitar

data class StringSpec(val name: String, val openMidi: Int)

/** Стандартный строй шестиструнной гитары EADGBE. Струны перечислены от 1-й (самой тонкой) к 6-й. */
object GuitarTuning {
    val strings: List<StringSpec> = listOf(
        StringSpec("e", 64), // E4 — 1-я
        StringSpec("B", 59), // B3 — 2-я
        StringSpec("G", 55), // G3 — 3-я
        StringSpec("D", 50), // D3 — 4-я
        StringSpec("A", 45), // A2 — 5-я
        StringSpec("E", 40), // E2 — 6-я
    )

    const val MAX_FRET = 24

    /** Пары (индекс струны, лад), на которых нота играбельна в стандартном строе. */
    fun candidates(pitch: Int): List<Pair<Int, Int>> =
        strings.indices.mapNotNull { s ->
            val fret = pitch - strings[s].openMidi
            if (fret in 0..MAX_FRET) s to fret else null
        }
}
