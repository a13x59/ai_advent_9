package com.guitartab.quality

import com.guitartab.guitar.GuitarTuning
import com.guitartab.model.NoteEvent

/** Дешёвые детерминированные проверки результата (до вызова LLM). */
object DeterministicChecks {

    data class Result(val ok: Boolean, val reason: String?)

    fun evaluate(notes: List<NoteEvent>, tab: String): Result {
        if (notes.isEmpty()) return Result(false, "Не найдено ни одной ноты в аудио")
        if (tab.isBlank()) return Result(false, "Не удалось построить табулатуру")
        val outOfRange = notes.count { GuitarTuning.candidates(it.pitch).isEmpty() }
        if (outOfRange > 0) return Result(false, "$outOfRange нот(ы) вне диапазона гитары в стандартном строе")
        return Result(true, null)
    }
}
