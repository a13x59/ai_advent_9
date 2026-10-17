package com.guitartab.quality

import com.guitartab.model.Meta
import com.guitartab.model.NoteEvent
import kotlinx.coroutines.withTimeoutOrNull
import org.slf4j.LoggerFactory

/**
 * Гейт качества: детерминированные проверки + опциональная оценка через DeepSeek.
 * При сбое LLM — fail-open (пропускаем по детерминированным проверкам).
 */
class QualityGate(
    private val deepSeek: DeepSeekClient?,
    private val llmEnabled: Boolean,
) {
    private val logger = LoggerFactory.getLogger(QualityGate::class.java)

    suspend fun evaluate(tab: String, notes: List<NoteEvent>, meta: Meta): Pair<String, String?> {
        val det = DeterministicChecks.evaluate(notes, tab)
        if (!det.ok) {
            return "warning" to det.reason
        }
        if (!llmEnabled || deepSeek == null) {
            return "ok" to null
        }

        var verdict: Verdict? = null
        try {
            verdict = withTimeoutOrNull(15_000) { deepSeek.evaluate(tab, meta) }
            if (verdict == null) {
                verdict = withTimeoutOrNull(15_000) { deepSeek.evaluate(tab, meta) }
            }
        } catch (e: Exception) {
            logger.warn("DeepSeek gate failed, fail-open: {}", e.message)
        }

        return when {
            verdict == null -> "ok" to null
            verdict.ok -> "ok" to null
            else -> "warning" to verdict.reason
        }
    }
}
