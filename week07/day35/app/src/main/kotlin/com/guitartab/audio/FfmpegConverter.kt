package com.guitartab.audio

import com.guitartab.TranscriptionException
import java.nio.file.Files
import java.util.concurrent.CompletableFuture

/** Конвертирует входной аудиофайл в WAV через ffmpeg (используется как есть). */
class FfmpegConverter(private val ffmpegPath: String = "ffmpeg") {

    fun toWav(input: ByteArray, format: AudioFormat): ByteArray {
        if (format == AudioFormat.WAV) {
            return input
        }
        val tmp = Files.createTempFile("input", "." + format.ext)
        try {
            tmp.toFile().writeBytes(input)
            val process = ProcessBuilder(
                ffmpegPath, "-y", "-i", tmp.toString(),
                "-f", "wav", "-ac", "1", "-ar", "22050", "pipe:1",
            ).start()

            val stderr = CompletableFuture.supplyAsync { process.errorStream.readBytes() }
            val out = process.inputStream.readBytes()
            val exit = process.waitFor()
            val err = stderr.get()

            if (exit != 0) {
                throw TranscriptionException(400, "Не удалось декодировать аудио: " + String(err).take(200))
            }
            return out
        } catch (e: TranscriptionException) {
            throw e
        } catch (e: Exception) {
            throw TranscriptionException(500, "Ошибка ffmpeg: ${e.message}")
        } finally {
            try {
                Files.deleteIfExists(tmp)
            } catch (_: Exception) {
            }
        }
    }
}
