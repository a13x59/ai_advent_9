package com.guitartab.worker

import com.guitartab.TranscriptionException
import io.ktor.client.HttpClient
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.bodyAsText
import io.ktor.client.statement.readBytes
import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.http.contentType

/** HTTP-клиент к Python-воркеру Basic Pitch. */
class BasicPitchClient(private val client: HttpClient, private val baseUrl: String) {

    suspend fun predict(wav: ByteArray): ByteArray {
        val resp = client.post("$baseUrl/predict") {
            contentType(ContentType.parse("audio/wav"))
            setBody(wav)
        }
        if (resp.status != HttpStatusCode.OK) {
            val text = runCatching { resp.bodyAsText() }.getOrDefault("")
            throw TranscriptionException(502, "Ошибка воркера транскрипции (${resp.status.value}): ${text.take(300)}")
        }
        return resp.readBytes()
    }
}
