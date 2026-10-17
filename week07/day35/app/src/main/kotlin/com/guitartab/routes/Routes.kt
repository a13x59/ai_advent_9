package com.guitartab.routes

import com.guitartab.TranscriptionException
import com.guitartab.TranscriptionService
import com.guitartab.audio.AudioFormat
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.call
import io.ktor.server.request.receive
import io.ktor.server.response.respond
import io.ktor.server.response.respondText
import io.ktor.server.routing.Route
import io.ktor.server.routing.get
import io.ktor.server.routing.post

fun Route.configureRoutes(service: TranscriptionService) {

    get("/api/health") {
        call.respondText("OK")
    }

    post("/api/transcribe") {
        val bytes = call.receive<ByteArray>()
        val contentType = call.request.headers[HttpHeaders.ContentType]
        val format = AudioFormat.detect(bytes, contentType)
            ?: throw TranscriptionException(400, "Неподдерживаемый формат аудио (поддерживаются WAV, MP3, M4A)")

        val response = service.transcribe(bytes, format)
        call.respond(HttpStatusCode.OK, response)
    }
}
