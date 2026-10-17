package com.guitartab

import com.guitartab.audio.FfmpegConverter
import com.guitartab.model.ErrorResponse
import com.guitartab.quality.DeepSeekClient
import com.guitartab.quality.QualityGate
import com.guitartab.routes.configureRoutes
import com.guitartab.worker.BasicPitchClient
import io.ktor.client.HttpClient
import io.ktor.client.engine.cio.CIO
import io.ktor.client.plugins.HttpTimeout
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpMethod
import io.ktor.http.HttpStatusCode
import io.ktor.serialization.kotlinx.json.json
import io.ktor.server.application.Application
import io.ktor.server.application.call
import io.ktor.server.application.install
import io.ktor.server.engine.embeddedServer
import io.ktor.server.netty.Netty
import io.ktor.server.plugins.callloging.CallLogging
import io.ktor.server.plugins.contentnegotiation.ContentNegotiation
import io.ktor.server.plugins.cors.routing.CORS
import io.ktor.server.plugins.statuspages.StatusPages
import io.ktor.server.response.respond
import io.ktor.server.response.respondText
import io.ktor.server.routing.get
import io.ktor.server.routing.routing
import kotlinx.serialization.json.Json
import org.slf4j.LoggerFactory

fun main() {
    embeddedServer(Netty, port = Config.port, host = "0.0.0.0") { module() }.start(wait = true)
}

fun Application.module() {
    val logger = LoggerFactory.getLogger("Application")

    install(ContentNegotiation) {
        json(Json { ignoreUnknownKeys = true; encodeDefaults = true })
    }
    install(CallLogging)
    install(CORS) {
        anyHost()
        allowMethod(HttpMethod.Get)
        allowMethod(HttpMethod.Post)
        allowHeader(HttpHeaders.ContentType)
    }
    install(StatusPages) {
        exception<TranscriptionException> { call, cause ->
            call.respond(
                HttpStatusCode(cause.statusCode, cause.message ?: "error"),
                ErrorResponse(cause.message ?: "error"),
            )
        }
        exception<Throwable> { call, cause ->
            logger.error("Unhandled exception", cause)
            call.respond(HttpStatusCode.InternalServerError, ErrorResponse("Внутренняя ошибка сервера"))
        }
    }

    val httpClient = HttpClient(CIO) {
        install(HttpTimeout) {
            requestTimeoutMillis = 180_000
            connectTimeoutMillis = 10_000
            socketTimeoutMillis = 180_000
        }
    }

    val ffmpeg = FfmpegConverter(Config.ffmpegPath)
    val worker = BasicPitchClient(httpClient, Config.workerUrl)
    val deepSeek = if (Config.deepSeekApiKey.isBlank()) null
        else DeepSeekClient(httpClient, Config.deepSeekApiKey, Config.deepSeekModel)
    val gate = QualityGate(deepSeek, Config.enableLlmGate)
    val service = TranscriptionService(ffmpeg, worker, gate)

    routing {
        configureRoutes(service)
        get("/") {
            call.respondText(loadIndexHtml(), ContentType.Text.Html)
        }
    }
}

private fun loadIndexHtml(): String {
    val stream = object {}.javaClass.classLoader.getResourceAsStream("static/index.html")
    return stream?.readBytes()?.toString(Charsets.UTF_8) ?: "index.html not found"
}
