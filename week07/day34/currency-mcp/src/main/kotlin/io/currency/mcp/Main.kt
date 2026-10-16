package io.currency.mcp

import io.currency.mcp.db.RateDb
import io.currency.mcp.rates.RateService
import io.currency.mcp.rates.SummaryService
import io.currency.mcp.server.registerCurrencyTools
import io.currency.mcp.web.currencyRoutes
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpMethod
import io.ktor.server.application.install
import io.ktor.server.engine.embeddedServer
import io.ktor.server.netty.Netty
import io.ktor.server.plugins.cors.routing.CORS
import io.ktor.server.routing.routing
import io.modelcontextprotocol.kotlin.sdk.server.Server
import io.modelcontextprotocol.kotlin.sdk.server.ServerOptions
import io.modelcontextprotocol.kotlin.sdk.server.mcpStreamableHttp
import io.modelcontextprotocol.kotlin.sdk.types.Implementation
import io.modelcontextprotocol.kotlin.sdk.types.ServerCapabilities
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

fun main() {
    val db = RateDb(Config.dbPath)
    val rateService = RateService(db)
    val summaries = SummaryService(db)

    val mcpServer = Server(
        serverInfo = Implementation(name = "currency-mcp", version = "1.0.0"),
        options = ServerOptions(
            capabilities = ServerCapabilities(
                tools = ServerCapabilities.Tools(listChanged = false),
            ),
        ),
    )
    mcpServer.registerCurrencyTools(db, summaries)

    // Background jobs: rate polling (every pollIntervalMs) and daily summarization
    // (every summaryIntervalMs), running 24/7 as long as the service is up.
    val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    scope.launch {
        while (isActive) {
            rateService.pollOnce()
            delay(Config.pollIntervalMs)
        }
    }
    scope.launch {
        while (isActive) {
            delay(Config.summaryIntervalMs)
            summaries.summarizeToday()
        }
    }

    embeddedServer(Netty, host = "0.0.0.0", port = Config.port) {
        install(CORS) {
            anyHost()
            allowHeader(HttpHeaders.ContentType)
            allowHeader(HttpHeaders.Authorization)
            allowMethod(HttpMethod.Get)
            allowMethod(HttpMethod.Post)
            allowMethod(HttpMethod.Delete)
            allowMethod(HttpMethod.Options)
        }

        // MCP Streamable HTTP transport (for MCP Inspector): POST /mcp
        mcpStreamableHttp(path = "/mcp", enableDnsRebindingProtection = false) {
            mcpServer
        }

        // Plain REST endpoints for browser testing
        routing {
            currencyRoutes(db, summaries)
        }
    }.start(wait = true)
}
