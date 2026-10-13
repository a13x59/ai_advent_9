package io.pipeline.mcp

import io.pipeline.mcp.save.SaveService
import io.pipeline.mcp.search.SearchService
import io.pipeline.mcp.server.registerPipelineTools
import io.pipeline.mcp.summarize.SummarizeService
import io.pipeline.mcp.web.pipelineRoutes
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

fun main() {
    val searchService = SearchService()
    val summarizeService = SummarizeService()
    val saveService = SaveService()

    val mcpServer = Server(
        serverInfo = Implementation(name = "pipeline-mcp", version = "1.0.0"),
        options = ServerOptions(
            capabilities = ServerCapabilities(
                tools = ServerCapabilities.Tools(listChanged = false),
            ),
        ),
    )
    mcpServer.registerPipelineTools(searchService, summarizeService, saveService)

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
            pipelineRoutes(searchService, summarizeService, saveService)
        }
    }.start(wait = true)
}
