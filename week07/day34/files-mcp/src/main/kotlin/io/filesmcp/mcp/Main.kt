package io.filesmcp.mcp

import io.filesmcp.mcp.files.FileService
import io.filesmcp.mcp.server.registerFileTools
import io.filesmcp.mcp.web.fileRoutes
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
    val fileService = FileService()

    val mcpServer = Server(
        serverInfo = Implementation(name = "files-mcp", version = "1.0.0"),
        options = ServerOptions(
            capabilities = ServerCapabilities(
                tools = ServerCapabilities.Tools(listChanged = false),
            ),
        ),
    )
    mcpServer.registerFileTools(fileService)

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

        // MCP Streamable HTTP transport (for MCP Inspector / agent): POST /mcp
        mcpStreamableHttp(path = "/mcp", enableDnsRebindingProtection = false) {
            mcpServer
        }

        // Plain REST endpoints for browser testing
        routing {
            fileRoutes(fileService)
        }
    }.start(wait = true)
}
