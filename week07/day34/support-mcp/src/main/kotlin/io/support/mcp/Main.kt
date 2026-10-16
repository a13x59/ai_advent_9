package io.support.mcp

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
import io.support.mcp.db.SupportDb
import io.support.mcp.server.registerSupportTools
import io.support.mcp.web.supportRoutes

fun main() {
    val db = SupportDb(Config.dbPath)
    if (Config.seedData) {
        db.seedIfEmpty()
    }

    val mcpServer = Server(
        serverInfo = Implementation(name = "support-mcp", version = "1.0.0"),
        options = ServerOptions(
            capabilities = ServerCapabilities(
                tools = ServerCapabilities.Tools(listChanged = false),
            ),
        ),
    )
    mcpServer.registerSupportTools(db)

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

        // MCP Streamable HTTP transport (для MCP Inspector / агента): POST /mcp
        mcpStreamableHttp(path = "/mcp", enableDnsRebindingProtection = false) {
            mcpServer
        }

        // Plain REST endpoints для браузерного тестирования
        routing {
            supportRoutes(db)
        }
    }.start(wait = true)
}
