package io.fortunetelling.mcp.client

import io.ktor.client.HttpClient
import io.ktor.client.engine.cio.CIO
import io.ktor.client.plugins.sse.SSE
import io.modelcontextprotocol.kotlin.sdk.client.Client
import io.modelcontextprotocol.kotlin.sdk.client.StreamableHttpClientTransport
import io.modelcontextprotocol.kotlin.sdk.types.Implementation
import io.modelcontextprotocol.kotlin.sdk.types.ToolSchema
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlin.system.exitProcess

private const val DEFAULT_URL = "http://localhost:8888/mcp"

private val prettyJson = Json {
    prettyPrint = true
    prettyPrintIndent = "  "
    encodeDefaults = true
    explicitNulls = false
}

fun main(args: Array<String>) {
    val url = args.firstOrNull()?.takeIf { it.isNotBlank() } ?: DEFAULT_URL

    val httpClient = HttpClient(CIO) {
        install(SSE)
    }

    try {
        runBlocking {
            val client = Client(
                clientInfo = Implementation(name = "fortune-telling-client", version = "1.0.0"),
            )
            try {
                println("Connecting to MCP server at $url ...")

                // 1) Establish the MCP connection (includes the initialize handshake).
                client.connect(StreamableHttpClientTransport(client = httpClient, url = url))

                val server = client.serverVersion
                println("Connected to: ${server?.name ?: "server"} v${server?.version ?: "?"}")
                println()

                // 2) Fetch the list of available tools.
                val tools = client.listTools().tools
                println("Available tools: ${tools.size}")
                println()

                tools.forEachIndexed { index, tool ->
                    println("${index + 1}. ${tool.name}${tool.title?.let { " (title: $it)" } ?: ""}")
                    tool.description?.let { println("   Description: $it") }
                    println("   Input schema:")
                    println(prettyJson.encodeToString(ToolSchema.serializer(), tool.inputSchema).prependIndent("     "))
                    println()
                }
            } finally {
                client.close()
            }
        }
    } catch (e: Exception) {
        System.err.println()
        System.err.println("Failed to talk to the MCP server at $url")
        System.err.println("Cause: ${e.message ?: e::class.simpleName}")
        System.err.println()
        System.err.println("Hint: make sure the server is running (e.g. from the fortune-telling-mcp project):")
        System.err.println("  ./gradlew run          # then re-run this client")
        exitProcess(1)
    } finally {
        httpClient.close()
    }
}
