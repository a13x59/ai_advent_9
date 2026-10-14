package io.pipeline.mcp

import io.pipeline.mcp.save.SaveService
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SaveServiceTest {

    private val json = Json { ignoreUnknownKeys = true }

    @Test
    fun sanitizesFilenameAndWrites() {
        val service = SaveService()
        val result = json.parseToJsonElement(service.saveToFile("my  summary!.txt", "hello")).jsonObject

        assertEquals("my__summary_.txt", result["filename"]?.jsonPrimitive?.content)
        assertEquals(5, result["bytes"]?.jsonPrimitive?.content?.toIntOrNull())
        assertTrue(result["path"]?.jsonPrimitive?.content.orEmpty().endsWith("my__summary_.txt"))
    }
}
