package io.filesmcp.mcp

import io.filesmcp.mcp.files.FileService
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.readText
import kotlin.io.path.writeText
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class FileServiceTest {

    private lateinit var tmp: Path
    private lateinit var fs: FileService

    @BeforeTest
    fun setUp() {
        tmp = Files.createTempDirectory("files-mcp-test")
        fs = FileService(tmp.toString())
    }

    @AfterTest
    fun tearDown() {
        tmp.toFile().deleteRecursively()
    }

    @Test
    fun `resolve rejects path traversal and absolute paths`() {
        assertFailsWith<FileService.FileException> { fs.resolve("../secret.txt") }
        assertFailsWith<FileService.FileException> { fs.resolve("../../etc/passwd") }
        assertFailsWith<FileService.FileException> { fs.resolve("/etc/passwd") }
    }

    @Test
    fun `read file returns content and line range`() {
        tmp.resolve("a.txt").writeText("one\ntwo\nthree\n")
        val r = fs.readFile("a.txt", 2, 3)
        assertEquals(3, r.totalLines)
        assertEquals("two\nthree", r.content)
        assertFalse(r.truncated)
    }

    @Test
    fun `read missing file fails`() {
        assertFailsWith<FileService.FileException> { fs.readFile("nope.txt", null, null) }
    }

    @Test
    fun `list dir returns files with relative paths`() {
        tmp.resolve("x.py").writeText("")
        Files.createDirectories(tmp.resolve("sub"))
        tmp.resolve("sub/y.py").writeText("")
        val (count, files) = fs.listDir(".", 0)
        assertEquals(2, count)
        assertTrue(files.contains("x.py"))
        assertTrue(files.contains("sub/y.py"))
    }

    @Test
    fun `search finds matches across multiple files`() {
        tmp.resolve("x.py").writeText("def foo():\n    pass\n")
        Files.createDirectories(tmp.resolve("sub"))
        tmp.resolve("sub/y.py").writeText("foo()\n")
        val r = fs.searchFiles("foo", ".", "*", false, 50)
        assertEquals(2, r.count)
        assertEquals(setOf("x.py", "sub/y.py"), r.matches.map { it.file }.toSet())
    }

    @Test
    fun `search respects glob filter`() {
        tmp.resolve("x.py").writeText("foo\n")
        tmp.resolve("y.txt").writeText("foo\n")
        val r = fs.searchFiles("foo", ".", "*.py", false, 50)
        assertEquals(1, r.count)
        assertEquals("x.py", r.matches[0].file)
    }

    @Test
    fun `search skips binary files and noise dirs`() {
        tmp.resolve("code.py").writeText("foo\n")
        Files.write(tmp.resolve("blob.bin"), byteArrayOf(0x00.toByte(), 0xFF.toByte(), 0xFE.toByte()))
        Files.createDirectories(tmp.resolve(".venv"))
        tmp.resolve(".venv/ignored.py").writeText("foo\n")
        val r = fs.searchFiles("foo", ".", "*", false, 50)
        assertEquals(1, r.count)
        assertEquals("code.py", r.matches[0].file)
    }

    @Test
    fun `write file in dry-run does not touch disk and returns diff`() {
        val r = fs.writeFile("new.md", "# Title\n")
        assertEquals("created", r.status)
        assertTrue(r.diff.contains("+# Title"))
        assertFalse(Files.exists(tmp.resolve("new.md")))
        assertEquals(1, fs.pendingCount())
    }

    @Test
    fun `edit file replaces single occurrence and diff is minimal`() {
        tmp.resolve("doc.md").writeText("line1\nOLD\nline3\n")
        val r = fs.editFile("doc.md", "OLD", "NEW", false)
        assertEquals(1, r.occurrences)
        assertTrue(r.diff.contains("-OLD"))
        assertTrue(r.diff.contains("+NEW"))
    }

    @Test
    fun `edit rejects ambiguous old string`() {
        tmp.resolve("doc.md").writeText("X X\n")
        assertFailsWith<FileService.FileException> { fs.editFile("doc.md", "X", "Y", false) }
    }

    @Test
    fun `edit with replace_all replaces every occurrence`() {
        tmp.resolve("doc.md").writeText("X and X\n")
        val r = fs.editFile("doc.md", "X", "Y", true)
        assertEquals(2, r.occurrences)
    }

    @Test
    fun `apply writes pending changes to disk and clears staging`() {
        fs.writeFile("a.md", "# A\n")
        fs.editFile("a.md", "# A", "# B", false)
        val applied = fs.applyChanges()
        assertEquals(listOf("a.md"), applied.applied)
        assertEquals("# B\n", tmp.resolve("a.md").readText())
        assertEquals(0, fs.pendingCount())
    }

    @Test
    fun `discard clears staging without writing`() {
        fs.writeFile("a.md", "# A\n")
        assertEquals(1, fs.discardChanges())
        assertEquals(0, fs.pendingCount())
        assertFalse(Files.exists(tmp.resolve("a.md")))
    }
}
