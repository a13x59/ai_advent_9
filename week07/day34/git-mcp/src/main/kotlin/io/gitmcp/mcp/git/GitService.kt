package io.gitmcp.mcp.git

import io.gitmcp.mcp.Config
import java.util.concurrent.TimeUnit

/**
 * Выполняет git-команды в каталоге проекта [Config.projectPath].
 *
 * Скоуп к «текущей папке» обеспечивается двумя вещами:
 *   1) `git -C <projectPath>` — git сам находит репозиторий, поднимаясь по дереву;
 *   2) pathspec `.` (или `path`) — ограничивает вывод подкаталогом проекта,
 *      даже если репозиторий больше проекта.
 */
class GitService(
    private val projectPath: String = Config.projectPath,
) {

    class GitException(message: String) : RuntimeException(message)

    data class Result(val exitCode: Int, val stdout: String, val stderr: String) {
        val isOk: Boolean get() = exitCode == 0
        val output: String get() = stdout.ifBlank { stderr }
    }

    fun run(vararg args: String): Result {
        val cmd = mutableListOf(Config.gitBin, "-C", projectPath)
        cmd.addAll(args)
        val process = ProcessBuilder(cmd)
            .start()
        val out = process.inputStream.bufferedReader().use { it.readText() }
        val err = process.errorStream.bufferedReader().use { it.readText() }
        val timedOut = !process.waitFor(30, TimeUnit.SECONDS)
        if (timedOut) {
            process.destroyForcibly()
            return Result(124, out, "git command timed out: $err")
        }
        return Result(process.exitValue(), out, err)
    }

    /** Текущая ветка (или "detached HEAD"). */
    fun currentBranch(): String {
        val r = run("branch", "--show-current")
        if (!r.isOk) throw GitException(r.stderr.trim().ifEmpty { "git branch failed" })
        return r.stdout.trim().ifEmpty { "detached HEAD" }
    }

    /** Список веток (без `*` у текущей). */
    fun branches(): List<String> {
        val r = run("branch", "--list")
        if (!r.isOk) throw GitException(r.stderr.trim().ifEmpty { "git branch failed" })
        return r.stdout.lines()
            .map { it.removePrefix("*").trim() }
            .filter { it.isNotBlank() }
    }

    /** Список отслеживаемых файлов проекта, скоупленный pathspec [path]. */
    fun listFiles(path: String, maxFiles: Int): Pair<Int, List<String>> {
        val r = run("ls-files", "--", path)
        if (!r.isOk) throw GitException(r.stderr.trim().ifEmpty { "git ls-files failed" })
        val files = r.stdout.lines().map { it.trim() }.filter { it.isNotBlank() }
        val limited = if (maxFiles > 0 && files.size > maxFiles) files.take(maxFiles) else files
        return files.size to limited
    }

    /** Дифф рабочей копии (или staged при [staged]), скоупленный pathspec [path].

     * `--relative` показывает пути относительно каталога проекта (а не корня
     * репозитория) — согласованно с `list_files` и требованием «скоуп = текущая папка».
     */
    fun diff(staged: Boolean, stat: Boolean, path: String): String {
        val args = mutableListOf("diff", "--relative")
        if (staged) args.add("--cached")
        if (stat) args.add("--stat")
        args.add("--")
        args.add(path)
        val r = run(*args.toTypedArray())
        if (!r.isOk) throw GitException(r.stderr.trim().ifEmpty { "git diff failed" })
        return r.stdout
    }
}
