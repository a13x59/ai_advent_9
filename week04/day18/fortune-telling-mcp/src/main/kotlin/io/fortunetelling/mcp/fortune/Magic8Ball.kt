package io.fortunetelling.mcp.fortune

import kotlin.random.Random

/**
 * The classic Magic 8 Ball with the standard set of 20 answers.
 */
object Magic8Ball {

    private val answers: List<String> = listOf(
        "It is certain.",
        "It is decidedly so.",
        "Without a doubt.",
        "Yes — definitely.",
        "You may rely on it.",
        "As I see it, yes.",
        "Most likely.",
        "Outlook good.",
        "Yes.",
        "Signs point to yes.",
        "Reply hazy, try again.",
        "Ask again later.",
        "Better not tell you now.",
        "Cannot predict now.",
        "Concentrate and ask again.",
        "Don't count on it.",
        "My reply is no.",
        "My sources say no.",
        "Outlook not so good.",
        "Very doubtful.",
    )

    /** One prediction outcome: the question and the randomly chosen answer. */
    data class Outcome(val question: String, val answer: String) {
        /** Formatted as ❓$question 🔮 $answer */
        val text: String
            get() = "❓$question 🔮 $answer"
    }

    fun randomAnswer(random: Random = Random.Default): String = answers.random(random)

    fun predict(question: String, random: Random = Random.Default): Outcome =
        Outcome(question = question, answer = randomAnswer(random))
}
