package io.fortunetelling.mcp.fortune

import kotlin.random.Random

/**
 * A single page of a book. The page has a passage read from the [top] and one from the [bottom],
 * so the "line" argument decides which one is used for the divination.
 */
data class Page(val number: Int, val top: String, val bottom: String)

/**
 * Whether to read the passage from the top or the bottom of the page.
 */
enum class Line(val id: String) {
    TOP("top"),
    BOTTOM("bottom");

    companion object {
        fun fromId(id: String?): Line? = entries.firstOrNull { it.id == id }
    }
}

/**
 * The books available for bibliomancy.
 */
enum class Book(val id: String, val title: String, val pages: List<Page>) {
    THE_BIBLE(
        "the_bible",
        "The Bible",
        listOf(
            Page(1, "In the beginning God created the heaven and the earth.", "And God said, Let there be light: and there was light."),
            Page(2, "The Lord is my shepherd; I shall not want.", "He maketh me to lie down in green pastures."),
            Page(3, "Blessed are the meek: for they shall inherit the earth.", "Blessed are the pure in heart: for they shall see God."),
            Page(4, "Ask, and it shall be given you.", "Seek, and ye shall find."),
        ),
    ),
    WAR_AND_PEACE(
        "war_and_peace",
        "War and Peace",
        listOf(
            Page(1, "All happy families are alike; each unhappy family is unhappy in its own way.", "Everything depends on upbringing."),
            Page(2, "We can know only that we know nothing. And that is the highest degree of human wisdom.", "Nothing is so necessary for a young man as the company of intelligent women."),
            Page(3, "There is no greatness where there is no simplicity, goodness and truth.", "The strongest of all warriors are these two — Time and Patience."),
            Page(4, "If everyone fought for their own convictions there would be no war.", "Man lives consciously for himself, but is an unconscious instrument in the attainment of the historic aims of humanity."),
        ),
    ),
    ALICE_IN_WONDERLAND(
        "alice_in_wonderland",
        "Alice's Adventures in Wonderland",
        listOf(
            Page(1, "Begin at the beginning and go on till you come to the end: then stop.", "Curiouser and curiouser!"),
            Page(2, "I can't go back to yesterday because I was a different person then.", "Who in the world am I? Ah, that's the great puzzle."),
            Page(3, "We're all mad here.", "If you don't know where you are going any road can take you there."),
            Page(4, "It's no use going back to yesterday.", "Everything's got a moral, if only you can find it."),
        ),
    ),
    THE_ART_OF_WAR(
        "the_art_of_war",
        "The Art of War",
        listOf(
            Page(1, "The supreme art of war is to subdue the enemy without fighting.", "All warfare is based on deception."),
            Page(2, "Know thy self, know thy enemy. A thousand battles, a thousand victories.", "In the midst of chaos, there is also opportunity."),
            Page(3, "The greatest victory is that which requires no battle.", "Appear weak when you are strong, and strong when you are weak."),
            Page(4, "Victorious warriors win first and then go to war.", "Opportunities multiply as they are seized."),
        ),
    ),
    HAMLET(
        "hamlet",
        "Hamlet",
        listOf(
            Page(1, "To be, or not to be, that is the question.", "There is nothing either good or bad, but thinking makes it so."),
            Page(2, "This above all: to thine own self be true.", "Though this be madness, yet there is method in't."),
            Page(3, "The readiness is all.", "There are more things in heaven and earth, Horatio, than are dreamt of in your philosophy."),
            Page(4, "Brevity is the soul of wit.", "Give every man thy ear, but few thy voice."),
        ),
    ),
    THE_LITTLE_PRINCE(
        "the_little_prince",
        "The Little Prince",
        listOf(
            Page(1, "It is only with the heart that one can see rightly; what is essential is invisible to the eye.", "All grown-ups were once children... but only few of them remember it."),
            Page(2, "You become responsible, forever, for what you have tamed.", "What makes the desert beautiful is that somewhere it hides a well."),
            Page(3, "The most beautiful things in the world cannot be seen or touched, they are felt with the heart.", "Well, I must endure the presence of a few caterpillars if I wish to become acquainted with the butterflies."),
            Page(4, "It is the time you have wasted for your rose that makes your rose so important.", "A goal without a plan is just a wish."),
        ),
    );

    companion object {
        fun fromId(id: String?): Book? = entries.firstOrNull { it.id == id }
    }
}

/**
 * Bibliomancy (divination by books): open a random page of the chosen book and read a passage
 * from the top or the bottom of the page.
 */
object Bibliomancy {

    data class Prediction(
        val question: String,
        val book: Book,
        val line: Line,
        val page: Int,
        val passage: String,
    ) {
        /** Formatted as ❓Question: $question 📚 From book: $title 📄 Page: $page 📖 Answer: $passage */
        val text: String
            get() = "❓Question: $question 📚 From book: ${book.title} 📄 Page: $page 📖 Answer: $passage"
    }

    fun predict(question: String, book: Book, line: Line, random: Random = Random.Default): Prediction {
        val page = book.pages.random(random)
        val passage = if (line == Line.TOP) page.top else page.bottom
        return Prediction(
            question = question,
            book = book,
            line = line,
            page = page.number,
            passage = passage,
        )
    }
}
