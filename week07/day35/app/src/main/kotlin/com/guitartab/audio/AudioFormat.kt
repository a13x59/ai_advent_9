package com.guitartab.audio

enum class AudioFormat(val ext: String) {
    WAV("wav"),
    MP3("mp3"),
    M4A("m4a");

    companion object {
        /** Определяет формат по Content-Type и/или магическим байтам. */
        fun detect(bytes: ByteArray, contentType: String?): AudioFormat? {
            val ct = contentType?.lowercase()
            if (ct != null) {
                when {
                    ct.contains("wav") || ct.contains("wave") -> return WAV
                    ct.contains("mpeg") || ct.contains("mp3") -> return MP3
                    ct.contains("mp4") || ct.contains("m4a") || ct.contains("aac") -> return M4A
                }
            }
            // RIFF....WAVE
            if (bytes.size >= 12 &&
                bytes[0] == 'R'.code.toByte() && bytes[1] == 'I'.code.toByte() &&
                bytes[2] == 'F'.code.toByte() && bytes[3] == 'F'.code.toByte() &&
                bytes[8] == 'W'.code.toByte() && bytes[9] == 'A'.code.toByte() &&
                bytes[10] == 'V'.code.toByte() && bytes[11] == 'E'.code.toByte()
            ) return WAV
            // ID3
            if (bytes.size >= 3 && bytes[0] == 'I'.code.toByte() && bytes[1] == 'D'.code.toByte() && bytes[2] == '3'.code.toByte()) return MP3
            // MPEG frame sync 0xFFEx
            if (bytes.size >= 2 && (bytes[0].toInt() and 0xFF) == 0xFF && (bytes[1].toInt() and 0xE0) == 0xE0) return MP3
            // ftyp (MP4/M4A)
            if (bytes.size >= 12 && bytes[4] == 'f'.code.toByte() && bytes[5] == 't'.code.toByte() &&
                bytes[6] == 'y'.code.toByte() && bytes[7] == 'p'.code.toByte()
            ) return M4A
            return null
        }
    }
}
