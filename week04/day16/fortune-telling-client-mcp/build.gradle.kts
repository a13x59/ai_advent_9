plugins {
    kotlin("jvm") version "2.4.0"
    application
    id("com.gradleup.shadow") version "8.3.6"
}

group = "io.fortunetelling"
version = "1.0.0"

repositories {
    mavenCentral()
}

val ktorVersion = "3.5.1"
val mcpVersion = "0.15.0"

dependencies {
    implementation("io.modelcontextprotocol:kotlin-sdk-client:$mcpVersion")
    implementation("io.ktor:ktor-client-cio:$ktorVersion")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.11.0")
    implementation("org.slf4j:slf4j-nop:2.0.19")

    testImplementation(kotlin("test"))
}

application {
    mainClass.set("io.fortunetelling.mcp.client.MainKt")
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

tasks.shadowJar {
    archiveBaseName.set("fortune-telling-mcp-client")
    archiveClassifier.set("all")
    archiveVersion.set("")
    mergeServiceFiles()
}
