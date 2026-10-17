FROM gradle:8.11.1-jdk21 AS build
WORKDIR /src
COPY build.gradle.kts settings.gradle.kts ./
COPY src/ ./src/
RUN gradle --no-daemon installDist

FROM eclipse-temurin:21-jre
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /src/build/install/guitar-tab-service/ /app/
EXPOSE 8080
ENTRYPOINT ["/app/bin/guitar-tab-service"]
