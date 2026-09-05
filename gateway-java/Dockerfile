FROM maven:3.9-eclipse-temurin-21 AS build
# Proxy for dependency downloads. Maven ignores the *_PROXY env vars, so they are
# translated into JVM proxy properties (MAVEN_OPTS) for both mvn invocations.
ARG HTTPS_PROXY
ARG HTTP_PROXY
WORKDIR /src
COPY pom.xml .
RUN set -eu; opts=""; \
    for spec in "https=${HTTPS_PROXY:-}" "http=${HTTP_PROXY:-}"; do \
      scheme=${spec%%=*}; url=${spec#*=}; [ -n "$url" ] || continue; \
      hp=${url#*://}; hp=${hp%%/*}; host=${hp%%:*}; \
      case "$hp" in *:*) port=${hp##*:};; *) port=80;; esac; \
      opts="$opts -D${scheme}.proxyHost=${host} -D${scheme}.proxyPort=${port}"; \
    done; \
    echo "$opts" > /tmp/maven-proxy-opts; \
    MAVEN_OPTS="$opts" mvn -q -B dependency:go-offline
COPY src src
RUN MAVEN_OPTS="$(cat /tmp/maven-proxy-opts)" mvn -q -B -DskipTests package

FROM eclipse-temurin:21-jre
RUN useradd -r -u 10001 -s /usr/sbin/nologin gateway
COPY --from=build /src/target/gateway-0.1.0.jar /app.jar
ENV PORT=8080
EXPOSE 8080
USER gateway
# The JRE image ships neither curl nor wget; bash's /dev/tcp is enough for a GET.
# Health returns 200 only when the aggregate status is UP (Kafka reachable), else 503.
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=3 \
  CMD ["bash","-c","exec 3<>/dev/tcp/127.0.0.1/${PORT:-8080} && printf 'GET /actuator/health HTTP/1.0\\r\\nHost: localhost\\r\\n\\r\\n' >&3 && head -n 1 <&3 | grep -q ' 200 '"]
ENTRYPOINT ["java", "-jar", "/app.jar"]
