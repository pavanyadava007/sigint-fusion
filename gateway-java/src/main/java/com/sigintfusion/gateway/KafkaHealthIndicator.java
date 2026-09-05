package com.sigintfusion.gateway;

import org.apache.kafka.clients.admin.AdminClient;
import org.apache.kafka.clients.admin.ListTopicsOptions;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.kafka.core.KafkaAdmin;
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * Readiness signal for Kafka: lists topics through a long-lived AdminClient with a 2 s bound.
 * Registered as the "kafka" component of /actuator/health and of the readiness group.
 */
@Component("kafka")
public class KafkaHealthIndicator implements HealthIndicator, DisposableBean {
    private static final Duration TIMEOUT = Duration.ofSeconds(2);

    private final KafkaAdmin admin;
    private AdminClient client;

    public KafkaHealthIndicator(KafkaAdmin admin) {
        this.admin = admin;
    }

    @Override
    public Health health() {
        try {
            Set<String> names = client()
                    .listTopics(new ListTopicsOptions().timeoutMs((int) TIMEOUT.toMillis()))
                    .names()
                    .get(TIMEOUT.toMillis(), TimeUnit.MILLISECONDS);
            List<String> missing = new ArrayList<>();
            for (String t : List.of(Topics.IQ_CHUNKS, Topics.PDW_BATCHES)) {
                if (!names.contains(t)) missing.add(t);
            }
            Health.Builder b = Health.up().withDetail("topics", names.size());
            if (!missing.isEmpty()) b.withDetail("missingTopics", missing);
            return b.build();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return Health.down().withDetail("error", "interrupted").build();
        } catch (Exception e) {
            return Health.down().withDetail("error", e.getClass().getSimpleName() + ": " + e.getMessage()).build();
        }
    }

    private synchronized AdminClient client() {
        if (client == null) {
            client = AdminClient.create(admin.getConfigurationProperties());
        }
        return client;
    }

    @Override
    public synchronized void destroy() {
        if (client != null) {
            client.close(TIMEOUT);
            client = null;
        }
    }
}
