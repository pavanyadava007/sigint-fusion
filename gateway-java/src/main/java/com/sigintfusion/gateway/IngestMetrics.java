package com.sigintfusion.gateway;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Ingest counters. Registered in the MeterRegistry so they show up under /actuator/metrics and
 * /actuator/prometheus (ingest_accepted_total, ingest_iq_total, ingest_pdw_total, ingest_rejected_total)
 * and are also served as JSON by GET /ingest/stats.
 */
@Component
public class IngestMetrics {
    private final Counter accepted;
    private final Counter iq;
    private final Counter pdw;
    private final Counter rejected;

    public IngestMetrics(MeterRegistry registry) {
        accepted = Counter.builder("ingest.accepted").description("Payloads validated and handed to the Kafka producer").register(registry);
        iq = Counter.builder("ingest.iq").description("Accepted I/Q chunks").register(registry);
        pdw = Counter.builder("ingest.pdw").description("Accepted PDW batches").register(registry);
        rejected = Counter.builder("ingest.rejected").description("Requests rejected with HTTP 400").register(registry);
    }

    /** @return accepted_total after the increment, for the 202 body. */
    public long acceptedIq() {
        iq.increment();
        accepted.increment();
        return count(accepted);
    }

    public long acceptedPdw() {
        pdw.increment();
        accepted.increment();
        return count(accepted);
    }

    public void rejected() {
        rejected.increment();
    }

    public Map<String, Object> snapshot() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("accepted_total", count(accepted));
        out.put("iq_total", count(iq));
        out.put("pdw_total", count(pdw));
        out.put("rejected_total", count(rejected));
        return out;
    }

    private static long count(Counter c) {
        return Math.round(c.count());
    }
}
