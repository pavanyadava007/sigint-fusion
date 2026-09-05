package com.sigintfusion.gateway;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.LinkedHashMap;
import java.util.Map;

@RestController
@RequestMapping("/ingest")
public class IngestController {
    private static final Logger log = LoggerFactory.getLogger(IngestController.class);

    private final KafkaTemplate<String, String> kafka;
    private final ObjectMapper mapper;
    private final IngestMetrics metrics;

    public IngestController(KafkaTemplate<String, String> kafka, ObjectMapper mapper, IngestMetrics metrics) {
        this.kafka = kafka;
        this.mapper = mapper;
        this.metrics = metrics;
    }

    @PostMapping("/iq")
    public ResponseEntity<Map<String, Object>> iq(@Valid @RequestBody IqChunk chunk) throws JsonProcessingException {
        // LinkedHashMap rather than Map.of: rf_mhz and aoa are optional and may be null.
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("sensor_id", chunk.sensorId());
        payload.put("rf_mhz", chunk.rfMhz());
        payload.put("aoa", chunk.aoa());
        payload.put("i", chunk.i());
        payload.put("q", chunk.q());
        publish(Topics.IQ_CHUNKS, chunk.sensorId(), payload);
        long total = metrics.acceptedIq();
        return ResponseEntity.accepted().body(Map.of("accepted", total, "samples", chunk.i().length));
    }

    @PostMapping("/pdw")
    public ResponseEntity<Map<String, Object>> pdw(@Valid @RequestBody PdwBatch batch) throws JsonProcessingException {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("sensor_id", batch.sensorId());
        payload.put("pdw", batch.pdw());
        publish(Topics.PDW_BATCHES, batch.sensorId(), payload);
        long total = metrics.acceptedPdw();
        return ResponseEntity.accepted().body(Map.of("accepted", total, "pulses", batch.pdw().length));
    }

    @GetMapping("/stats")
    public Map<String, Object> stats() {
        return metrics.snapshot();
    }

    /** Key by sensorId so per-sensor ordering is preserved within a partition. The send is not awaited. */
    private void publish(String topic, String key, Map<String, Object> payload) throws JsonProcessingException {
        String json = mapper.writeValueAsString(payload);
        kafka.send(topic, key, json).whenComplete((result, ex) -> {
            if (ex != null) {
                log.error("kafka publish failed topic={} key={} bytes={}: {}", topic, key, json.length(), ex.toString());
            } else if (log.isDebugEnabled() && result != null) {
                log.debug("kafka publish ok topic={} key={} partition={} offset={}", topic, key,
                        result.getRecordMetadata().partition(), result.getRecordMetadata().offset());
            }
        });
    }
}
