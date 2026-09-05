package com.sigintfusion.gateway;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Import;
import org.springframework.http.MediaType;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.support.SendResult;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import static org.assertj.core.api.Assertions.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.when;
import static org.mockito.ArgumentMatchers.eq;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(IngestController.class)
@Import(IngestMetrics.class)
class IngestControllerTest {

    @TestConfiguration
    static class MetricsTestConfig {
        @Bean
        MeterRegistry meterRegistry() {
            return new SimpleMeterRegistry();
        }
    }

    @Autowired MockMvc mvc;
    @Autowired ObjectMapper mapper;
    @MockBean KafkaTemplate<String, String> kafka;

    @BeforeEach
    void stubKafka() {
        CompletableFuture<SendResult<String, String>> done = CompletableFuture.completedFuture(null);
        when(kafka.send(anyString(), anyString(), anyString())).thenReturn(done);
    }

    @Test
    void validIqIsAcceptedAndPublishedKeyedBySensor() throws Exception {
        mvc.perform(post("/ingest/iq").contentType(MediaType.APPLICATION_JSON).content(iq("sirius-0", 128, 128, 9400.0, 12.3)))
                .andExpect(status().isAccepted())
                .andExpect(jsonPath("$.samples").value(128))
                .andExpect(jsonPath("$.accepted").isNumber());

        ArgumentCaptor<String> payload = ArgumentCaptor.forClass(String.class);
        verify(kafka).send(eq("iq-chunks"), eq("sirius-0"), payload.capture());
        JsonNode msg = mapper.readTree(payload.getValue());
        List<String> keys = new ArrayList<>();
        msg.fieldNames().forEachRemaining(keys::add);
        assertThat(keys).containsExactly("sensor_id", "rf_mhz", "aoa", "i", "q");
        assertThat(msg.get("sensor_id").asText()).isEqualTo("sirius-0");
        assertThat(msg.get("rf_mhz").asDouble()).isEqualTo(9400.0);
        assertThat(msg.get("aoa").asDouble()).isEqualTo(12.3);
        assertThat(msg.get("i").size()).isEqualTo(128);
        assertThat(msg.get("q").size()).isEqualTo(128);
    }

    @Test
    void iqWithoutOptionalFieldsIsAccepted() throws Exception {
        mvc.perform(post("/ingest/iq").contentType(MediaType.APPLICATION_JSON).content(iq("sirius-1", 64, 64, null, null)))
                .andExpect(status().isAccepted());
        ArgumentCaptor<String> payload = ArgumentCaptor.forClass(String.class);
        verify(kafka).send(eq("iq-chunks"), eq("sirius-1"), payload.capture());
        JsonNode msg = mapper.readTree(payload.getValue());
        assertThat(msg.get("rf_mhz").isNull()).isTrue();
        assertThat(msg.get("aoa").isNull()).isTrue();
    }

    @Test
    void iqLengthMismatchIsRejected() throws Exception {
        mvc.perform(post("/ingest/iq").contentType(MediaType.APPLICATION_JSON).content(iq("sirius-0", 128, 127, null, null)))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value(containsString("same length")));
        verify(kafka, never()).send(anyString(), anyString(), anyString());
    }

    @Test
    void iqTooShortIsRejected() throws Exception {
        mvc.perform(post("/ingest/iq").contentType(MediaType.APPLICATION_JSON).content(iq("sirius-0", 10, 10, null, null)))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value(containsString("64..65536")));
        verify(kafka, never()).send(anyString(), anyString(), anyString());
    }

    @Test
    void iqOutOfRangeRfAndAoaAreRejected() throws Exception {
        mvc.perform(post("/ingest/iq").contentType(MediaType.APPLICATION_JSON).content(iq("sirius-0", 64, 64, -5.0, 12.0)))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value(containsString("rfMhz")));
        mvc.perform(post("/ingest/iq").contentType(MediaType.APPLICATION_JSON).content(iq("sirius-0", 64, 64, 9400.0, 361.0)))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value(containsString("aoa")));
        verify(kafka, never()).send(anyString(), anyString(), anyString());
    }

    @Test
    void iqNonFiniteSampleIsRejected() throws Exception {
        // 1e400 parses to Infinity; "NaN" tokens are refused by the JSON parser (also 400).
        String body = iq("sirius-0", 64, 64, null, null).replaceFirst("\"i\":\\[0\\.0", "\"i\":[1e400");
        mvc.perform(post("/ingest/iq").contentType(MediaType.APPLICATION_JSON).content(body))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value(containsString("finite")));
        verify(kafka, never()).send(anyString(), anyString(), anyString());
    }

    @Test
    void validPdwIsAcceptedAndPublishedSnakeCase() throws Exception {
        double[][] rows = {{1.0, 9400.0, 0.5, 12.0}, {2.0, 9400.0, 0.5, 12.5}, {3.0, 9401.0, 0.6, 13.0}};
        mvc.perform(post("/ingest/pdw").contentType(MediaType.APPLICATION_JSON).content(pdw("sirius-esm", rows)))
                .andExpect(status().isAccepted())
                .andExpect(jsonPath("$.pulses").value(3));

        ArgumentCaptor<String> payload = ArgumentCaptor.forClass(String.class);
        verify(kafka).send(eq("pdw-batches"), eq("sirius-esm"), payload.capture());
        JsonNode msg = mapper.readTree(payload.getValue());
        List<String> keys = new ArrayList<>();
        msg.fieldNames().forEachRemaining(keys::add);
        assertThat(keys).containsExactly("sensor_id", "pdw");
        assertThat(msg.get("sensor_id").asText()).isEqualTo("sirius-esm");
        assertThat(msg.get("pdw").size()).isEqualTo(3);
        assertThat(msg.get("pdw").get(2).get(1).asDouble()).isEqualTo(9401.0);
    }

    @Test
    void malformedPdwRowIsRejected() throws Exception {
        double[][] shortRow = {{1.0, 9400.0, 0.5}};
        mvc.perform(post("/ingest/pdw").contentType(MediaType.APPLICATION_JSON).content(pdw("sirius-esm", shortRow)))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value(containsString("exactly 4")));

        String nonNumeric = "{\"sensorId\":\"sirius-esm\",\"pdw\":[[1.0,\"x\",0.5,12.0]]}";
        mvc.perform(post("/ingest/pdw").contentType(MediaType.APPLICATION_JSON).content(nonNumeric))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value(containsString("malformed")));

        mvc.perform(post("/ingest/pdw").contentType(MediaType.APPLICATION_JSON).content(pdw("sirius-esm", new double[0][])))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value(containsString("1..10000")));
        verify(kafka, never()).send(anyString(), anyString(), anyString());
    }

    @Test
    void malformedJsonIsRejected() throws Exception {
        mvc.perform(post("/ingest/iq").contentType(MediaType.APPLICATION_JSON).content("{not json"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value(containsString("malformed")));
        verify(kafka, never()).send(anyString(), anyString(), anyString());
    }

    @Test
    void statsCountAcceptedAndRejected() throws Exception {
        // The registry is shared across tests in this context, so assert on deltas.
        JsonNode before = stats();

        mvc.perform(post("/ingest/iq").contentType(MediaType.APPLICATION_JSON).content(iq("sirius-0", 64, 64, null, null)))
                .andExpect(status().isAccepted());
        mvc.perform(post("/ingest/pdw").contentType(MediaType.APPLICATION_JSON).content(pdw("sirius-esm", new double[][]{{1, 2, 3, 4}})))
                .andExpect(status().isAccepted());
        mvc.perform(post("/ingest/iq").contentType(MediaType.APPLICATION_JSON).content(iq("sirius-0", 64, 63, null, null)))
                .andExpect(status().isBadRequest());

        JsonNode after = stats();
        assertThat(after.get("accepted_total").asLong() - before.get("accepted_total").asLong()).isEqualTo(2);
        assertThat(after.get("iq_total").asLong() - before.get("iq_total").asLong()).isEqualTo(1);
        assertThat(after.get("pdw_total").asLong() - before.get("pdw_total").asLong()).isEqualTo(1);
        assertThat(after.get("rejected_total").asLong() - before.get("rejected_total").asLong()).isEqualTo(1);
    }

    private JsonNode stats() throws Exception {
        MvcResult r = mvc.perform(get("/ingest/stats")).andExpect(status().isOk())
                .andExpect(jsonPath("$.accepted_total").isNumber())
                .andExpect(jsonPath("$.iq_total").isNumber())
                .andExpect(jsonPath("$.pdw_total").isNumber())
                .andExpect(jsonPath("$.rejected_total").isNumber())
                .andReturn();
        return mapper.readTree(r.getResponse().getContentAsString());
    }

    private String iq(String sensor, int nI, int nQ, Double rfMhz, Double aoa) throws Exception {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("sensorId", sensor);
        if (rfMhz != null) body.put("rfMhz", rfMhz);
        if (aoa != null) body.put("aoa", aoa);
        body.put("i", samples(nI));
        body.put("q", samples(nQ));
        return mapper.writeValueAsString(body);
    }

    private String pdw(String sensor, double[][] rows) throws Exception {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("sensorId", sensor);
        body.put("pdw", rows);
        return mapper.writeValueAsString(body);
    }

    private static double[] samples(int n) {
        double[] v = new double[n];
        for (int k = 0; k < n; k++) v[k] = k * 0.001;
        return v;
    }
}
