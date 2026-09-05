package com.sigintfusion.gateway;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import java.util.Map;
import java.util.stream.Collectors;

/** Maps client errors to 400 {"error": "..."} and counts them; maps producer failures to 503. */
@RestControllerAdvice
public class IngestExceptionHandler {
    private static final Logger log = LoggerFactory.getLogger(IngestExceptionHandler.class);
    private static final int MAX_DETAIL = 200;

    private final IngestMetrics metrics;

    public IngestExceptionHandler(IngestMetrics metrics) {
        this.metrics = metrics;
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<Map<String, String>> onInvalid(MethodArgumentNotValidException ex) {
        String detail = ex.getBindingResult().getAllErrors().stream()
                .map(e -> {
                    // @AssertTrue checks carry a self-contained message; field constraints get the field name.
                    if (e instanceof FieldError fe && !"AssertTrue".equals(fe.getCode())) {
                        return fe.getField() + ": " + fe.getDefaultMessage();
                    }
                    return e.getDefaultMessage();
                })
                .sorted()
                .collect(Collectors.joining("; "));
        return reject(detail);
    }

    @ExceptionHandler(HttpMessageNotReadableException.class)
    public ResponseEntity<Map<String, String>> onUnreadable(HttpMessageNotReadableException ex) {
        String cause = ex.getMostSpecificCause().getMessage();
        return reject("malformed request body: " + firstLine(cause));
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<Map<String, String>> onIllegalArgument(IllegalArgumentException ex) {
        return reject(ex.getMessage() == null ? "invalid request" : firstLine(ex.getMessage()));
    }

    @ExceptionHandler({org.apache.kafka.common.KafkaException.class, org.springframework.kafka.KafkaException.class})
    public ResponseEntity<Map<String, String>> onKafka(Exception ex) {
        log.error("kafka producer failure: {}", ex.toString());
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(Map.of("error", "kafka unavailable"));
    }

    private ResponseEntity<Map<String, String>> reject(String detail) {
        metrics.rejected();
        log.debug("rejected: {}", detail);
        return ResponseEntity.badRequest().body(Map.of("error", detail));
    }

    private static String firstLine(String s) {
        if (s == null) return "unreadable";
        int nl = s.indexOf('\n');
        String line = nl >= 0 ? s.substring(0, nl) : s;
        return line.length() > MAX_DETAIL ? line.substring(0, MAX_DETAIL) : line;
    }
}
