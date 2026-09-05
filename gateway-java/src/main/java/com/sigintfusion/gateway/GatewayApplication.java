package com.sigintfusion.gateway;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/** Sensor ingest gateway: validates edge-sensor payloads (I/Q chunks, PDWs) and publishes them to Kafka for the ML service. */
@SpringBootApplication
public class GatewayApplication {
    public static void main(String[] args) { SpringApplication.run(GatewayApplication.class, args); }
}
