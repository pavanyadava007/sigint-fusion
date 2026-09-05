package com.sigintfusion.gateway;

/** Kafka topic names shared by the controller, topic provisioning and the health check. */
final class Topics {
    static final String IQ_CHUNKS = "iq-chunks";
    static final String PDW_BATCHES = "pdw-batches";

    private Topics() {}
}
