package com.sigintfusion.gateway;

import org.apache.kafka.clients.admin.NewTopic;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.kafka.config.TopicBuilder;

/** Topics are provisioned by KafkaAdmin at startup (fail-fast is off, so a missing broker only logs). */
@Configuration
public class KafkaTopicsConfig {
    private static final int PARTITIONS = 3;
    private static final int REPLICAS = 1;

    @Bean
    public NewTopic iqChunksTopic() {
        return TopicBuilder.name(Topics.IQ_CHUNKS).partitions(PARTITIONS).replicas(REPLICAS).build();
    }

    @Bean
    public NewTopic pdwBatchesTopic() {
        return TopicBuilder.name(Topics.PDW_BATCHES).partitions(PARTITIONS).replicas(REPLICAS).build();
    }
}
