package com.sigintfusion.gateway;

import jakarta.validation.constraints.AssertTrue;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

/** Pulse descriptor words: 1..10000 rows of exactly [toa_ms, rf_mhz, pw_us, aoa_deg], all finite. */
public record PdwBatch(
        @NotBlank @Size(max = 64) @Pattern(regexp = "[A-Za-z0-9._-]+", message = "must match [A-Za-z0-9._-]+") String sensorId,
        @NotNull @Size(min = 1, max = 10000, message = "must contain 1..10000 rows") double[][] pdw) {

    static final int ROW_LENGTH = 4;

    @AssertTrue(message = "each pdw row must contain exactly 4 finite numbers [toa_ms, rf_mhz, pw_us, aoa_deg]")
    public boolean isRowsWellFormed() {
        if (pdw == null) return true;
        for (double[] row : pdw) {
            if (row == null || row.length != ROW_LENGTH || !IqChunk.allFinite(row)) return false;
        }
        return true;
    }
}
