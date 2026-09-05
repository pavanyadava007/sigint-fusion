package com.sigintfusion.gateway;

import jakarta.validation.constraints.AssertTrue;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

/**
 * One I/Q capture from an edge sensor. Sensors hold no threat library; all analysis happens in the fusion backend.
 * Rules: i and q are 64..65536 finite samples of equal length; rfMhz optional in (0, 1e6); aoa optional in [0, 360].
 */
public record IqChunk(
        @NotBlank @Size(max = 64) @Pattern(regexp = "[A-Za-z0-9._-]+", message = "must match [A-Za-z0-9._-]+") String sensorId,
        Double rfMhz,
        Double aoa,
        @NotNull @Size(min = 64, max = 65536, message = "must contain 64..65536 samples") double[] i,
        @NotNull @Size(min = 64, max = 65536, message = "must contain 64..65536 samples") double[] q) {

    @AssertTrue(message = "i and q must have the same length")
    public boolean isSameLength() {
        return i == null || q == null || i.length == q.length;
    }

    @AssertTrue(message = "i and q samples must be finite numbers")
    public boolean isFinite() {
        return allFinite(i) && allFinite(q);
    }

    @AssertTrue(message = "rfMhz must be greater than 0 and less than 1e6")
    public boolean isRfMhzInRange() {
        return rfMhz == null || (rfMhz > 0 && rfMhz < 1_000_000);
    }

    @AssertTrue(message = "aoa must be between 0 and 360")
    public boolean isAoaInRange() {
        return aoa == null || (aoa >= 0 && aoa <= 360);
    }

    static boolean allFinite(double[] values) {
        if (values == null) return true;
        for (double v : values) {
            if (!Double.isFinite(v)) return false;
        }
        return true;
    }
}
