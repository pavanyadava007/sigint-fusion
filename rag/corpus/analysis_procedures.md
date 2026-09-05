# Analyst procedures (internal working notes for the SIGINT-Fusion console)

## Identifying an unknown emitter
1. Pull the last 10 minutes of detections and the current EOB. Note RF, bearing, source (comint or resm), modulation and confidence.
2. Query the emitter catalogue by RF (allow +/- 50 MHz for radars, +/- 2 MHz for VHF/UHF communications).
3. Search the knowledge base for the band allocation. Cite the source document in brackets.
4. If the classifier confidence is below 0.6 mark the identification as tentative and recommend a longer capture.
5. For radars, compare PRI type and pulse width with the catalogue entry before naming it.

## Report format
A report has the sections: Summary, Emitters (table: RF, bearing, modulation, PRI/PW when known, sources, confidence), Assessment, Confidence, Recommended actions. Reports are saved with save_report and appear in the console's Reports panel.

## Confidence language
- High: two or more sensors agree and the catalogue match is unambiguous.
- Medium: single sensor or catalogue match within tolerance only.
- Low / tentative: classifier probability below 0.6, or conflicting modulation labels on one track (label agreement below 0.7).

## Common false identifications
- 1090 MHz OOK detections are SSR/ADS-B replies, not a jammer.
- 2.4 GHz QPSK/OFDM near an airfield is usually Wi-Fi or a drone link; correlate bearing with the drone-control ISM band.
- Marine X-band radars near 9410 MHz are ubiquitous in coastal areas and are civilian unless PRI and scan pattern say otherwise.
- A stagger PRI on 2.8 GHz with 1 microsecond pulses is consistent with an airport surveillance radar.
