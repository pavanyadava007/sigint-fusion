# Frequency allocations and well-known emitters (open, paraphrased from public ITU Radio Regulations and ICAO/IMO material)

## Aeronautical
- 108-118 MHz: aeronautical radionavigation. VOR beacons and ILS localiser (108-112 MHz). ILS glide path uses 328-335 MHz.
- 118-137 MHz: aeronautical mobile (R), the VHF "airband". Voice is AM double-sideband with carrier (AM-DSB-WC); channel spacing is 25 kHz, and 8.33 kHz in Europe. 121.5 MHz is the international aeronautical emergency frequency. 123.1 MHz is used for search and rescue on scene.
- 243.0 MHz: military UHF aeronautical emergency (guard) frequency; the military UHF airband spans 225-400 MHz, AM voice.
- 960-1215 MHz: aeronautical radionavigation: DME and TACAN (pulse pairs), SSR/IFF and ADS-B.
- 1030 MHz: secondary surveillance radar (SSR) interrogation from the ground. 1090 MHz: transponder replies (Mode A/C/S) and ADS-B extended squitter. Mode S replies use pulse position modulation at 1 Mbit/s; the 1090 MHz downlink can be treated as on-off keyed pulses.
- 1575.42 MHz: GPS L1 (C/A code, BPSK spread spectrum); 1227.60 MHz GPS L2; 1176.45 MHz GPS L5. Galileo E1 is centred at 1575.42 MHz as well.
- 2700-2900 MHz: aeronautical radionavigation, primary airport surveillance radar (ASR), S-band. Typical ASR pulse width around 1 microsecond, PRF 700-1200 Hz (PRI about 0.8-1.4 ms), mechanical rotation 12-15 rpm.
- 5600-5650 MHz: meteorological (weather) radar, C-band, often with pulse-to-pulse PRI jitter or staggered PRF to resolve range ambiguities.
- 9000-9200 MHz: precision approach radar (PAR) and airport surface detection equipment, X-band.

## Maritime
- 156-162.025 MHz: maritime mobile VHF, FM voice with 25 kHz channel spacing. Channel 16 (156.800 MHz) is the international distress, safety and calling channel. Channel 70 (156.525 MHz) carries DSC (digital selective calling) data. Channel 13 (156.650 MHz) is bridge-to-bridge navigation.
- 161.975 MHz (AIS 1) and 162.025 MHz (AIS 2): Automatic Identification System, GMSK at 9600 bit/s.
- 406.0-406.1 MHz: COSPAS-SARSAT emergency beacons (EPIRB, ELT, PLB); 121.5 MHz homing signal.
- 2900-3100 MHz: S-band marine radar (long range, better in rain).
- 9300-9500 MHz: maritime radionavigation, X-band marine navigation radar. Typical magnetron navigation radar: pulse width 0.05-1 microsecond, PRF 1000-3000 Hz (PRI 0.3-1 ms), antenna rotation 20-45 rpm, frequency near 9410 MHz.
- 518 kHz: NAVTEX broadcasts (FSK).

## Broadcast and land mobile
- 87.5-108 MHz: FM broadcast, wideband FM with 75 kHz deviation.
- 144-146 MHz and 430-440 MHz: amateur radio (FM voice, digital modes).
- 380-400 MHz: TETRA public safety trunked radio in Europe (pi/4-DQPSK, 25 kHz channels).
- 446.0-446.2 MHz: PMR446 licence-free handhelds, FM.
- 470-694 MHz: DVB-T/T2 television broadcast (OFDM).
- 700-960 MHz, 1.7-2.7 GHz, 3.4-3.8 GHz: cellular (LTE/5G NR, OFDM); 2400-2483.5 MHz ISM: Wi-Fi, Bluetooth, drone control links (often GFSK or OFDM, frequency hopping).
- 5725-5875 MHz: ISM, Wi-Fi and drone video links.

## Radar bands (IEEE letter designations)
- L band 1-2 GHz: long-range air surveillance and SSR.
- S band 2-4 GHz: airport surveillance, weather, naval surveillance (e.g. 2700-3100 MHz).
- C band 4-8 GHz: weather radar (5.6 GHz), some fire-control.
- X band 8-12 GHz: marine navigation (9.3-9.5 GHz), airborne intercept, fire control, precision approach, synthetic aperture radar.
- Ku band 12-18 GHz: satellite communications, some short-range radar.
- 2-18 GHz is the typical electronic support measures (ESM) coverage band of an airborne radar warning receiver; modern systems extend to 40 GHz.

## Direction finding and ESM
- Direction finding accuracy of about 1 degree RMS is typical for interferometric DF; amplitude-comparison DF on RWRs is typically 5-15 degrees.
- Angle of arrival (AOA), radio frequency (RF), pulse width (PW), pulse repetition interval (PRI) and time of arrival (TOA) form a pulse descriptor word (PDW).
- PRI types: constant (stable), staggered (a repeating pattern of a few discrete PRIs, used to remove blind speeds), jittered (random variation, typically a few percent, used against range-gate pull-off and for ECCM), dwell-and-switch, and frequency-agile radars that hop RF pulse to pulse.
