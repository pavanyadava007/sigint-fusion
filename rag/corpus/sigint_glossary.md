# SIGINT, EW and fusion glossary (project working definitions)

- SIGINT: signals intelligence, split into COMINT (communications intelligence: voice and data links) and ELINT (electronic intelligence: radars and other non-communication emitters).
- ESM / R-ESM: electronic support measures; radar ESM receivers intercept radar pulses and output pulse descriptor words (PDWs).
- PDW: pulse descriptor word: time of arrival (TOA), radio frequency (RF), pulse width (PW), angle of arrival (AOA), amplitude, and sometimes intra-pulse modulation.
- Deinterleaving: sorting a mixed stream of pulses from several radars into per-emitter pulse trains, usually by clustering in RF/PW/AOA and then analysing TOA differences to estimate PRI.
- PRI: pulse repetition interval, the time between successive pulses; PRF = 1/PRI.
- EOB: electronic order of battle, the fused list of emitters (identity, location or bearing, parameters, activity) built from all sensors.
- Track: a filtered estimate of an emitter's parameters over time; this system uses a Kalman filter over RF and bearing with Hungarian (linear assignment) association gated on Mahalanobis distance.
- Fusion: combining detections from several sensors or sensor types (R-ESM pulses and COMINT I/Q classifications) into one track per emitter.
- Modulation classification (AMC): recognising the modulation type (BPSK, QPSK, 16QAM, FM, GMSK, ...) of an intercepted signal from raw I/Q samples; this system uses a 1-D residual CNN on I/Q trained with transfer learning, and a ViT on spectrograms for comparison.
- SEI: specific emitter identification, fingerprinting an individual transmitter from hardware impairments (I/Q imbalance, carrier frequency offset, phase noise, power-amplifier nonlinearity) independent of its modulation.
- Few-shot / prototypical classification: identify a new emitter or modulation from a handful of examples by comparing embeddings to class prototypes (mean embeddings); no retraining.
- Confidence: the classifier's softmax probability for its top class; below 0.6 an identification is reported as tentative.
- CFAR: constant false alarm rate detector; cell-averaging CFAR compares each spectrum cell with the mean of neighbouring training cells around guard cells.
- I/Q: in-phase and quadrature baseband samples, the complex representation of a received signal.
- Waterfall: time-frequency display of successive power spectra.
- Sirius Compact style split: edge sensors carry no threat library; all identification is done in the fusion backend where the library can be updated in seconds.
