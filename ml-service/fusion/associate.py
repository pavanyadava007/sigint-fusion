"""Multi-sensor emitter fusion -> Electronic Order of Battle (EOB).

One Kalman track per emitter with state [RF, AOA, RF-rate, AOA-rate]. New detections from R-ESM (PDW clusters)
and COMINT (I/Q classifier) sources are associated with the Hungarian algorithm on gated Mahalanobis distance.
Bearing residuals are wrapped to [-180, 180). Tracks are dropped after `max_misses` consecutive steps without an update.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import count

import numpy as np
from scipy.optimize import linear_sum_assignment

_H = np.eye(2, 4)
_R = np.diag([1.0, 2.0])  # measurement noise: RF (MHz^2), AOA (deg^2)
_ids = count(1)


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


@dataclass
class Track:
    x: np.ndarray
    P: np.ndarray
    t: float
    id: int = field(default_factory=lambda: next(_ids))
    first_seen: float = 0.0
    sources: set = field(default_factory=set)
    sensors: set = field(default_factory=set)
    hits: int = 0
    misses: int = 0
    labels: list = field(default_factory=list)
    attrs: dict = field(default_factory=dict)  # last R-ESM attributes (pw, pri, pri_type)

    @classmethod
    def new(cls, z, det: dict, t: float) -> Track:
        tr = cls(x=np.array([z[0], z[1], 0.0, 0.0]), P=np.eye(4) * 10.0, t=t, first_seen=t)
        tr.update(z, det, t)
        return tr

    def predict(self, dt: float) -> None:
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        Q = np.eye(4) * 0.01
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def innovation(self, z):
        S = _H @ self.P @ _H.T + _R
        y = z - _H @ self.x
        y[1] = _wrap(y[1])
        return y, S

    def mahalanobis(self, z) -> float:
        y, S = self.innovation(z)
        return float(np.sqrt(y @ np.linalg.solve(S, y)))

    def update(self, z, det: dict, t: float) -> None:
        y, S = self.innovation(z)
        K = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ _H) @ self.P
        self.t, self.hits, self.misses = t, self.hits + 1, 0
        self.sources.add(det["source"])
        if det.get("sensor_id"):
            self.sensors.add(det["sensor_id"])
        if det.get("label"):
            self.labels.append(det["label"])
        for k in ("pw_us", "pri_us", "pri_type"):
            if det.get(k) is not None:
                self.attrs[k] = det[k]

    def eob(self) -> dict:
        lab = Counter(self.labels).most_common(1)
        return dict(track_id=self.id, rf_mhz=round(float(self.x[0]), 2), aoa=round(float(self.x[1]) % 360, 1),
                    rf_rate=round(float(self.x[2]), 3), aoa_rate=round(float(self.x[3]), 3),
                    sources=sorted(self.sources), sensors=sorted(self.sensors), hits=self.hits, misses=self.misses,
                    first_seen=self.first_seen, last_seen=self.t,
                    modulation=lab[0][0] if lab else None,
                    label_agreement=round(lab[0][1] / len(self.labels), 2) if lab else None, **self.attrs)


def merge_duplicates(detections: list[dict], rf_tol: float = 0.005, aoa_tol: float = 5.0) -> list[dict]:
    """Merge repeated observations of one emitter inside a single detection set (same source, RF within 0.5 % or 2 MHz,
    bearing within aoa_tol): RF/AOA are averaged, the most frequent label kept, sensors preserved."""
    out: list[dict] = []
    for d in sorted(detections, key=lambda d: d["rf_mhz"]):
        for m in out:
            if m["source"] == d["source"] and abs(m["rf_mhz"] - d["rf_mhz"]) <= max(2.0, rf_tol * m["rf_mhz"]) and abs(_wrap(m["aoa"] - d["aoa"])) <= aoa_tol:
                n = m["_n"]
                m["rf_mhz"] = (m["rf_mhz"] * n + d["rf_mhz"]) / (n + 1)
                m["aoa"] = (m["aoa"] + _wrap(d["aoa"] - m["aoa"]) / (n + 1)) % 360
                m["_labels"].append(d.get("label"))
                m["_n"] = n + 1
                break
        else:
            out.append(dict(d, _n=1, _labels=[d.get("label")]))
    for m in out:
        labs = [x for x in m.pop("_labels") if x]
        m.pop("_n")
        if labs:
            m["label"] = Counter(labs).most_common(1)[0][0]
    return out


class Fuser:
    def __init__(self, gate: float = 4.0, max_misses: int = 5):
        self.tracks: list[Track] = []
        self.gate, self.max_misses, self.t = gate, max_misses, 0.0

    def step(self, detections: list[dict], t: float) -> list[dict]:
        """detections: dict(rf_mhz, aoa, source in {'resm','comint'}, sensor_id?, label?, pw_us?, pri_us?, pri_type?)."""
        return self.step_multi([detections], t)

    def step_multi(self, groups: list[list[dict]], t: float) -> list[dict]:
        """One time step with several independent detection sets (typically one per sensor). Each set is associated
        in turn, so two sensors reporting the same emitter both update one track; ageing happens once per step."""
        dt = max(t - self.t, 0.0)
        self.t = t
        for tr in self.tracks:
            tr.predict(dt)
        updated: set[int] = set()
        for detections in groups:
            updated |= self._associate(merge_duplicates(detections), t)
        for i, tr in enumerate(self.tracks):
            if i not in updated and tr.t != t:
                tr.misses += 1
        self.tracks = [tr for tr in self.tracks if tr.misses <= self.max_misses]
        return self.eob()

    def _associate(self, detections: list[dict], t: float) -> set[int]:
        Z = np.array([[d["rf_mhz"], d["aoa"]] for d in detections], float).reshape(-1, 2)
        updated: set[int] = set()
        if not len(Z):
            return updated
        matched: set[int] = set()
        if self.tracks:
            C = np.array([[tr.mahalanobis(z) for z in Z] for tr in self.tracks])
            rows, cols = linear_sum_assignment(C)
            for i, j in zip(rows, cols):
                if C[i, j] < self.gate:
                    self.tracks[i].update(Z[j], detections[j], t)
                    updated.add(i)
                    matched.add(j)
        for j in set(range(len(Z))) - matched:
            self.tracks.append(Track.new(Z[j], detections[j], t))
            updated.add(len(self.tracks) - 1)
        return updated

    def eob(self) -> list[dict]:
        return [tr.eob() for tr in self.tracks]
