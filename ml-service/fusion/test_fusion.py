from fusion.associate import Fuser


def test_two_sources_fuse_to_one_track():
    f = Fuser()
    f.step([dict(rf_mhz=9400, aoa=45, source="resm", sensor_id="esm-1", pri_us=500, pri_type="constant")], 0.0)
    eob = f.step([dict(rf_mhz=9401, aoa=46, source="comint", sensor_id="rx-2", label="pulse")], 1.0)
    assert len(eob) == 1 and eob[0]["sources"] == ["comint", "resm"]
    assert eob[0]["sensors"] == ["esm-1", "rx-2"] and eob[0]["pri_type"] == "constant" and eob[0]["modulation"] == "pulse"


def test_distinct_emitters_stay_separate():
    f = Fuser()
    eob = f.step([dict(rf_mhz=9400, aoa=45, source="resm"), dict(rf_mhz=2800, aoa=200, source="resm")], 0.0)
    assert len(eob) == 2


def test_updated_track_has_zero_misses_and_stale_track_expires():
    f = Fuser(max_misses=2)
    f.step([dict(rf_mhz=9400, aoa=45, source="resm")], 0.0)
    eob = f.step([dict(rf_mhz=9400, aoa=45, source="resm")], 1.0)
    assert eob[0]["misses"] == 0 and eob[0]["hits"] == 2
    for t in (2.0, 3.0):
        eob = f.step([], t)
    assert len(eob) == 1 and eob[0]["misses"] == 2
    assert f.step([], 4.0) == []


def test_bearing_wraparound_associates():
    f = Fuser()
    f.step([dict(rf_mhz=1090, aoa=359.5, source="resm")], 0.0)
    eob = f.step([dict(rf_mhz=1090, aoa=0.5, source="resm")], 1.0)
    assert len(eob) == 1 and eob[0]["hits"] == 2


def test_step_multi_fuses_two_sensors_and_merges_repeats():
    from fusion.associate import merge_duplicates

    reps = [dict(rf_mhz=156.8, aoa=40 + k * 0.5, source="comint", sensor_id="a", label="FM") for k in range(4)]
    assert len(merge_duplicates(reps)) == 1
    f = Fuser()
    eob = f.step_multi([reps, [dict(rf_mhz=156.9, aoa=41, source="comint", sensor_id="b", label="FM")]], 0.0)
    assert len(eob) == 1 and eob[0]["sensors"] == ["a", "b"] and eob[0]["hits"] == 2
