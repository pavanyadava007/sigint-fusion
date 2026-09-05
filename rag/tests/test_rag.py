import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest import chunk_markdown, load_corpus  # noqa: E402

from retrieval import emitter_text, frequencies_mhz, numbers_in  # noqa: E402

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus")


def test_chunks_carry_heading_and_overlap():
    text = "# Maritime\n" + " ".join(f"w{i}" for i in range(400)) + "\n# Other\nshort section"
    chunks = list(chunk_markdown(text, size=100, overlap=20))
    assert chunks[0].startswith("Maritime: w0") and chunks[-1] == "Other: short section"
    assert "w80" in chunks[0] and "w80" in chunks[1]  # overlap
    assert all(len(c.split()) <= 101 for c in chunks)


def test_emitter_text_mentions_band_and_pri():
    t = emitter_text(dict(name="X radar", type="nav", rf_min_mhz=9300, rf_max_mhz=9500, pri_us=500, pw_us=0.5, modulation="pulse", notes="n"))
    assert "9300-9500 MHz" in t and "PRI 500 us" in t and t.startswith("X radar (nav)")


def test_corpus_loads_and_is_nontrivial():
    rows, emitters = load_corpus(CORPUS)
    assert len(emitters) >= 20 and len(rows) >= 30
    assert all(r[2].strip() for r in rows) and len({(r[0], r[1]) for r in rows}) == len(rows)


def test_numbers_in_extracts_frequencies():
    assert numbers_in("distress on 156.8 MHz and 1090 MHz, channel 16") == ["156.8", "1090"]


def test_frequencies_mhz_units():
    assert frequencies_mhz("around 9.4 GHz and 156.8 MHz, NAVTEX on 518 kHz") == [9400.0, 156.8, 0.518]
