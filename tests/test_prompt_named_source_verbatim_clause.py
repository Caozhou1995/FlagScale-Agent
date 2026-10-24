"""Anti-softening tests for the NAMED-SOURCE VERBATIM RECORD clause (lever v12).

Root cause it guards: protein-assembly v12 batch — agent reconstructed the donor
sequence by reasoning (chromophore analysis) instead of fetching the named PDB API
record; checker's literal find() returned -1. The clause must state:
  1. deliverable-embedded sequences = named source's own record, fetched verbatim
  2. tags/signal peptides/purification structure included (no tidied variant)
  3. reasoning is for IDENTIFYING the record, never for WRITING its content
  4. fetched record beats reconstruction on disagreement
"""
import re
from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"


def _clause_text() -> str:
    src = PROMPT.read_text()
    m = re.search(r"A sequence or record your deliverable EMBEDS.*?(?=\n\n)", src, re.S)
    assert m, "named-source verbatim record clause not found in prompt.py"
    return m.group(0)


def test_clause_present_with_all_four_mechanism_rules():
    c = _clause_text()
    assert "NAMED SOURCE" in c
    assert "fetched verbatim" in c
    # rule 2: tags included
    assert "INCLUDING leading tags" in c
    # rule 3: identify vs write split
    assert "legitimate for IDENTIFYING" in c
    assert "never for WRITING" in c
    # rule 4: record beats reconstruction
    assert "the record wins" in c


def test_clause_has_falsifiable_consumer_shape():
    c = _clause_text()
    # the mechanism that decides: literal substring find()
    assert "literal substring find()" in c
    assert "fails find()" in c


def test_clause_no_task_literals():
    src = PROMPT.read_text()
    clause_start = src.find("A sequence or record your deliverable EMBEDS")
    m = re.search(r"A sequence or record your deliverable EMBEDS.*?(?=\n\n)", src[clause_start:], re.S)
    c = m.group(0)
    for banned in ("protein", "pdb", "fasta", "chromophore", "donor", "Clover",
                   "DYKDDDDK", "antibody", "gblock", "plasmid"):
        assert banned.lower() not in c.lower(), f"task literal leaked: {banned}"
