"""What a Mac would do with a file is worked out by the engine's own rule, and says what that rule leaves out."""

from types import SimpleNamespace

import numpy as np
import pytest

from strata_mlx import check, dense, paged, serve
from strata_mlx.check import GB, KNOWN, Model, lines, of, plan
from strata_mlx.gguf import Tensor

Q2, IQ3S = KNOWN


@pytest.mark.parametrize("model, gb, what", [
    (Q2, 8, "too small"), (Q2, 16, "too small"), (Q2, 32, "in part"), (Q2, 48, "in part"),
    (Q2, 64, "held whole in memory:"), (Q2, 128, "held whole in memory:"), (IQ3S, 16, "too small"),
    (IQ3S, 48, "in part"), (IQ3S, 64, "in part"), (IQ3S, 81, "held whole in memory, without"),
    (IQ3S, 96, "held whole in memory:")])
def test_what_a_mac_of_some_memory_does_with_each_file(model, gb, what):
    assert what in lines(model, gb * GB)[1]


@pytest.mark.parametrize("model", KNOWN)
def test_whole_is_what_the_engine_decides(model):
    for gb in range(8, 130):
        assert plan(model, gb * GB).whole == paged.all_fit(gb * GB, model.experts, model.dense)


def test_the_experts_kept_are_about_what_a_start_measured():
    # ROADMAP: Q2_0 with the draft head, started as a machine of 16, 24 and 32 GB, gave its experts this much
    # (1.5, 10.1 and 18.7 GB before its dense tensors of two types stayed blocks). macOS was left 4.5 GB then and
    # is left 8.5 now (#6) or a quarter of the memory where that is more, and the working memory was 4 GB and is 5
    # (prompts read a layer at a time), so the experts get 5.0 GB less and that quarter's excess: on a 16 GB machine
    # less than the least there is
    for gb, measured in ((16, 3.3e9), (24, 11.9e9), (32, 20.5e9)):
        excess = paged.left_to_system(gb * GB) - paged.SYSTEM
        assert abs(plan(Q2, gb * GB).kept - (measured - 5.0e9 - excess)) < 0.4e9
    assert plan(Q2, 16 * GB).kept < paged.LEAST <= plan(Q2, 24 * GB).kept
    assert Q2.spared == pytest.approx(1.74e9, rel=0.01)  # each of those starts gave the experts that much more


def test_floats_for_prompts_are_kept_only_where_they_fit():
    tight = plan(IQ3S, 81 * GB)  # no Mac has 81 GB: the rule holds the file whole from 80, with its floats 83.7
    assert tight.whole and not tight.floats and tight.needs - 81 * GB == pytest.approx(2.7 * GB, abs=0.05 * GB)
    assert "a tenth slower" in "\n".join(lines(IQ3S, 81 * GB))
    for model in KNOWN:
        for gb in range(8, 130):
            p = plan(model, gb * GB)
            assert p.floats == (p.needs <= gb * GB) and (p.whole or not p.floats)
    assert not plan(IQ3S, 64 * GB).whole and plan(IQ3S, 96 * GB).needs < 96 * GB
    assert plan(Q2, 64 * GB).needs == pytest.approx(57.9 * GB, abs=0.05 * GB)
    assert "not run this way yet" in "\n".join(lines(IQ3S, 48 * GB))
    assert "not run this way yet" not in "\n".join(lines(Q2, 24 * GB))


def test_sizes_are_read_from_a_models_files(tmp_path):
    def t(name, shape, kind, nbytes):
        return Tensor(name, shape, kind, np.zeros(nbytes, dtype=np.uint8))

    first = {t.name: t for t in (
        t("blk.0.ffn_gate_exps.weight", (4, 64, 128), "Q2_0", 1000),
        t("blk.0.attn_k.weight", (512, 2560), "Q6_K", 300),  # a chunk of a prompt multiplies it as floats
        t("output.weight", (248320, 2560), "Q6_K", 700),  # no chunk reaches the output layer
        t("blk.0.odd.weight", (6, 512), "Q6_K", 50),  # not a shape the kernel takes: floats from the start
        t("token_embd.weight", (10, 16), "Q4_K", 20))}
    second = {t.name: t for t in (t("blk.1.ffn_up_exps.weight", (4, 64, 128), "IQ3_S", 500),
                                  t("per_layer_token_embd.weight", (9, 9), "IQ4_NL", 9999))}  # stays on the disk
    paths = [tmp_path / f"some-Q9_Z-0000{i}-of-00002.gguf" for i in (1, 2)]
    for path, size in zip(paths, (11, 22)):
        path.write_bytes(b"0" * size)
    files = [SimpleNamespace(tensors=tensors, path=str(path)) for tensors, path in zip((first, second), paths)]
    m = of(str(paths[0]), files)
    assert (m.name, m.experts, m.dense, m.floats, m.disk) == ("Q9_Z", 1500, 1070, 2 * 512 * 2560, 33)
    assert m.spared == 0  # none of these is of a type kept as blocks


def test_blocks_spare_what_the_arrays_of_their_tensors_take_more(tmp_path):
    def t(name, shape, kind):
        per_block, block_bytes = {"Q3_K": (256, 110), "IQ4_XS": (256, 136), "Q4_K": (256, 144)}[kind]
        return Tensor(name, shape, kind, np.zeros(int(np.prod(shape)) // per_block * block_bytes, dtype=np.uint8))

    tensors = {t.name: t for t in (
        t("token_embd.weight", (8, 2560), "Q3_K"),  # a byte and an eighth a weight as arrays, 110 bytes to 256 here
        t("blk.0.attn_v.weight", (4, 2560), "Q3_K"),  # a layer's Q3_K as planes: 114 bytes to 256
        t("blk.0.attn_q.weight", (4, 2560), "IQ4_XS"),
        t("blk.0.attn_k.weight", (4, 2560), "Q4_K"),  # as arrays either way
        t("output.weight", (8, 2560), "Q3_K"),  # and so is the output layer, whatever its type
        t("blk.0.ffn_up_exps.weight", (2, 4, 2560), "IQ4_XS"))}  # experts are blocks already
    assert dense.spared(tensors) == (8 * 2560 // 256 * (288 - 110) + 4 * 2560 // 256 * (288 - 114)
                                     + 4 * 2560 // 256 * (288 - 136))
    path = tmp_path / "some-Q9_Z-00001-of-00001.gguf"
    path.write_bytes(b"0")
    m = of(str(path), [SimpleNamespace(tensors=tensors, path=str(path))])
    whole_rule = plan(Model("x", m.experts, m.dense, 0, 0), 16 * GB)
    assert plan(m, 16 * GB).kept - whole_rule.kept == m.spared == dense.spared(tensors)
    assert plan(m, 16 * GB).least == whole_rule.least - m.spared
    assert plan(m, 16 * GB).needs == whole_rule.needs  # a Mac that holds the file whole holds arrays


def test_the_report_is_of_this_mac_or_of_the_one_asked_about(capsys):
    assert check.main(["--machine-memory", "48"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("A Mac with 48 GB of memory")
    assert "Q2_0, 66.4 GB on disk" in out and "IQ3_S, 83.6 GB on disk" in out
    assert "Free on the disk" not in out  # only this Mac's disk can be looked at
    assert check.main([]) == 0
    out = capsys.readouterr().out
    assert out.startswith("This Mac has") and "Free on the disk" in out


def test_parked_conversations_are_memory_the_model_does_not_have(capsys):
    assert check.main(["--machine-memory", "64", "--conversation-cache-mib", "8192"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("A Mac with 64 GB of memory") and "8.0 GB of parked conversations" in out
    assert lines(Q2, 56 * GB)[1] in out and lines(Q2, 64 * GB)[1] not in out


def test_the_launcher_checks_instead_of_serving(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["strata-mlx", "--serve", "--check", "--machine-memory", "16", "--spec", "1"])
    assert serve.main() == 0
    assert capsys.readouterr().out.startswith("A Mac with 16 GB of memory")
