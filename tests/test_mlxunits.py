"""The MLX versions of Neuron and Erosion (mlxunits.py) against the VAU units they stand in for.
Runs where MLX is installed: .venv-mrt2/bin/python -m pytest tests/test_mlxunits.py"""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")
vau = pytest.importorskip("vau")

from vau import hostfx  # noqa: E402
from vau.units import erosion as V_erosion  # noqa: E402
from vau.units import neuron as V_neuron  # noqa: E402

from abliterator import mlxunits, units  # noqa: E402

RNG = np.random.default_rng(3)


def close(a_mx, b_t, tol=1e-5):
    a = np.array(a_mx.astype(mx.float32)).reshape(-1)
    b = b_t.float().numpy().reshape(-1)
    assert np.allclose(a, b, atol=tol, rtol=1e-5), np.abs(a - b).max()


@pytest.mark.parametrize("op,value", [("scale", 6.0), ("scale", -1.5), ("add", 2.0), ("set", 0.7), ("zero", 0), ("flip", 0), ("clamp", 0.3)])
@pytest.mark.parametrize("f,keepnorm", [(1.0, False), (0.4, False), (1.0, True), (0.6, True)])
def test_neuron_matches_vau(op, value, f, keepnorm):
    ns = [3, 17, 200]
    h = RNG.standard_normal(256).astype(np.float32)
    ref = V_neuron.neuron_hook(torch.tensor(ns), op, value)
    if f != 1.0 or keepnorm:
        ref = hostfx.wrap_input(ref, f, keepnorm)
    want = ref(None, (torch.from_numpy(h).reshape(1, 1, -1),))[0]
    got = mlxunits.neuron_fn(ns, op, value, f, keepnorm)(mx.array(h).reshape(1, 1, -1))
    close(got, want)


@pytest.mark.parametrize("bits,hold", [(6, 1), (3, 1), (16, 3), (4, 4)])
@pytest.mark.parametrize("f,keepnorm", [(1.0, False), (0.5, True)])
def test_erosion_matches_vau(bits, hold, f, keepnorm):
    std = np.abs(RNG.standard_normal(64)) + 0.2
    step = None if bits >= 16 else std * (6.0 / 2 ** bits)
    ref = V_erosion.erosion_edit(None if step is None else torch.tensor(step), hold, {})
    if f != 1.0 or keepnorm:
        ref = hostfx.wrap_edit(ref, f, keepnorm)
    mine = mlxunits.erosion_fn(step, hold, {}, f, keepnorm)
    for frame in range(0, 12):
        h = RNG.standard_normal((1, 1, 64)).astype(np.float32)
        want = ref(torch.from_numpy(h), torch.tensor([frame]))
        got = mine(mx.array(h), frame)
        close(got, want, tol=1e-4)


class _Stats:
    def stats(self):
        return {"std": np.abs(RNG.standard_normal((13, 64))) + 0.1}


class _Comp:
    def __init__(self, out, inp, name):
        self.weight = torch.zeros(out, inp)
        self.bias = None
        self.name = name


def _attach(lines, native):
    comps = {(b, k): _Comp(64, 256 if k == "mlp" else 64, f"b{b}.{k}") for b in range(12) for k in ("attn", "mlp")}
    return units.attach(lines, (12, _Stats(), {}, lambda: 5, comps), strict=True, native=native)[0]


def test_attach_ports_only_what_it_can():
    b = _attach(["neuron block=all n=3 op=scale value=6", "erosion layer=6 bits=6", "saturator layer=4 drive=2"], True)
    assert all(getattr(f, "mlx", False) for fs in b["ins"].values() for f in fs) and len(b["ins"]) == 12
    assert all(getattr(f, "mlx", False) for f in b["edits"][6])
    assert not any(getattr(f, "mlx", False) for f in b["edits"][4])          # Saturator stays in torch
    t = _attach(["neuron block=all n=3 op=scale value=6"], False)
    assert not any(getattr(f, "mlx", False) for fs in t["ins"].values() for f in fs)
    w = _attach(["erosion layer=6 bits=6 wfrom=3"], True)                       # a window keeps the torch path
    assert not any(getattr(f, "mlx", False) for f in w["edits"][6])
