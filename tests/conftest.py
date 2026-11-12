import pytest

from mister4d.synth import SynthConfig, make_prepared_dataset


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory):
    """4 cameras, 5 frames, 96x72 px: small enough to train on CPU in seconds."""
    out = tmp_path_factory.mktemp("tiny")
    cfg = SynthConfig(num_cameras=4, width=96, height=72, take_frames=5, noise=1.0)
    make_prepared_dataset(out, cfg, init_points=400)
    return out
