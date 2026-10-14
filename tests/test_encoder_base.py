import numpy as np
import torch
from treeclip.encoders.base import Encoder, EmbeddingBundle


class DummyEncoder(Encoder):
    name = "dummy"
    embedding_dim = 4

    def encode(self, sample):
        return torch.tensor([1.0, 2.0, 3.0, 4.0])


def test_encoder_base_contract():
    enc = DummyEncoder()
    assert enc.name == "dummy"
    assert enc.embedding_dim == 4


def test_embedding_bundle_npz_roundtrip(tmp_path):
    eb = EmbeddingBundle(
        tree_id="t001", site="OSBS", species="PIPA2",
        z_struct=np.array([1.0, 2.0], dtype=np.float32),
        z_rgb=np.array([0.5, 0.5], dtype=np.float32),
        z_spec=np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )
    path = tmp_path / "t001.npz"
    eb.save_npz(path)
    eb2 = EmbeddingBundle.load_npz(path)
    assert eb2.tree_id == "t001"
    assert eb2.species == "PIPA2"
    np.testing.assert_array_equal(eb2.z_struct, eb.z_struct)
    np.testing.assert_array_equal(eb2.z_rgb, eb.z_rgb)
    np.testing.assert_array_equal(eb2.z_spec, eb.z_spec)


def test_embedding_bundle_handles_missing_modality(tmp_path):
    eb = EmbeddingBundle(
        tree_id="t002", site="MLBS", species=None,
        z_struct=None, z_rgb=np.array([0.1], dtype=np.float32), z_spec=None,
    )
    path = tmp_path / "t002.npz"
    eb.save_npz(path)
    eb2 = EmbeddingBundle.load_npz(path)
    assert eb2.z_struct is None
    assert eb2.z_spec is None
    assert eb2.species is None
    np.testing.assert_array_equal(eb2.z_rgb, eb.z_rgb)
