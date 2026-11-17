import numpy as np

from mister4d import human
from mister4d.dataset import MultiViewDataset
from mister4d.human import MATERIALS, StudioView, human_at, human_pose, load_human, pose_vertices
from mister4d.synth import SynthConfig, make_prepared_dataset, make_rig


def test_asset_loads_and_is_consistent():
    h = load_human()
    assert h.verts.shape[1] == 3 and len(h.verts) > 10_000
    assert h.tris.min() == 0 and h.tris.max() == len(h.verts) - 1
    assert set(np.unique(h.material)) == set(range(len(MATERIALS)))
    assert np.allclose(h.skin_weights.sum(axis=1), 1.0, atol=1e-5)
    assert h.skin_bones.max() < len(h.bones) and (h.parents < np.arange(len(h.bones))).all()
    # Standing on the floor, about as tall as an adult, facing +X (the nose is in front).
    assert abs(h.verts[:, 2].min()) < 0.005 and 1.6 < h.verts[:, 2].max() < 1.95
    assert h.joint("eye.L")[0] > h.joint("head")[0] + 0.05 and h.joint("eye.L")[1] > 0 > h.joint("eye.R")[1]
    for m, levels in enumerate(h.textures):  # every quad has its block of texels, at every level
        quads = int((h.material == m).sum())
        for atlas, n in levels:
            assert atlas.shape[0] % n == 0 and atlas.shape[1] % n == 0
            assert atlas.shape[0] // n * (atlas.shape[1] // n) >= quads


def test_pose_keeps_the_figure_together():
    h = load_human()
    rest_edges = np.linalg.norm(h.verts[h.tris[:, 0]] - h.verts[h.tris[:, 1]], axis=1)
    tip = int(np.argmax(h.skin_bones[:, 0] == h.index["finger3-3.R"]))
    left_shoe = (h.skin_bones[:, 0] == h.index["foot.L"]) & (h.skin_weights[:, 0] > 0.99)
    previous, fastest, planted = None, 0.0, []
    for frame in range(0, 61, 2):
        v = pose_vertices(h, human_pose(h, frame / 30.0))
        assert np.isfinite(v).all()
        assert -0.012 < v[:, 2].min() < 0.012, "the feet stay on the floor"
        assert 1.6 < v[:, 2].max() < 2.2 and np.abs(v[:, :2]).max() < 1.1
        edges = np.linalg.norm(v[h.tris[:, 0]] - v[h.tris[:, 1]], axis=1)
        assert np.percentile(edges / rest_edges, 99.5) < 2.0, "skinning does not tear the mesh"
        planted.append(v[left_shoe].mean(axis=0))
        if previous is not None:
            fastest = max(fastest, float(np.linalg.norm(v[tip] - previous)) * 15.0)
        previous = v[tip]
    assert np.ptp(np.array(planted), axis=0).max() < 0.01, "the supporting foot does not slide"
    assert 2.0 < fastest < 8.0, "one fast gesture, at a human speed"
    assert np.allclose(pose_vertices(h, human_pose(h, 1.0)), pose_vertices(h, human_pose(h, 3.0)), atol=1e-5)


def test_render_mask_depth_and_determinism():
    cfg = SynthConfig(num_cameras=3, width=160, height=120)
    cam = make_rig(cfg, distorted=True).cameras["cam02"]
    view = StudioView(cam)
    image, cover, depth = view.render(0.6)
    assert image.shape == (120, 160, 3) and cover.shape == depth.shape == (120, 160)
    assert 0.0 <= image.min() and image.max() <= 1.0 and 0.0 <= cover.min() and cover.max() <= 1.0
    inside = cover > 0.99
    assert 0.05 < inside.mean() < 0.35
    rows = np.nonzero(inside.any(axis=1))[0]
    assert np.ptp(rows) > 0.55 * cam.height, "the figure fills most of the height"
    # Depth along the optical axis: around the distance of the rig, only where the subject is.
    assert (depth[inside] > cfg.rig_radius - 0.8).all() and (depth[inside] < cfg.rig_radius + 0.8).all()
    assert (depth[cover == 0] == 0).all() and (depth[inside] > 0).all()
    # The depth agrees with the geometry: project the posed vertices and compare.
    px, z = cam.project(human_at(0.6).verts)
    ij = np.round(px).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < 160) & (ij[:, 1] >= 0) & (ij[:, 1] < 120)
    seen = depth[ij[ok, 1], ij[ok, 0]]
    assert (seen[seen > 0] <= z[ok][seen > 0] + 0.03).mean() > 0.97, "nothing is behind the depth map"
    # The subject is not the background, and its shadow darkens the floor next to it.
    assert np.abs(image - view.background)[inside].max(axis=1).mean() > 0.08
    shade = (view.background - image).max(axis=2) * (cover == 0)
    assert shade.max() > 0.08 and (shade > 0.03).mean() > 0.005
    assert (shade[: cam.height // 3] < 0.01).all(), "no shadow on the wall"
    # Same input, same output, also after the caches are emptied.
    human_at.cache_clear()
    again = StudioView(cam).render(0.6)
    assert all(np.array_equal(a, b) for a, b in zip((image, cover, depth), again))


def test_prepared_dataset_with_the_human(tmp_path):
    cfg = SynthConfig(num_cameras=2, width=96, height=72, take_frames=2, subject="human")
    make_prepared_dataset(tmp_path, cfg, init_points=300)
    data = MultiViewDataset(tmp_path)
    img, mask = data.load(1, "cam02")
    assert img.shape == (72, 96, 3) and 0.04 < (mask > 127).mean() < 0.4
    plate = human.cv2.imread(str(tmp_path / "background" / "cam02.jpg"))[..., ::-1].astype(int)
    assert np.abs(img.astype(int) - plate)[mask > 250].max(axis=1).mean() > 20
