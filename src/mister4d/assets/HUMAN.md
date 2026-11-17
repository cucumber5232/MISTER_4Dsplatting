# human.npz

The clothed figure rendered by `mister4d/human.py` for synthetic capture
sessions (`mister4d demo --subject human`). It is only a test subject: it is
never used on real footage.

## Source and licence

Built by `scripts/build_human_asset.py` from the MakeHuman repository
(https://github.com/makehumancommunity/makehuman, checked out in November
2026). MakeHuman releases its assets (base mesh, targets, skeletons, weights)
under **CC0 1.0 Universal**: see `LICENSE.md` (section C) and
`LICENSE.ASSETS.md` in that repository, and the header of each file. The
MakeHuman program itself is AGPL: none of its code is used or included.

Files read from `makehuman/data/`, all marked CC0 in their headers:

| File | What is taken from it |
|---|---|
| `3dobjs/base.obj` | body mesh `hm08`; helper geometry: the full-body suit (`helper-tights`), the eyeballs, the joint markers |
| `targets/macrodetails/caucasian-male-young.target` | the shape of an adult man |
| `rigs/default.mhskel` | bones and their joints |
| `rigs/default_weights.mhw` | skinning weights |

No MakeHuman texture, clothing, hair or pose file is used (most of them are
not in that repository). Everything else is made by the build script:

- sweatshirt, trousers and shoes are the suit helper cut in three, moved off
  the body, subdivided and shaped; the hair is the top of the scalp moved
  outwards;
- every texture is painted by the script from 3D position and rig weights
  (skin tones, brows, lips, eyes, stripes and emblem of the top, denim, soles
  and laces); no image comes from elsewhere;
- ambient occlusion is computed from the geometry.

The result is our own work on CC0 data and is distributed with the package
under the licence of the repository.

## Contents

`verts`, `quads`, `material` (index into `human.MATERIALS`), `ao`, the
skeleton (`bones`, `parents`, `heads`), four skinning weights per vertex
(`skin_bones`, `skin_weights`) and one PNG atlas per material (`texture0`...)
in which every quad owns a block of `texture_block` x `texture_block` texels.
Units are metres, +Z up, the figure faces +X and stands on z = 0.

## Rebuild

    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/makehumancommunity/makehuman
    python scripts/build_human_asset.py makehuman
