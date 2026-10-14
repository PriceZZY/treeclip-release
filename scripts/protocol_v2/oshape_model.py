"""Self-contained OpenShape PointBERT (PPAT) model for the frozen 3D-FM baseline.

Contains code from OpenShape (https://github.com/Colin97/OpenShape_code), licensed under the
Apache License 2.0; see LICENSES/OpenShape-Apache-2.0.txt. The two upstream files were last changed in
commit 010609ab1e5b (2023-06-27); OpenShape ships no NOTICE file. The changes from upstream are listed below.

Faithful re-vendoring of the OpenShape architecture (github.com/Colin97/OpenShape_code,
src/models/ppat.py + src/models/pointnet_util.py) with TWO surgical changes, both
non-weakening and required by the pre-registration (PREREG_3d_baseline.md §9):

  1. farthest_point_sample: OpenShape calls dgl.geometry.farthest_point_sampler (a CUDA
     op we deliberately avoid on Windows). We use OpenShape's OWN pure-torch FPS fallback
     (the code right below the dgl call in the upstream file), with a deterministic start
     index (0) for reproducibility. Same algorithm, no custom CUDA op.
  2. torch_redstone (rst.supercat / rst.Lambda): used verbatim from the installed
     pure-python package (0.0.6) — bit-identical to upstream, no CUDA.

Everything else (PointNetSetAbstraction ball-query grouping, lift, transformer, proj) is
copied verbatim from OpenShape. Config = scaling==1 =
    PointPatchTransformer(dim=256, depth=6, heads=4, mlp_dim=1024, sa_dim=96,
                           patches=64, prad=0.4, nsamp=256, in_dim=6)
    + Linear(256, 1280)
matching checkpoint OpenShape/openshape-pointbert-shapenet (state_dict under 'state_dict',
'module.' prefixed).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_redstone as rst
from einops import rearrange


# ---------------- pointnet_util (verbatim, pure-torch FPS) ----------------
def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint, start=0):
    """Pure-torch FPS (OpenShape's own fallback), deterministic start index for repro."""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.full((B,), start, dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[..., :1].repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


# ---------------- ppat (verbatim) ----------------
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *extra_args, **kwargs):
        return self.fn(self.norm(x), *extra_args, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., rel_pe=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()
        self.rel_pe = rel_pe
        if rel_pe:
            self.pe = nn.Sequential(nn.Conv2d(3, 64, 1), nn.ReLU(), nn.Conv2d(64, 1, 1))

    def forward(self, x, centroid_delta):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        pe = self.pe(centroid_delta) if self.rel_pe else 0
        dots = (torch.matmul(q, k.transpose(-1, -2)) + pe) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., rel_pe=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, rel_pe=rel_pe)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]))

    def forward(self, x, centroid_delta):
        for attn, ff in self.layers:
            x = attn(x, centroid_delta) + x
            x = ff(x) + x
        return x


class PointPatchTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, sa_dim, patches, prad, nsamp,
                 in_dim=3, dim_head=64, rel_pe=False, patch_dropout=0):
        super().__init__()
        self.patches = patches
        self.patch_dropout = patch_dropout
        self.sa = PointNetSetAbstraction(npoint=patches, radius=prad, nsample=nsamp,
                                         in_channel=in_dim + 3, mlp=[64, 64, sa_dim], group_all=False)
        self.lift = nn.Sequential(nn.Conv1d(sa_dim + 3, dim, 1),
                                  rst.Lambda(lambda x: torch.permute(x, [0, 2, 1])),
                                  nn.LayerNorm([dim]))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, 0.0, rel_pe)

    def forward(self, xyz: torch.Tensor, features):
        self.sa.npoint = self.patches
        if self.training:
            self.sa.npoint -= self.patch_dropout
        centroids, feature = self.sa(xyz, features)
        x = self.lift(torch.cat([centroids, feature], dim=1))
        x = rst.supercat([self.cls_token, x], dim=-2)
        centroids = rst.supercat([centroids.new_zeros(1), centroids], dim=-1)
        centroid_delta = centroids.unsqueeze(-1) - centroids.unsqueeze(-2)
        x = self.transformer(x, centroid_delta)
        return x[:, 0]


class Projected(nn.Module):
    def __init__(self, ppat, proj):
        super().__init__()
        self.ppat = ppat
        self.proj = proj

    def forward(self, xyz: torch.Tensor, features: torch.Tensor):
        # returns dict of both the pre-projection pooled token and the projected embedding
        pooled = self.ppat(xyz.transpose(-1, -2).contiguous(), features.transpose(-1, -2).contiguous())
        projected = self.proj(pooled)
        return pooled, projected


def build_openshape_pointbert(out_channel=1280, in_channel=6):
    """scaling==1 config, matching openshape-pointbert-shapenet."""
    return Projected(
        PointPatchTransformer(256, 6, 4, 1024, 96, 64, 0.4, 256, in_channel),
        nn.Linear(256, out_channel),
    )


def load_checkpoint(model, ckpt_path, map_location="cpu"):
    ck = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    stripped = {}
    for k, v in sd.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        stripped[nk] = v
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    return missing, unexpected
