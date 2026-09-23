# real_st_loader.py — 空间转录组真实感数据生成器
# 生成具有六边形网格、区域结构、LR信号热点的配对切片

import numpy as np
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

SEED = 42

LR_DB = {
    ("Ntng1", "Ntrk2"): 0.95, ("Bdnf", "Ntrk2"): 0.92,
    ("Ntf3", "Ntrk3"): 0.90, ("Ngf", "Ntrk1"): 0.88,
    ("Gdnf", "Gfra1"): 0.85, ("Artn", "Gfra3"): 0.80,
    ("Nrtn", "Gfra2"): 0.78, ("Fgf8", "Fgfr1"): 0.90,
    ("Fgf15", "Fgfr2"): 0.85, ("Fgf10", "Fgfr2"): 0.82,
    ("Fgf2", "Fgfr1"): 0.88, ("Egf", "Egfr"): 0.85,
    ("Hgf", "Met"): 0.83, ("Vegfa", "Kdr"): 0.90,
    ("Vegfb", "Flt1"): 0.78, ("Pdgfa", "Pdgfra"): 0.80,
    ("Pdgfb", "Pdgfrb"): 0.82, ("Igf1", "Igf1r"): 0.85,
    ("Igf2", "Igf1r"): 0.80, ("Wnt3a", "Fzd1"): 0.88,
    ("Wnt5a", "Fzd5"): 0.85, ("Wnt7a", "Fzd10"): 0.82,
    ("Wnt1", "Fzd1"): 0.80, ("Dll1", "Notch1"): 0.90,
    ("Dll4", "Notch4"): 0.85, ("Jag1", "Notch1"): 0.88,
    ("Jag2", "Notch2"): 0.83, ("Efnb1", "Ephb2"): 0.90,
    ("Efna1", "Epha4"): 0.85, ("Efnb2", "Ephb4"): 0.82,
    ("Efna5", "Epha3"): 0.78, ("Sema3a", "Nrp1"): 0.90,
    ("Sema3f", "Nrp2"): 0.85, ("Sema4d", "Plxnb1"): 0.80,
    ("Sema6a", "Plxna2"): 0.75, ("Slit1", "Robo1"): 0.88,
    ("Slit2", "Robo2"): 0.85, ("Slit3", "Robo2"): 0.80,
    ("Cxcl12", "Cxcr4"): 0.95, ("Cxcl13", "Cxcr5"): 0.85,
    ("Ccl2", "Ccr2"): 0.82, ("Ccl5", "Ccr5"): 0.80,
    ("Il1b", "Il1r1"): 0.90, ("Tnf", "Tnfrsf1a"): 0.88,
    ("Tgfb1", "Tgfbr1"): 0.92, ("Tgfb2", "Tgfbr2"): 0.88,
    ("Bmp4", "Bmpr1a"): 0.85, ("Bmp7", "Bmpr2"): 0.80,
    ("Bmp2", "Acvr1"): 0.82, ("Cdh1", "Cdh1"): 0.95,
    ("Cdh2", "Cdh2"): 0.92, ("Ncam1", "Ncam1"): 0.85,
    ("L1cam", "L1cam"): 0.80, ("Lama1", "Itga1"): 0.78,
    ("Col1a1", "Itga2"): 0.75, ("Vtn", "Itgav"): 0.72,
    ("Shh", "Ptch1"): 0.90, ("Ihh", "Ptch2"): 0.82,
    ("Dhh", "Ptch1"): 0.78, ("Reln", "Lrp8"): 0.90,
    ("Reln", "Vldlr"): 0.85,
}

def lr_gene_list(lr_db=None):
    if lr_db is None:
        lr_db = LR_DB
    ligands = sorted(set(k[0] for k in lr_db))
    receptors = sorted(set(k[1] for k in lr_db))
    return ligands, receptors

LIGANDS, RECEPTORS = lr_gene_list(LR_DB)
ALL_LR_GENES = sorted(set(LIGANDS + RECEPTORS))  # sorted: 避免 PYTHONHASHSEED 跨进程顺序不同
N_REGIONS = 6

REGION_NAMES = ["Cortex", "Hippocampus", "Thalamus", "Hypothalamus", "Striatum", "Cerebellum"]

@dataclass
class STSlice:
    """空间转录组切片数据结构"""
    coords: np.ndarray
    expr: np.ndarray
    region_labels: np.ndarray
    gene_names: np.ndarray
    n_spots: int
    n_genes: int

    def to_dict(self):
        """转换为字典格式（兼容lrot_core.py的slice_A/slice_B接口）"""
        return {
            'coords': self.coords,
            'expr': self.expr,
            'gene_names': list(self.gene_names),
            'region_labels': self.region_labels,
            'n_spots': self.n_spots,
            'n_genes': self.n_genes,
        }

@dataclass
class PairedSTData:
    """配对切片数据"""
    slice_A: STSlice
    slice_B: STSlice
    transform_params: Dict

class RealisticSTGenerator:
    """真实感空间转录组数据生成器
    生成具有六边形网格、区域结构、LR信号热点的配对切片
    """

    def __init__(self, n_spots_A=400, n_spots_B=400, n_genes=200,
                 n_regions=6, lr_db=None, seed=SEED):
        self.n_spots_A = n_spots_A
        self.n_spots_B = n_spots_B
        self.n_genes = n_genes
        self.n_regions = n_regions
        self.lr_db = lr_db if lr_db is not None else LR_DB
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def _generate_hex_grid(self, n_spots, jitter=0.05):
        """生成六边形网格坐标"""
        side = int(np.ceil(np.sqrt(n_spots)))
        xs, ys = [], []
        for i in range(side):
            for j in range(side):
                x = i * np.sqrt(3)
                y = j * 1.5 + (0.75 if i % 2 == 1 else 0.0)
                xs.append(x)
                ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        if len(xs) > n_spots:
            idx = self.rng.choice(len(xs), n_spots, replace=False)
        else:
            idx = np.arange(len(xs))
        coords = np.column_stack([xs[idx], ys[idx]])
        coords += self.rng.randn(n_spots, 2) * jitter
        return coords

    def _generate_region_profiles(self, n_regions, n_genes):
        """生成各区域的基因表达谱"""
        profiles = np.zeros((n_regions, n_genes))
        n_marker = max(n_genes // (n_regions * 2), 5)
        for r in range(n_regions):
            marker_genes = self.rng.choice(n_genes, n_marker, replace=False)
            profiles[r, marker_genes] = self.rng.uniform(2.0, 5.0, n_marker)
            bg = np.setdiff1d(np.arange(n_genes), marker_genes)
            profiles[r, bg] = self.rng.uniform(0.1, 0.5, len(bg))
        # 区域间共享部分marker基因（模拟组织连续性）
        for r in range(n_regions - 1):
            overlap = self.rng.choice(n_genes, n_marker // 3, replace=False)
            profiles[r + 1, overlap] += profiles[r, overlap] * 0.3
        return profiles

    def _assign_spots_to_regions(self, coords):
        """基于空间临近分配区域标签"""
        n_spots = coords.shape[0]
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        mx = (x_max - x_min) * 0.15
        my = (y_max - y_min) * 0.15
        centroids = np.column_stack([
            self.rng.uniform(x_min + mx, x_max - mx, self.n_regions),
            self.rng.uniform(y_min + my, y_max - my, self.n_regions),
        ])
        dists = cdist(coords, centroids)
        region_labels = dists.argmin(axis=1)
        return region_labels, centroids

    def _generate_expression(self, coords, region_labels, region_profiles, gene_names):
        """基于Gamma-Poisson生成表达矩阵"""
        n_spots = coords.shape[0]
        n_genes = len(gene_names)
        expr = np.zeros((n_spots, n_genes))
        for r in range(self.n_regions):
            mask = region_labels == r
            n_in = mask.sum()
            if n_in == 0:
                continue
            base = region_profiles[r]
            for s in np.where(mask)[0]:
                shape = 5.0
                scale = base / shape
                gamma_sample = self.rng.gamma(shape, scale)
                expr[s] = self.rng.poisson(gamma_sample + 1e-8)
        expr = np.log1p(expr)
        return expr

    def _add_lr_hotspots(self, coords, expr, region_labels, lr_gene_indices,
                         hotspot_fraction=0.15):
        """添加配体-受体信号热点区域"""
        n_spots = coords.shape[0]
        n_hotspots = max(int(n_spots * hotspot_fraction), 5)
        hotspot_indices = self.rng.choice(n_spots, n_hotspots, replace=False)
        n_center = min(3, n_hotspots // 5)
        center_idx = hotspot_indices[:n_center]
        for cidx in center_idx:
            c_coord = coords[cidx:cidx + 1]
            dists = cdist(c_coord, coords)[0]
            radius = np.percentile(dists, 15)
            local_mask = dists < radius
            for gi in lr_gene_indices:
                expr[local_mask, gi] *= self.rng.uniform(1.5, 3.0)
        return expr

    def _add_batch_effects(self, expr, scale_range=(0.8, 1.2)):
        """添加批次效应"""
        scales = self.rng.uniform(scale_range[0], scale_range[1], expr.shape[0])
        expr = expr * scales[:, np.newaxis]
        return expr

    def apply_transform(self, coords, rotation_deg=5.0, shear=0.02,
                        scale=0.95, warp_strength=0.1):
        """对坐标施加非线性变换（模拟真实切片间变形）"""
        center = coords.mean(axis=0)
        coords_c = coords - center
        theta = np.deg2rad(rotation_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        S = np.array([[scale, 0], [0, scale]])
        Sh = np.array([[1, shear], [shear, 1]])
        M = R @ S @ Sh
        coords_t = coords_c @ M.T
        # 平滑随机扭曲（模拟组织变形）
        warp = self.rng.randn(coords.shape[0], 2) * warp_strength
        dist_mat = cdist(coords, coords)
        sigma = np.percentile(dist_mat, 20)
        weights = np.exp(-dist_mat ** 2 / (2 * sigma ** 2))
        weights = weights / weights.sum(axis=1, keepdims=True)
        smooth_warp = weights @ warp
        coords_t += smooth_warp + center
        return coords_t

    def generate_paired_slices(self, rotation=5.0, shear=0.02, scale=0.95,
                               batch_effect=0.15):
        """生成一对配对的ST切片"""
        ligs, recs = lr_gene_list(self.lr_db)
        all_lr_genes = sorted(set(ligs + recs))  # sorted: 避免 PYTHONHASHSEED 跨进程顺序不同
        n_lr = min(len(all_lr_genes), self.n_genes // 3)
        n_other = self.n_genes - n_lr
        other_genes = [f"Gene_{i}" for i in range(n_other)]
        selected_lr = all_lr_genes[:n_lr]
        gene_names = selected_lr + other_genes
        self.n_genes = len(gene_names)
        lr_gene_indices = [i for i, g in enumerate(gene_names) if g in all_lr_genes]

        # 切片A
        coords_A = self._generate_hex_grid(self.n_spots_A)
        region_profiles = self._generate_region_profiles(self.n_regions, self.n_genes)
        region_labels_A, centroids = self._assign_spots_to_regions(coords_A)
        expr_A = self._generate_expression(coords_A, region_labels_A,
                                           region_profiles, gene_names)
        expr_A = self._add_lr_hotspots(coords_A, expr_A, region_labels_A,
                                       lr_gene_indices, 0.15)

        slice_A = STSlice(
            coords=coords_A, expr=expr_A, region_labels=region_labels_A,
            gene_names=np.array(gene_names),
            n_spots=self.n_spots_A, n_genes=self.n_genes
        )

        # 切片B（对A施加变换）
        coords_B = self.apply_transform(coords_A.copy(),
                                        rotation_deg=rotation,
                                        shear=shear, scale=scale)
        region_labels_B, _ = self._assign_spots_to_regions(coords_B)
        expr_B = self._generate_expression(coords_B, region_labels_B,
                                           region_profiles, gene_names)
        expr_B = self._add_lr_hotspots(coords_B, expr_B, region_labels_B,
                                       lr_gene_indices, 0.12)
        expr_B = self._add_batch_effects(expr_B,
                                         (1.0 - batch_effect,
                                          1.0 + batch_effect))

        slice_B = STSlice(
            coords=coords_B, expr=expr_B, region_labels=region_labels_B,
            gene_names=np.array(gene_names),
            n_spots=self.n_spots_B, n_genes=self.n_genes
        )

        tparams = {
            'rotation_deg': rotation, 'shear': shear,
            'scale': scale, 'batch_effect': batch_effect
        }
        return PairedSTData(slice_A=slice_A, slice_B=slice_B,
                            transform_params=tparams)

    def print_info(self, paired_data):
        """打印配对数据摘要"""
        A = paired_data.slice_A
        B = paired_data.slice_B
        tparams = paired_data.transform_params
        print("=" * 60)
        print("  RealisticSTGenerator - 配对数据摘要")
        print("=" * 60)
        print(f"  Slice A: {A.n_spots} spots x {A.n_genes} genes")
        print(f"    Coords: x=[{A.coords[:,0].min():.2f},{A.coords[:,0].max():.2f}], "
              f"y=[{A.coords[:,1].min():.2f},{A.coords[:,1].max():.2f}]")
        print(f"    Regions: {len(np.unique(A.region_labels))}")
        print(f"    Expr range: [{A.expr.min():.4f},{A.expr.max():.4f}], "
              f"mean={A.expr.mean():.4f}")
        print(f"  Slice B: {B.n_spots} spots x {B.n_genes} genes")
        print(f"    Coords: x=[{B.coords[:,0].min():.2f},{B.coords[:,0].max():.2f}], "
              f"y=[{B.coords[:,1].min():.2f},{B.coords[:,1].max():.2f}]")
        print(f"    Regions: {len(np.unique(B.region_labels))}")
        print(f"    Expr range: [{B.expr.min():.4f},{B.expr.max():.4f}], "
              f"mean={B.expr.mean():.4f}")
        print(f"  Transform: rot={tparams['rotation_deg']} deg, "
              f"shear={tparams['shear']}, scale={tparams['scale']}")
        print(f"  Batch effect: {tparams['batch_effect']}")
        print("=" * 60)


if __name__ == "__main__":
    gen = RealisticSTGenerator(n_spots_A=300, n_spots_B=300,
                               n_genes=150, seed=42)
    data = gen.generate_paired_slices(rotation=5, shear=0.02,
                                      scale=0.95, batch_effect=0.15)
    gen.print_info(data)
