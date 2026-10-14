# 预注册：Frozen 3D-SSL baseline（TreeCLIP 补充实验）

*锁定日期：2026-07-01（结果产出前）· 作者：PriceZZY · 状态：pre-registered，结果未知*

> 本文档在**看到任何 3D-SSL 结果之前**锁定协议与报告规则，作为诚实锚点。critic 门控要求：防止（a）阉割 baseline 来赢、（b）只在结果有利时报告、（c）在多个分析变体里挑最好的。任何对本协议的偏离必须在最终稿/回复审稿人中显式记录并说明理由。

## 1. 研究问题（为什么做这个实验）
论文核心 claim 是"frozen 视觉基础模型特征是比手工几何*更强的 label-free 表征*"。审稿人最可预测、也最正当的质疑是：*唯一的竞争者只有一个手工 Weinmann 几何描述子*，而领域内真正的 label-free 竞争者——**冻结的自监督 3D 点云编码器**（Point-MAE/PointNeXt）——被引用却从未评估。

本实验补上这个 apples-to-apples 竞争者，回答：**frozen 2D 基础模型的优势，是 2D 基础模型特有的，还是任何预训练 3D 编码器都能达到甚至更好？**

## 2. 预先承诺的报告规则（不可协商）
- **无论结果如何，头对头比较都进正文**（2D-FM vs 3D-SSL vs geometry）。无论 3D 赢、平、输。
- *仅*额外的 per-source 分解 / 消融可留作 rebuttal 储备；主比较不得留后手。
- 三种结局都诚实可发：
  - 2D-FM 仍胜 → claim 显著加强（"depth-projected FM 无需 3D 预训练即胜过冻结 3D-SSL"）。
  - 2D-FM 与 3D-SSL 相当 → 诚实改写为 label-free 表征景观研究。
  - 3D-SSL 胜 → 诚实改 framing（"Foundation Model-based"标题需重写为不同但仍诚实的论文——见 §7 应急）。

## 3. 模型（冻结，不微调）
- **主**：Point-MAE，ShapeNet 预训练官方 checkpoint，**冻结**（无 fine-tune、无 probing 训练）。
- **次（时间允许）**：PointNeXt 或 PointMLP，ScanObjectNN 预训练（真实扫描，域更近 LiDAR）。
- 特征 = 冻结前向的全局池化 token（与论文对 2D 特征的池化精神一致）。
- **不阉割**：使用官方推荐输入规格、合理 checkpoint、公平预处理。故意弱化 3D baseline 与埋藏有利结果同样禁止。

## 4. 数据与对齐
- 原始点云来自本地 LAZ：FOR = `data/raw/for_species20k/dev.zip`（`{treeID}.laz`）；BioDiv = `data/raw/biodiv_3dtrees/biodiv_ULS.tar.zst`（`ULS/{treeID}_ULS.laz`）。
- 按 **treeID 对齐到现有实验的同一 analyzable 子集**：同样的 `analyzable_mask`（每物种 ≥30 树、≥2 blocks）+ BioDiv 的 Fagus/优势种 cap 0.39（与 `gen_score.py` / 现有管线逐位一致），保证与 CLIP/geo 在同一批树上比较。
- 物种标签、源（FOR `dataset` / BioDiv `region`）、传感器类型均取自现有 metadata CSV。

## 5. 预处理（预先指定，唯一变体）
- 每棵树点云：若点数 > N 随机下采样、若 < N 有放回上采样，到 **N = 1024** 点（Point-MAE 标准输入）。
- 归一化：**平移到质心 + 缩放到单位球**（zero-mean, unit max-norm）。
- 随机种子固定；下采样 RNG seed = 0。
- *不*尝试多个 N 或多个归一化再挑最好——N=1024 + 单位球是唯一预先指定方案。

## 6. 评估（完全复用现有 harness，不新造）
- 聚类：`StandardScaler → PCA(11-d) → KMeans(K=真物种数, n_init=10)`，与 `gen_permutation_test.cluster_11d` / `gen_unsup_robust` 完全相同。
- 主指标：**matching-free NMI**（15 seed，均值±SD = init-jitter）。
- 次指标（Hungarian 匹配后、显式标注为上界）：macro-F1 / OA / mIoU。
- **源控制**：与现有 `gen_source_control_*` 相同——报告 within-source 残差 NMI（去源后仍剩多少），与 CLIP/geo 并排。
- 报告表：在现有 Table 2 / Table 5 结构中加一行 3D-SSL，全数据集、全指标。

## 7. Domain-gap caveat（预先承诺的解释）
标准 Point-MAE/PointNeXt checkpoint 在合成 CAD（ShapeNet）或室内扫描（ScanObjectNN）上预训练，**对森林 LiDAR out-of-domain**。因此：
- 若 3D-SSL 输：结论限定为"**此 out-of-domain 3D 预训练**在本任务弱"，**不得**推广为"3D 预训练普遍弱于 2D-FM"。此歧义须在正文显式写明。
- 若可得，优先/补充一个植被或真实户外扫描预训练的点云 checkpoint 以缓解歧义。

## 8. "3D 赢"应急（结果前先想好，防埋藏压力）
若冻结 3D-SSL 稳定胜过 2D-FM：论文重心从"2D 基础模型"转为"冻结预训练表征（2D 或 3D）皆胜手工几何"，标题去掉"Foundation Model-based"的独占含义。这是一篇不同但同样诚实、同样可发的论文；预先承诺此路径，以消除任何埋藏不利结果的动机。

## 9. 偏离记录
- **2026-07-01 — 模型：Point-MAE → OpenShape PointBERT（冻结 3D 基础模型，ShapeNet 预训练 `openshape-pointbert-shapenet`）。** 理由：(1) HF 无 Point-MAE checkpoint，且避开 Windows 自定义 CUDA op（pointnet2_ops/knn_cuda）编译；(2) OpenShape 是对齐 CLIP 的 3D 基础模型，是比 Point-MAE SSL **更强、更相关**的竞争者——直接对标论文的"2D 基础模型"claim，用更强 baseline 属**保守/反阉割**方向（不是弱化）。grouping（FPS/kNN）用纯 torch 实现避 CUDA op；输入 xyz（+ 常数颜色，OpenShape 无色时的标准做法）。domain-gap caveat 不变（ShapeNet CAD 预训练 → 对森林 LiDAR out-of-domain）。§2 报告规则、§6 评估 harness、§7 caveat 全部不变。Fallback ladder（若加载失败）：Uni3D → 纯 torch 移植 Point-MAE。所有结局仍按 §2 无论输赢入正文。
