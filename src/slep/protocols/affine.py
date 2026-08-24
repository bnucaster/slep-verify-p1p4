"""仿射律拟合与温度辨识的统计量（CAL-P4 与 EXP-P4 用）。

只计算统计量；通过 / 失败的判定属 judge 代码（任务七冻结），R² 下限等
阈值待校准（来源缺口：CAL-P4 误差带尚未产出）。

模型与检验（docs/plan_v2.md 第 8 节 EXP-P4 第 3、4 条）：

- 线性拟合 Î = a + b·V̂：a 为截距，b 为斜率；语义温度 T̂ = 1/b
  （母论文 C4.1：斜率倒数估温度）。
- 二次备择 Î = a + b·V̂ + c·V̂²，lack-of-fit 用嵌套模型 F 检验：

      F = (RSS_lin − RSS_quad) / (RSS_quad / (n − 3))

  RSS 为残差平方和，n 为样本数，分子自由度 1（新增二次项），分母自由度
  n − 3（二次模型参数 3 个）；p 值取 F(1, n−3) 分布上尾。
- ΔBIC = BIC_linear − BIC_quadratic，BIC = n·ln(RSS/n) + k·ln(n)，k 为
  参数个数（高斯残差下与似然形式等价，去掉公共常数）。plan 判据要求
  ΔBIC < 2（线性不比二次差太多）。
- R² = 1 − RSS_lin/TSS，TSS 为总平方和。
- 曲率效应量 ρ = |c|·SD(V̂)/|b|：c 为二次项系数（中心化坐标），
  SD(V̂) 为横轴标准差；ρ 是"每一个横轴标准差上二次项贡献与线性项
  贡献之比"，无量纲。动机：样本量大时 F 检验对估计器残余的微小
  光滑曲率也拒绝（CAL-P4 实测：真仿射系统上 p 普遍 < 0.01），p 值
  判据须配效应量上限才能区分"统计可检的曲率"与"实质弯曲"；上限
  数值由校准产物定（thresholds_v1），母论文原文 p 值判据照测照报。

数值注意：二次拟合先把 V̂ 中心化再求解，避免设计矩阵病态；斜率与
检验统计量不受中心化影响。
"""
from __future__ import annotations

import math

import torch
from scipy import stats


def affine_fit_report(v_hat: torch.Tensor, i_hat: torch.Tensor) -> dict:
    """对 (V̂, Î) 散点做线性 / 二次拟合与 lack-of-fit 统计。返回 dict。"""
    v = torch.as_tensor(v_hat, dtype=torch.float64).reshape(-1)
    y = torch.as_tensor(i_hat, dtype=torch.float64).reshape(-1)
    n = v.shape[0]
    if n < 8:
        raise ValueError("样本太少，lack-of-fit 无意义")
    vc = v - v.mean()

    ones = torch.ones_like(vc)
    x_lin = torch.stack([ones, vc], dim=1)
    x_quad = torch.stack([ones, vc, vc**2], dim=1)

    beta_lin = torch.linalg.lstsq(x_lin, y.unsqueeze(1)).solution.squeeze(1)
    beta_quad = torch.linalg.lstsq(x_quad, y.unsqueeze(1)).solution.squeeze(1)
    rss_lin = float(((y - x_lin @ beta_lin) ** 2).sum())
    rss_quad = float(((y - x_quad @ beta_quad) ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())

    f_stat = (rss_lin - rss_quad) / (rss_quad / (n - 3)) if rss_quad > 0 else math.inf
    p_lack_of_fit = float(stats.f.sf(f_stat, 1, n - 3))
    bic_lin = n * math.log(rss_lin / n) + 2 * math.log(n)
    bic_quad = n * math.log(rss_quad / n) + 3 * math.log(n)

    slope = float(beta_lin[1])
    intercept = float(beta_lin[0] - beta_lin[1] * v.mean())
    # OLS 斜率标准误（同方差假设）：sqrt( (RSS/(n−2)) / Σ(v−v̄)² )
    se_slope = math.sqrt(rss_lin / (n - 2) / float((vc**2).sum()))

    sd_x = float(vc.std())
    return {
        "n": n,
        "slope": slope,
        "intercept": intercept,
        "se_slope": se_slope,
        "sd_x": sd_x,
        "r_squared": 1.0 - rss_lin / tss if tss > 0 else float("nan"),
        "f_lack_of_fit": f_stat,
        "p_lack_of_fit": p_lack_of_fit,
        "delta_bic_lin_minus_quad": bic_lin - bic_quad,
        "quad_coeff": float(beta_quad[2]),
        "curvature_effect_ratio": (
            abs(float(beta_quad[2])) * sd_x / abs(slope) if slope != 0 else float("nan")
        ),
        "temperature_hat": 1.0 / slope if slope > 0 else float("nan"),
        "slope_positive": slope > 0,
    }


def split_half_temperature(
    v_hat: torch.Tensor, i_hat: torch.Tensor, first_half_mask: torch.Tensor
) -> dict:
    """按给定切分各拟合温度，报告相对差 |T̂₁ − T̂₂| / T̄。

    切分口径（时间前后两段 / 两独立运行）由调用方经 first_half_mask 提供，
    对应 plan 第 8 节 EXP-P4 第 4 条的双口径。
    """
    mask = torch.as_tensor(first_half_mask, dtype=torch.bool).reshape(-1)
    r1 = affine_fit_report(v_hat[mask], i_hat[mask])
    r2 = affine_fit_report(v_hat[~mask], i_hat[~mask])
    t1, t2 = r1["temperature_hat"], r2["temperature_hat"]
    mean_t = 0.5 * (t1 + t2)
    return {
        "t_first": t1,
        "t_second": t2,
        "relative_gap": abs(t1 - t2) / mean_t if mean_t > 0 else float("nan"),
    }
