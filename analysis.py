from pathlib import Path
import argparse
import base64
import hashlib
import html
import io
import json
import platform
import warnings
from urllib.request import urlopen

import numpy as np
import pandas as pd
import scipy
from scipy.special import logsumexp
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

SERIES = {'DGS10': 'Treasury', 'BAMLC0A0CM': 'IG', 'BAMLH0A0HYM2': 'HY'}
ASOF = '2026-11-03'
MODEL_NAMES = ['OLS', 'MS variance', 'MS intercept+variance', 'MS full']


def load_data(folder, download=False):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    frames, manifest = [], {}
    for series in SERIES:
        path = folder / (series + '.csv')
        url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd=1996-12-31&coed={ASOF}'
        if download:
            with urlopen(url, timeout=60) as response:
                payload = response.read()
            candidate = pd.read_csv(io.BytesIO(payload))
            if series not in candidate or len(candidate) < 100:
                raise ValueError(f'Invalid FRED response: {series}')
            path.write_bytes(payload)
        if not path.exists():
            raise FileNotFoundError(f'{path}: supply FRED-format CSV or run --download; no synthetic fallback.')
        frame = pd.read_csv(path, na_values=['.'])
        frame = frame.rename(columns={frame.columns[0]: 'date'})
        frame['date'] = pd.to_datetime(frame['date'], errors='raise')
        if frame['date'].duplicated().any():
            raise ValueError(f'Duplicate dates in {series}')
        frame[series] = pd.to_numeric(frame[series], errors='raise')
        frame = frame.set_index('date')[[series]].sort_index().loc[:ASOF]
        if not np.isfinite(frame[series].dropna()).all():
            raise ValueError(f'Nonfinite values in {series}')
        manifest[series] = {'url': url, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'start': str(frame.index.min().date()), 'end': str(frame.index.max().date()),
                            'rows': len(frame), 'nonmissing': int(frame[series].count())}
        frames.append(frame)
    raw = pd.concat(frames, axis=1).sort_index().rename(columns=SERIES)
    complete = raw.dropna()
    complete = complete[complete.index.dayofweek < 5]
    if len(complete) < 250:
        raise ValueError('Need at least 250 common daily observations.')
    return raw, complete, manifest


def prepare(complete):
    weekly = complete.resample('W-FRI').last()
    weekly = weekly.loc[weekly.index <= complete.index.max()]
    changes = weekly.diff().mul(100).dropna()
    changes['lag_HY'] = weekly['HY'].shift(1).reindex(changes.index)
    changes['lag_rate'] = weekly['Treasury'].shift(1).reindex(changes.index)
    daily = complete.diff().mul(100)
    daily.loc[complete.index.to_series().diff().dt.days > 4] = np.nan
    return weekly, changes, daily.dropna()


def ols_fit(frame, asset, hac=4):
    return sm.OLS(frame[asset], sm.add_constant(frame['Treasury'])).fit(cov_type='HAC', cov_kwds={'maxlags': hac})


def fit_ms(frame, asset, name, starts=4):
    switching_intercept = name != 'MS variance'
    switching_beta = name == 'MS full'
    model = MarkovRegression(frame[asset].to_numpy(), k_regimes=2,
                             exog=frame[['Treasury']].to_numpy(), trend='c',
                             switching_trend=switching_intercept, switching_exog=switching_beta,
                             switching_variance=True)
    candidates, attempts = [], []
    for seed in range(starts):
        np.random.seed(2400 + seed)
        try:
            with warnings.catch_warnings(record=True) as caught:
                result = model.fit(search_reps=8, search_iter=8, em_iter=15, maxiter=600, disp=False)
            valid = bool(result.mle_retvals.get('converged', False) and np.isfinite(result.llf))
            attempts.append({'seed': 2400 + seed, 'converged': valid, 'llf': float(result.llf),
                             'warnings': [str(item.message) for item in caught]})
            if valid:
                candidates.append(result)
        except Exception as error:
            attempts.append({'seed': 2400 + seed, 'error': str(error)})
    if not candidates:
        raise RuntimeError(f'All starts failed: {asset}, {name}; {attempts}')
    return max(candidates, key=lambda result: result.llf), attempts


def ms_components(result):
    params = dict(zip(result.model.param_names, result.params))
    coefficients = np.array([result.params[result.model.parameters[state, 'exog']] for state in range(2)])
    intercept = coefficients[:, 0]
    beta = coefficients[:, 1]
    variance = np.array([params[f'sigma2[{state}]'] for state in range(2)])
    transition = np.asarray(result.regime_transition)[:, :, 0]
    return intercept, beta, variance, transition


def conditional_holdout(train, test, asset, name):
    observed = test[asset].to_numpy()
    rate = test['Treasury'].to_numpy()
    if name == 'OLS':
        fitted = ols_fit(train, asset)
        predicted = fitted.params.iloc[0] + fitted.params.iloc[1] * rate
        sigma = np.sqrt(np.mean(fitted.resid ** 2))
        log_density = norm.logpdf(observed, predicted, sigma)
    else:
        fitted, _ = fit_ms(train, asset, name, starts=3)
        intercept, beta, variance, transition = ms_components(fitted)
        posterior = np.asarray(fitted.filtered_marginal_probabilities)[-1]
        predicted, log_density = [], []
        for rate_change, spread_change in zip(rate, observed):
            prior = transition @ posterior
            means = intercept + beta * rate_change
            predicted.append(prior @ means)
            joint = np.log(np.maximum(prior, 1e-300)) + norm.logpdf(spread_change, means, np.sqrt(variance))
            score = logsumexp(joint)
            log_density.append(score)
            posterior = np.exp(joint - score)
        predicted = np.asarray(predicted)
    return {'asset': asset, 'model': name, 'n_train': len(train), 'n_test': len(test),
            'RMSE_bp': float(np.sqrt(np.mean((observed - predicted) ** 2))),
            'MAE_bp': float(np.mean(np.abs(observed - predicted))),
            'mean_log_density': float(np.mean(log_density))}


def weighted_corr(rate, spread, weights):
    rate_center = rate - np.average(rate, weights=weights)
    spread_center = spread - np.average(spread, weights=weights)
    return np.sum(weights * rate_center * spread_center) / np.sqrt(np.sum(weights * rate_center ** 2) * np.sum(weights * spread_center ** 2))


def image_tag(path):
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{encoded}" alt="{path.stem}">'


def analyze(data_dir, out_dir, download=False):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw, complete, manifest = load_data(data_dir, download)
    weekly, changes, daily = prepare(complete)
    split = int(len(changes) * .70)
    train, test = changes.iloc[:split], changes.iloc[split:]
    summary, comparisons, states, holdout, robustness, known, diagnostics = [], [], [], [], [], [], []
    fits, search_log, probability_frames, rolling_frames = {}, {}, [], []
    plt.rcParams.update({'figure.dpi': 145, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    for asset in ['IG', 'HY']:
        print(f'Fitting {asset}, {len(changes)} weekly changes', flush=True)
        base = ols_fit(changes, asset)
        interval = base.conf_int().loc['Treasury']
        summary.append({'asset': asset, 'n': len(changes), 'corr': changes[asset].corr(changes.Treasury),
                        'beta': base.params.Treasury, 'HAC95_low': interval.iloc[0], 'HAC95_high': interval.iloc[1],
                        'R2': base.rsquared})
        for name in MODEL_NAMES:
            if name == 'OLS':
                result, count = base, 3
                likelihood = result.llf
            else:
                result, attempts = fit_ms(changes, asset, name)
                search_log[asset + '_' + name] = attempts
                count, likelihood = len(result.params), result.llf
                if name == 'MS full':
                    fits[asset] = result
            comparisons.append({'asset': asset, 'model': name, 'n': len(changes), 'k_including_variance': count,
                                'loglik': likelihood, 'AIC': -2 * likelihood + 2 * count,
                                'BIC': -2 * likelihood + np.log(len(changes)) * count})
            try:
                holdout.append(conditional_holdout(train, test, asset, name))
            except Exception as error:
                holdout.append({'asset': asset, 'model': name, 'error': str(error)})
        full = fits[asset]
        intercept, beta, variance, transition = ms_components(full)
        order = np.argsort(variance)
        smooth = np.asarray(full.smoothed_marginal_probabilities)
        filtered = np.asarray(full.filtered_marginal_probabilities)
        confidence = np.asarray(full.conf_int())
        for label, state in zip(['lower residual volatility', 'higher residual volatility'], order):
            beta_index = full.model.param_names.index(f'x1[{state}]')
            weights = smooth[:, state]
            mask = smooth.argmax(axis=1) == state
            states.append({'asset': asset, 'state': label, 'beta': beta[state],
                           'MLE95_low': confidence[beta_index, 0], 'MLE95_high': confidence[beta_index, 1],
                           'weighted_corr': weighted_corr(changes.Treasury.to_numpy(), changes[asset].to_numpy(), weights),
                           'hard_corr': changes.loc[mask, 'Treasury'].corr(changes.loc[mask, asset]),
                           'hard_n': int(mask.sum()), 'probability_mass': weights.sum(),
                           'residual_sd_bp': np.sqrt(variance[state]),
                           'expected_duration_weeks': 1 / (1 - transition[state, state])})
        probability_frames.append(pd.DataFrame({'asset': asset, 'smoothed_high': smooth[:, order[1]],
                                                 'filtered_high': filtered[:, order[1]]}, index=changes.index))
        param_indices = [full.model.param_names.index(f'x1[{state}]') for state in order]
        covariance = np.asarray(full.cov_params())
        diff_var = covariance[param_indices[0], param_indices[0]] + covariance[param_indices[1], param_indices[1]] - 2 * covariance[param_indices[0], param_indices[1]]
        gap = beta[order[1]] - beta[order[0]]
        diagnostics.append({'asset': asset, 'beta_high_minus_low': gap,
                            'approx_Wald_p': 2 * norm.sf(abs(gap) / np.sqrt(diff_var)) if diff_var > 0 else np.nan,
                            'OLS_LjungBox_p_lag8': acorr_ljungbox(base.resid, lags=[8], return_df=True).lb_pvalue.iloc[0],
                            'OLS_squared_LjungBox_p_lag8': acorr_ljungbox(base.resid ** 2, lags=[8], return_df=True).lb_pvalue.iloc[0]})
        for window in [26, 52]:
            for end in range(window, len(changes) + 1):
                result = ols_fit(changes.iloc[end-window:end], asset)
                band = result.conf_int().loc['Treasury']
                rolling_frames.append({'date': changes.index[end-1], 'asset': asset, 'window': window,
                                       'beta': result.params.Treasury, 'low': band.iloc[0], 'high': band.iloc[1]})
        for label, subset in [('daily', daily), ('weekly', changes),
                              ('weekly excluding top 1% absolute OAS changes', changes[changes[asset].abs() <= changes[asset].abs().quantile(.99)])]:
            result = ols_fit(subset, asset, hac=5 if label == 'daily' else 4)
            robustness.append({'asset': asset, 'sample': label, 'n': len(subset),
                               'corr': subset[asset].corr(subset.Treasury), 'beta': result.params.Treasury})
        labels = pd.cut(changes.lag_HY, [-np.inf, 4, 6, np.inf], right=False, labels=['HY <400bp', 'HY 400–600bp', 'HY >=600bp'])
        for label in labels.cat.categories:
            subset = changes.loc[labels == label]
            known.append({'asset': asset, 'previous_week_HY': label, 'n': len(subset),
                          'corr': subset.Treasury.corr(subset[asset]) if len(subset) >= 20 else np.nan,
                          'beta': ols_fit(subset, asset).params.Treasury if len(subset) >= 20 else np.nan,
                          'status': 'estimated' if len(subset) >= 20 else 'unavailable: fewer than 20 weeks'})
    tables = {name: pd.DataFrame(values) for name, values in [('pooled', summary), ('model_comparison', comparisons),
              ('regime_statistics', states), ('holdout', holdout), ('robustness', robustness), ('observable_regimes', known), ('diagnostics', diagnostics)]}
    probabilities = pd.concat(probability_frames)
    rolling = pd.DataFrame(rolling_frames)
    for name, frame in tables.items():
        frame.to_csv(out / (name + '.csv'), index=False)
    probabilities.to_csv(out / 'state_probabilities.csv', index_label='date')
    rolling.to_csv(out / 'rolling_betas.csv', index=False)
    changes.to_csv(out / 'weekly_changes_bp.csv', index_label='date')
    metadata = {'asof': ASOF, 'data_status': 'REAL FRED observations; no simulated values',
                'acquisition': 'Local CSV input. Bundled snapshot: FRED public table date/value pairs retrieved 2026-11-03. User replacements require separate provenance.',
                'series': manifest, 'common_daily_n': len(complete), 'weekly_change_n': len(changes),
                'weekly_start': str(changes.index.min().date()), 'weekly_end': str(changes.index.max().date()),
                'holdout_start': str(test.index.min().date()), 'python': platform.python_version(),
                'versions': {module.__name__: module.__version__ for module in [np, pd, scipy, statsmodels, matplotlib]}}
    if download:
        metadata['acquisition'] = 'Direct FRED CSV download; run-time snapshot, capped at asof observation date, not historical vintage'
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (out / 'optimizer_log.json').write_text(json.dumps(search_log, indent=2), encoding='utf-8')
    figures = []
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    weekly[['IG', 'HY']].mul(100).plot(ax=axes[0], color=['#007f87', '#c65825'])
    weekly.Treasury.plot(ax=axes[1], color='#223d6b')
    axes[0].set(ylabel='OAS (bp)', title='REAL FRED DATA | Limited 2023–2026 sample')
    axes[1].set(ylabel='10-year Treasury (%)', xlabel='Week ending Friday')
    fig.tight_layout(); fig.savefig(out / '01_levels.png'); plt.close(fig); figures.append('01_levels.png')
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for axis, asset in zip(axes, ['IG', 'HY']):
        for window, color in [(26, '#b46d32'), (52, '#007f87')]:
            frame = rolling[(rolling.asset == asset) & (rolling.window == window)]
            axis.plot(frame.date, frame.beta, label=f'{window}-week beta', color=color)
            if window == 52:
                axis.fill_between(frame.date, frame.low, frame.high, color=color, alpha=.16, label='52-week pointwise HAC 95% CI')
        axis.axhline(0, color='gray', lw=.8); axis.set(ylabel='OAS bp / Treasury bp', title=asset); axis.legend(fontsize=8)
    fig.suptitle('Rolling contemporaneous betas | REAL DATA'); fig.tight_layout()
    fig.savefig(out / '02_rolling_betas.png'); plt.close(fig); figures.append('02_rolling_betas.png')
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for axis, asset in zip(axes, ['IG', 'HY']):
        frame = probabilities[probabilities.asset == asset]
        axis.plot(frame.index, frame.smoothed_high, label='Smoothed (uses future observations)', color='#c65825')
        axis.plot(frame.index, frame.filtered_high, label='Filtered (full-sample parameters)', color='#223d6b', alpha=.6)
        axis.set(title=asset, ylabel='Probability', ylim=(-.03, 1.03)); axis.legend(fontsize=8)
    fig.suptitle('Higher residual volatility state | Two-state full model'); fig.tight_layout()
    fig.savefig(out / '03_probabilities.png'); plt.close(fig); figures.append('03_probabilities.png')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, asset in zip(axes, ['IG', 'HY']):
        result = fits[asset]; intercept, beta, variance, transition = ms_components(result)
        higher = np.argmax(variance)
        axis.scatter(changes.Treasury, changes[asset], c=np.asarray(result.smoothed_marginal_probabilities)[:, higher], cmap='coolwarm', vmin=0, vmax=1, s=20, alpha=.7)
        grid = np.linspace(changes.Treasury.min(), changes.Treasury.max(), 100)
        for state in np.argsort(variance):
            label = 'higher volatility' if state == higher else 'lower volatility'
            axis.plot(grid, intercept[state] + beta[state] * grid, label=label)
        axis.set(title=asset, xlabel='Weekly Treasury change (bp)', ylabel='Weekly OAS change (bp)'); axis.legend(fontsize=8)
    fig.suptitle('State-dependent regression lines | Red = higher-state probability'); fig.tight_layout()
    fig.savefig(out / '04_scatter.png'); plt.close(fig); figures.append('04_scatter.png')
    build_report(out, tables, metadata, figures)
    return tables, metadata


def build_report(out, tables, metadata, figures):
    sections = []
    for title, key, note in [
        ('全样本变化关系', 'pooled', 'beta 单位：国债收益率变化 1bp，对应 OAS 平均变化多少 bp。HAC 使用 4 阶滞后。'),
        ('模型比较', 'model_comparison', 'AIC/BIC 越低越好；所有模型使用相同样本，OLS 参数数含误差方差。优于 OLS 不等于 beta 必须切换，应比较 MS full 与 MS intercept+variance。'),
        ('状态内相关与 beta', 'regime_statistics', '按残差方差排序状态。weighted_corr 使用平滑概率作权重；hard_corr 使用最大概率归类。二者均为事后描述。MLE 区间是条件高斯模型的渐近区间，不含模型选择不确定性；不是 HAC 区间。'),
        ('时间顺序留出比较', 'holdout', '前 70% 训练、后 30% 检验；参数冻结，逐期更新过滤概率。RMSE/MAE 越低越好，平均对数密度越高越好。使用当期已实现的国债变化，属于条件预测评估，不是提前一周交易预测。'),
        ('可观测状态核对', 'observable_regimes', '用前一周 HY OAS 水平固定分箱，不根据本周收益或利差变化符号定义状态。不足 20 周不估计。'),
        ('稳健性', 'robustness', '比较日度、周度及排除最大 1% 利差变化的周度结果；删尾只作敏感性检查，主结果保留所有极端值。'),
        ('诊断', 'diagnostics', 'Wald p 值只是近似诊断，状态弱识别和模型选择会影响显著性。Ljung–Box 检查 OLS 残差及残差平方的序列相关。没有使用常规卡方 LR 检验选择状态数。')]:
        sections.append(f'<h2>{title}</h2><p>{note}</p><div class="table">{tables[key].to_html(index=False, float_format=lambda value: f"{value:.4f}", na_rep="不可估计")}</div>')
    pooled = tables['pooled']
    findings = []
    for asset in ['IG', 'HY']:
        row = pooled[pooled.asset == asset].iloc[0]
        comparison = tables['model_comparison'].query('asset == @asset').set_index('model')
        delta = comparison.loc['MS full', 'BIC'] - comparison.loc['MS intercept+variance', 'BIC']
        findings.append(f'<li>{asset}：周度相关系数 {row["corr"]:.3f}，固定 beta {row.beta:.3f}，HAC 95% 区间 [{row.HAC95_low:.3f}, {row.HAC95_high:.3f}]。允许 beta 切换相对于固定 beta 的状态模型，BIC 差为 {delta:.2f}（负数支持切换）。</li>')
        winner = comparison.BIC.idxmin()
        scores = tables['holdout'].query('asset == @asset')
        best_rmse = scores.loc[scores.RMSE_bp.idxmin(), 'model']
        findings.append(f'<li>{asset}：四模型中 BIC 最优为 {winner}；留出样本 RMSE 最优为 {best_rmse}。这些比较尚未检验预测损失差异的统计显著性。</li>')
        state_rows = tables['regime_statistics'].query('asset == @asset')
        for _, state_row in state_rows.iterrows():
            if state_row.hard_n < 20 or state_row.expected_duration_weeks < 1.5:
                findings.append(f'<li>{asset} 的 {state_row.state} 状态：最大概率分类 {int(state_row.hard_n)} 周，期望持续期 {state_row.expected_duration_weeks:.2f} 周。状态过短或样本稀少，beta 差异应视为探索性结果。</li>')
    body = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>利率与信用利差：状态切换实证</title>
    <style>body{{font-family:system-ui,sans-serif;color:#203047;max-width:1180px;margin:40px auto;padding:0 24px;line-height:1.7}}h1,h2{{color:#123c53}}.notice{{padding:18px;background:#fff3d6;border-left:5px solid #c88320}}img{{max-width:100%;height:auto}}.table{{overflow:auto}}table{{border-collapse:collapse;font-size:12px;white-space:nowrap}}th,td{{border-bottom:1px solid #dde4e8;padding:8px;text-align:right}}th{{background:#eff4f6}}code{{background:#edf2f5;padding:2px 5px}}a{{color:#007b88}}@media print{{body{{margin:0}}h2{{break-after:avoid}}img{{break-inside:avoid}}}}</style>
    <h1>美国国债收益率与 IG / HY OAS<br>关系是否随状态切换？</h1>
    <p>研究截止：2026-11-03 · 真实 FRED 数据 · 可复现 Python / Jupyter 分析</p>
    <div class="notice"><b>数据边界：</b>FRED 的两条 ICE OAS 序列自 2026 年 4 月起仅保留近三年。本次共同日度样本 {metadata['common_daily_n']} 条，周度变化 {metadata['weekly_change_n']} 条，{metadata['weekly_start']} 至 {metadata['weekly_end']}。无法检验 2008、2020 或 2022；没有模拟数据，没有把不存在的长历史结果补出来。</div>
    <h2>先看结论</h2><ul>{''.join(findings)}</ul>
    <p>这些结果回答“同一周利率变化与利差变化如何共同变动”。它们不能判定“利率水平越高，利差水平必然越高或越低”，也不能识别因果。是否有必要让 beta 切换，必须结合 BIC、留出表现、区间和状态样本量，而非仅看状态线条不同。</p>
    <div class="notice"><b>短样本状态识别警告：</b>请检查下方 expected_duration_weeks 和 hard_n。接近一周的持续期、接近边界的转移概率或极少数高波动周，可能只是在拟合异常值，并不是稳定的宏观 regime。此时状态 beta 的渐近显著性不宜当作可靠证据。</div>
    <h2>经济解释：原来的两个说法都缺少冲击来源</h2>
    <p>经济增长预期改善可能同时推高国债收益率并压低违约风险；通胀或融资成本冲击可能同时推高收益率和利差；避险需求可能压低国债收益率并推高利差。这些是待检验机制，不是本模型自动识别出来的经济状态。企业“付不起更高利息”也不会强制压低市场要求的利差，可能表现为债券价格下跌、融资受限或违约。</p>
    <p>10 年期国债收益率并不等同于政策利率。OAS 对整条即期国债曲线作期权调整；这里用 DGS10 作为简化的利率因子，不具有严格久期匹配。OAS 已减去国债基准，其与国债变化的负关系也需考虑基准构造、流动性和风险溢价，不能全部归因于信用基本面。</p>
    <h2>数据与方法</h2>
    <ol><li>DGS10、BAMLC0A0CM、BAMLH0A0HYM2 均以百分数水平输入。先取共同有效交易日，再选每周最后一个共同观测；剔除未完成的最后一周。先对齐后差分，变化乘 100 转为 bp，不前向填充。</li>
    <li>主式：ΔOASₜ = αₛ + βₛ ΔUST10ₜ + εₜ；隐状态遵循一阶、固定转移概率的两状态 Markov 链，εₜ 条件高斯、状态方差可变。IG 与 HY 分别估计，状态不强制一致。</li>
    <li>四种模型：固定参数 OLS；仅方差切换；截距与方差切换但 beta 固定；截距、beta 和方差均切换。状态由残差方差排序，命名为较低／较高残差波动，不预先命名增长、通胀或危机。</li>
    <li>完整样本每个 Markov 模型使用 4 个固定随机种子，每个种子 8 次随机初值搜索；仅保留收敛且似然有限的解，选最大似然。优化记录随附件提供。区间可能因局部最优、厚尾和短样本而不稳定。</li>
    <li>26/52 周滚动 OLS；52 周图显示逐点 HAC 95% 区间，不是全路径同时置信带。隐状态概率使用完整样本估计的参数；即使 filtered 曲线也不是真正实时估计。平滑概率另使用未来观测。</li>
    <li>留出检验起点 {metadata['holdout_start']}，训练／检验按时间顺序切分。各模型在训练期重新估计；测试期仅使用上一期过滤状态及当期国债变化，观察当期利差后才更新状态，不用平滑概率预测。</li></ol>
    {''.join(image_tag(out / name) for name in figures)}{''.join(sections)}
    <h2>如何重跑与扩展</h2><p>解压完整附件，在文件夹内安装 requirements.txt；打开 regime_analysis.ipynb 并 Run All，或运行 <code>python analysis.py --data data --out rerun</code>。联网重新获取运行 <code>python analysis.py --data fresh_data --out fresh_results --download</code>。下载失败会明确报错，不切换到模拟值。FRED 新下载可能只有滚动三年，与冻结快照不同。</p>
    <p>完整历史可放入相同格式的三个 CSV：第一列 date，第二列为对应 FRED series ID，数值单位为百分数而非 bp。替换后重跑即可。应另增加 2/5 年期利率、曲线斜率、实际利率／盈亏平衡通胀及外部冲击识别，再检验状态能否命名为增长或通胀。此次未取得这些控制变量，也未完成长历史或因果检验。</p>
    <h2>研究依据与来源</h2><ul>
    <li><a href="https://fred.stlouisfed.org/series/DGS10">FRED DGS10</a>；<a href="https://fred.stlouisfed.org/series/BAMLC0A0CM">IG OAS 定义和三年限制</a>；<a href="https://fred.stlouisfed.org/series/BAMLH0A0HYM2">HY OAS 定义和三年限制</a>。快照来自对应 /data/ 页面；原始下载在本地网络失败后采用网页日期／数值表提取，没有手工插补。</li>
    <li><a href="https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00089">Duffee (1998), The Relation Between Treasury Yields and Corporate Bond Yield Spreads</a>：研究国债利率与公司债利差关系；其传统利差和债券结构背景不等于本研究的指数 OAS。</li>
    <li><a href="https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00402/abstract">Collin-Dufresne, Goldstein & Martin (2001), The Determinants of Credit Spread Changes</a>：强调公司债收益率变化与信用利差变化是不同研究对象。</li>
    <li><a href="https://www.clevelandfed.org/publications/economic-commentary/2007/ec-20070301-credit-spreads-and-subordinated-debt">Cleveland Fed (2007)</a>：信用利差亦包含流动性、税收等因素。</li>
    <li><a href="https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_regression.MarkovRegression.html">statsmodels MarkovRegression</a>：高斯 Markov 切换回归实现。</li></ul>
    <p>ICE 数据仅用于你的内部研究，公开转发原始指数数据需遵守来源许可。快照不是 ALFRED 历史版本，严格实时回测仍需 vintage 数据。原对话的助手回复仅返回内容引用标记，未被作为可核验研究证据。</p></html>'''
    (out / 'report.html').write_text(body, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data')
    parser.add_argument('--out', default='results')
    parser.add_argument('--download', action='store_true')
    arguments = parser.parse_args()
    analyze(arguments.data, arguments.out, arguments.download)
