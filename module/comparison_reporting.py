"""Common raw prediction metrics and side-by-side diagnostic figures."""
import numpy as np
from module.xgb_common import regression_metrics


def metrics(actual, predicted):
    return {**regression_metrics(actual, predicted),
            'bias': float(np.mean(np.asarray(predicted) - np.asarray(actual)))}


def plots(oof, folds, directory, unit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models = list(oof.model.unique())
    low = min(oof.actual.min(), oof.predicted.min())
    high = max(oof.actual.max(), oof.predicted.max())
    pad = max((high - low) * .03, .1)
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)
    for ax, name in zip(axes[0], models):
        rows = oof[oof.model == name]
        score = metrics(rows.actual, rows.predicted)
        ax.scatter(rows.actual, rows.predicted, s=10, alpha=.5)
        ax.plot([low-pad, high+pad], [low-pad, high+pad], 'k--')
        ax.set(xlim=(low-pad, high+pad), ylim=(low-pad, high+pad),
               xlabel=f'Observed [{unit}]', ylabel=f'Predicted [{unit}]', title=name)
        ax.text(.04, .96, '\n'.join(f'{k.upper()}: {score[k]:.4f}' for k in ('r2', 'rmse', 'mae')),
                va='top', transform=ax.transAxes)
    fig.tight_layout(); fig.savefig(directory / 'scatter_comparison.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.histogram_bin_edges(oof.residual, bins=30)
    for name in models:
        ax.hist(oof.loc[oof.model == name, 'residual'], bins=bins, alpha=.45, label=name, density=True)
    ax.axvline(0, color='black', linestyle='--')
    ax.set(xlabel=f'Prediction - observation [{unit}]', ylabel='Density'); ax.legend()
    fig.tight_layout(); fig.savefig(directory / 'residual_comparison.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ('r2', 'rmse', 'mae')):
        for name in models:
            rows = folds[folds.model == name]
            ax.plot(rows.fold, rows[metric], 'o-', label=name)
        ax.set(xlabel='Spatial fold', ylabel=metric.upper()); ax.legend()
    fig.tight_layout(); fig.savefig(directory / 'cv_comparison.png', dpi=160); plt.close(fig)


def distribution_plots(display, output):
    import matplotlib.pyplot as plt
    for target, rows in display.groupby('target'):
        names = list(rows.model.unique())
        fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5), squeeze=False)
        for ax, name in zip(axes[0], names):
            data = rows[rows.model == name]
            obj = ax.scatter(data.x, data.y, c=data.predicted_raw, s=3,
                             vmin=rows.predicted_raw.min(), vmax=rows.predicted_raw.max())
            ax.set(title=f'{target}: {name} (raw)', xlabel='X [m]', ylabel='Y [m]', aspect='equal')
            fig.colorbar(obj, ax=ax)
        fig.tight_layout(); fig.savefig(output / target / 'distribution_comparison.png', dpi=160); plt.close(fig)
        if set(names) == {'xgboost', 'tabpfn'}:
            wide = rows.pivot(index=['x', 'y'], columns='model', values='predicted_raw')
            delta = wide.tabpfn - wide.xgboost
            fig, ax = plt.subplots(figsize=(6, 5))
            limit = max(abs(delta).max(), 1e-9)
            obj = ax.scatter(wide.index.get_level_values('x'), wide.index.get_level_values('y'),
                             c=delta, s=3, cmap='coolwarm', vmin=-limit, vmax=limit)
            ax.set(title=f'{target}: TabPFN - XGBoost', xlabel='X [m]', ylabel='Y [m]', aspect='equal')
            fig.colorbar(obj, ax=ax); fig.tight_layout()
            fig.savefig(output / target / 'difference_distribution.png', dpi=160); plt.close(fig)
