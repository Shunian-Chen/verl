#!/usr/bin/env python3
"""
Data Statistics Visualization Tool

Generates comprehensive visualizations for data construction pipeline statistics.

Usage:
    python3 visualize_stats.py <stats_json> [--output-dir <dir>]

Example:
    python3 visualize_stats.py ./test_output_10/data_statistics.json --output-dir ./visualizations
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    HAS_VIZ_LIBS = True
except ImportError:
    HAS_VIZ_LIBS = False
    print("WARNING: matplotlib/seaborn not available. Install with: pip install matplotlib seaborn")


class DataVisualizer:
    """Generate visualizations for data statistics"""

    def __init__(self, stats_path: str, output_dir: str = None):
        self.stats_path = Path(stats_path)

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.stats_path.parent / 'visualizations'

        self.output_dir.mkdir(exist_ok=True)

        # Load statistics
        with open(self.stats_path, 'r') as f:
            self.stats = json.load(f)

        # Set style
        if HAS_VIZ_LIBS:
            sns.set_style("whitegrid")
            sns.set_palette("husl")

    def generate_all_visualizations(self):
        """Generate all visualizations"""
        if not HAS_VIZ_LIBS:
            print("ERROR: Cannot generate visualizations without matplotlib/seaborn")
            print("Install with: pip install matplotlib seaborn")
            return False

        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)

        viz_functions = [
            ("Strategy Distribution", self._plot_strategy_distribution),
            ("Complexity Distribution", self._plot_complexity_distribution),
            ("Question Type Distribution", self._plot_question_type_distribution),
            ("Response Length Distribution", self._plot_response_length),
            ("Cycle Count Distribution", self._plot_cycle_distribution),
            ("Validation Score Distribution", self._plot_score_distribution),
            ("Score Dimensions Comparison", self._plot_score_dimensions),
            ("Validation Dimension Distributions", self._plot_validation_dimension_distributions),
            ("Multiple Choice Answer Distribution", self._plot_mc_answer_distribution),
            ("Success Rate by Strategy", self._plot_success_by_strategy),
            ("Success Rate by Complexity", self._plot_success_by_complexity),
            ("Category Coverage", self._plot_category_coverage),
            ("Quality Tiers", self._plot_quality_tiers),
            ("Failure Analysis", self._plot_failure_analysis),
            ("Score Correlation Heatmap", self._plot_score_correlation),
            ("Comprehensive Dashboard", self._plot_dashboard)
        ]

        generated = []
        for name, func in viz_functions:
            try:
                print(f"Generating: {name}...")
                output_path = func()
                if output_path:
                    generated.append(output_path)
            except Exception as e:
                print(f"  ERROR: {e}")

        print(f"\n{'='*80}")
        print(f"Generated {len(generated)} visualizations in: {self.output_dir}")
        print("="*80)

        return True

    def _plot_strategy_distribution(self):
        """Plot question strategy distribution"""
        dist = self.stats['distribution_analysis']['question_strategies']

        if not dist['distribution']:
            return None

        strategies = [item['value'] for item in dist['distribution']]
        counts = [item['count'] for item in dist['distribution']]

        fig, ax = plt.subplots(figsize=(12, 6))

        bars = ax.barh(strategies, counts, color=sns.color_palette("husl", len(strategies)))

        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count, i, f' {count}', va='center', fontsize=10)

        ax.set_xlabel('Count', fontsize=12)
        ax.set_ylabel('Strategy', fontsize=12)
        ax.set_title('Question Strategy Distribution', fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        plt.tight_layout()
        output_path = self.output_dir / 'strategy_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_complexity_distribution(self):
        """Plot complexity distribution"""
        dist = self.stats['distribution_analysis']['complexity']

        if not dist['distribution']:
            return None

        complexities = [item['value'] for item in dist['distribution']]
        counts = [item['count'] for item in dist['distribution']]
        percentages = [item['percentage'] for item in dist['distribution']]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart
        bars = ax1.bar(complexities, counts, color=sns.color_palette("husl", len(complexities)))
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom', fontsize=11)

        ax1.set_xlabel('Complexity Level', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Complexity Distribution (Count)', fontsize=13, fontweight='bold')

        # Pie chart
        colors = sns.color_palette("husl", len(complexities))
        ax2.pie(percentages, labels=complexities, autopct='%1.1f%%',
                startangle=90, colors=colors)
        ax2.set_title('Complexity Distribution (Percentage)', fontsize=13, fontweight='bold')

        plt.tight_layout()
        output_path = self.output_dir / 'complexity_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_question_type_distribution(self):
        """Plot question type distribution"""
        qt = self.stats['distribution_analysis']['question_types']

        types = ['Multiple Choice', 'Open-ended']
        counts = [qt['multiple_choice']['count'], qt['open_ended']['count']]
        percentages = [qt['multiple_choice']['percentage'], qt['open_ended']['percentage']]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Bar chart
        colors = ['#3498db', '#e74c3c']
        bars = ax1.bar(types, counts, color=colors)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom', fontsize=12)

        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Question Type Distribution', fontsize=13, fontweight='bold')

        # Pie chart
        ax2.pie(percentages, labels=types, autopct='%1.1f%%',
                startangle=90, colors=colors)
        ax2.set_title('Question Type Proportion', fontsize=13, fontweight='bold')

        plt.tight_layout()
        output_path = self.output_dir / 'question_type_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_response_length(self):
        """Plot response length distribution"""
        rl = self.stats['distribution_analysis']['response_length']

        if rl['count'] == 0:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        bins = rl['histogram']['bin_edges']
        counts = rl['histogram']['counts']

        ax1.bar(bins[:-1], counts, width=np.diff(bins), align='edge',
                color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(rl['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {rl["mean"]:.0f}')
        ax1.axvline(rl['percentiles']['p50'], color='green', linestyle='--', linewidth=2, label=f'Median: {rl["percentiles"]["p50"]:.0f}')

        ax1.set_xlabel('Word Count', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Response Length Histogram', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot
        stats_data = {
            'min': rl['min'],
            'q25': rl['percentiles']['p25'],
            'median': rl['percentiles']['p50'],
            'q75': rl['percentiles']['p75'],
            'max': rl['max'],
            'mean': rl['mean']
        }

        box_data = [stats_data]
        bp = ax2.boxplot([0], positions=[1], widths=0.6, patch_artist=True,
                         showmeans=True, meanline=True)

        # Manually set box plot statistics
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], linewidth=1.5)

        ax2.text(1.4, rl['min'], f'Min: {rl["min"]}', va='center', fontsize=10)
        ax2.text(1.4, rl['percentiles']['p25'], f'Q1: {rl["percentiles"]["p25"]:.0f}', va='center', fontsize=10)
        ax2.text(1.4, rl['percentiles']['p50'], f'Median: {rl["percentiles"]["p50"]:.0f}', va='center', fontsize=10)
        ax2.text(1.4, rl['percentiles']['p75'], f'Q3: {rl["percentiles"]["p75"]:.0f}', va='center', fontsize=10)
        ax2.text(1.4, rl['max'], f'Max: {rl["max"]}', va='center', fontsize=10)
        ax2.text(1.4, rl['mean'], f'Mean: {rl["mean"]:.0f}', va='center', fontsize=10, color='green')

        ax2.set_ylabel('Word Count', fontsize=12)
        ax2.set_title('Response Length Statistics', fontsize=13, fontweight='bold')
        ax2.set_xticks([])

        plt.tight_layout()
        output_path = self.output_dir / 'response_length.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_cycle_distribution(self):
        """Plot look-think cycle distribution"""
        cycles = self.stats['distribution_analysis']['cycle_count']

        if cycles['count'] == 0:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        bins = cycles['histogram']['bin_edges']
        counts = cycles['histogram']['counts']

        ax.bar(bins[:-1], counts, width=np.diff(bins), align='edge',
               color='lightcoral', edgecolor='black', alpha=0.7)
        ax.axvline(cycles['mean'], color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {cycles["mean"]:.1f}')

        ax.set_xlabel('Number of Look-Think Cycles', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Look-Think Cycle Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / 'cycle_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_score_distribution(self):
        """Plot validation score distribution"""
        scores = self.stats['quality_metrics']['validation_scores']

        if scores['count'] == 0:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        bins = scores['histogram']['bin_edges']
        counts = scores['histogram']['counts']

        ax1.bar(bins[:-1], counts, width=np.diff(bins), align='edge',
                color='mediumseagreen', edgecolor='black', alpha=0.7)
        ax1.axvline(scores['mean'], color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {scores["mean"]:.2f}')
        ax1.axvline(scores['percentiles']['p50'], color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {scores["percentiles"]["p50"]:.2f}')

        ax1.set_xlabel('Validation Score', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Validation Score Distribution', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Statistics summary
        ax2.axis('off')
        stats_text = f"""
        VALIDATION SCORE STATISTICS

        Count:        {scores['count']:,}
        Mean:         {scores['mean']:.2f}
        Std Dev:      {scores['std']:.2f}

        Min:          {scores['min']:.2f}
        25th %ile:    {scores['percentiles']['p25']:.2f}
        Median:       {scores['percentiles']['p50']:.2f}
        75th %ile:    {scores['percentiles']['p75']:.2f}
        95th %ile:    {scores['percentiles']['p95']:.2f}
        Max:          {scores['max']:.2f}
        """

        ax2.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center')

        plt.tight_layout()
        output_path = self.output_dir / 'score_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_score_dimensions(self):
        """Plot score dimension distributions (box plots)"""
        dims = self.stats['quality_metrics'].get('score_dimensions', {})

        if not dims:
            return None

        # Prepare data for box plots
        dim_names = []
        dim_data = []

        for dim, stats in sorted(dims.items()):
            dim_names.append(dim.replace('_score', '').replace('_', ' ').title())
            # Create approximate distribution from percentiles
            # This is an approximation since we don't have raw data
            percentiles = stats['percentiles']
            dim_data.append([
                stats['min'],
                percentiles['p25'],
                percentiles['p50'],
                percentiles['p75'],
                stats['max']
            ])

        fig, ax = plt.subplots(figsize=(12, 6))

        # Create box plot
        bp = ax.boxplot(dim_data,
                        labels=dim_names,
                        patch_artist=True,
                        showmeans=True,
                        meanline=True,
                        widths=0.6)

        # Color the boxes
        colors = sns.color_palette("husl", len(dim_names))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Style the plot elements
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], linewidth=1.5)

        plt.setp(bp['means'], color='red', linewidth=2)
        plt.setp(bp['medians'], color='green', linewidth=2)

        # Add statistics as text
        for i, (dim, stats) in enumerate(sorted(dims.items()), 1):
            mean = stats['mean']
            std = stats['std']
            ax.text(i, stats['max'] + 0.2, f'μ={mean:.2f}\nσ={std:.2f}',
                   ha='center', va='bottom', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('Dimension', fontsize=12)
        ax.set_title('Quality Score Dimension Distributions', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 10])
        ax.grid(True, alpha=0.3, axis='y')

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', linewidth=2, label='Median'),
            Line2D([0], [0], color='red', linewidth=2, label='Mean')
        ]
        ax.legend(handles=legend_elements, loc='lower left')

        plt.tight_layout()
        output_path = self.output_dir / 'score_dimensions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_success_by_strategy(self):
        """Plot success rate by strategy"""
        data = self.stats['quality_metrics']['success_by_strategy']['by_category']

        if not data:
            return None

        # Take top 10 by total count
        data = data[:10]

        strategies = [item['value'] for item in data]
        success_rates = [item['success_rate'] for item in data]
        totals = [item['total'] for item in data]

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ['#2ecc71' if sr >= 80 else '#f39c12' if sr >= 60 else '#e74c3c' for sr in success_rates]
        bars = ax.barh(strategies, success_rates, color=colors, alpha=0.7, edgecolor='black')

        # Add labels
        for i, (bar, sr, total) in enumerate(zip(bars, success_rates, totals)):
            ax.text(sr, i, f' {sr:.1f}% (n={total})', va='center', fontsize=9)

        ax.set_xlabel('Success Rate (%)', fontsize=12)
        ax.set_ylabel('Strategy', fontsize=12)
        ax.set_title('Success Rate by Question Strategy', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 105])
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Good (>= 80%)'),
            Patch(facecolor='#f39c12', label='Moderate (60-80%)'),
            Patch(facecolor='#e74c3c', label='Low (< 60%)')
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        output_path = self.output_dir / 'success_by_strategy.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_success_by_complexity(self):
        """Plot success rate by complexity"""
        data = self.stats['quality_metrics']['success_by_complexity']['by_category']

        if not data:
            return None

        complexities = [item['value'] for item in data]
        success_rates = [item['success_rate'] for item in data]
        valid_counts = [item['valid_count'] for item in data]
        failed_counts = [item['failed_count'] for item in data]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Success rate bar chart
        colors = ['#2ecc71' if sr >= 80 else '#f39c12' if sr >= 60 else '#e74c3c' for sr in success_rates]
        bars = ax1.bar(complexities, success_rates, color=colors, alpha=0.7, edgecolor='black')

        for bar, sr in zip(bars, success_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{sr:.1f}%', ha='center', va='bottom', fontsize=11)

        ax1.set_ylabel('Success Rate (%)', fontsize=12)
        ax1.set_xlabel('Complexity Level', fontsize=12)
        ax1.set_title('Success Rate by Complexity', fontsize=13, fontweight='bold')
        ax1.set_ylim([0, 105])
        ax1.grid(True, alpha=0.3, axis='y')

        # Stacked bar chart showing valid vs failed
        x = np.arange(len(complexities))
        ax2.bar(x, valid_counts, label='Valid', color='#2ecc71', alpha=0.7)
        ax2.bar(x, failed_counts, bottom=valid_counts, label='Failed', color='#e74c3c', alpha=0.7)

        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_xlabel('Complexity Level', fontsize=12)
        ax2.set_title('Valid vs Failed by Complexity', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(complexities)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = self.output_dir / 'success_by_complexity.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_category_coverage(self):
        """Plot top categories"""
        cat_data = self.stats['distribution_analysis']['categories']
        top_cats = cat_data['top_categories'][:15]

        if not top_cats:
            return None

        categories = [item['category'].replace('Category:', '') for item in top_cats]
        counts = [item['count'] for item in top_cats]

        fig, ax = plt.subplots(figsize=(12, 8))

        bars = ax.barh(categories, counts, color=sns.color_palette("viridis", len(categories)))

        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count, i, f' {count}', va='center', fontsize=9)

        ax.set_xlabel('Count', fontsize=12)
        ax.set_ylabel('Category', fontsize=12)
        ax.set_title('Top 15 Wikipedia Categories', fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        plt.tight_layout()
        output_path = self.output_dir / 'category_coverage.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_quality_tiers(self):
        """Plot quality tier distribution"""
        trends = self.stats['data_quality_insights'].get('quality_trends', {})

        if not trends or 'quality_tiers' not in trends:
            return None

        tiers = trends['quality_tiers']

        tier_names = ['Excellent\n(>= 9)', 'Good\n(7-9)', 'Acceptable\n(5-7)', 'Poor\n(< 5)']
        tier_counts = [tiers['excellent'], tiers['good'], tiers['acceptable'], tiers['poor']]
        colors = ['#27ae60', '#3498db', '#f39c12', '#e74c3c']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart
        bars = ax1.bar(tier_names, tier_counts, color=colors, alpha=0.7, edgecolor='black')
        for bar, count in zip(bars, tier_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom', fontsize=12)

        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Quality Tier Distribution', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Pie chart
        ax2.pie(tier_counts, labels=tier_names, autopct='%1.1f%%',
                startangle=90, colors=colors)
        ax2.set_title('Quality Tier Proportion', fontsize=13, fontweight='bold')

        plt.tight_layout()
        output_path = self.output_dir / 'quality_tiers.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_failure_analysis(self):
        """Plot failure analysis"""
        fa = self.stats['failure_analysis']

        if fa.get('no_failures'):
            return None

        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Failure types
        ax1 = fig.add_subplot(gs[0, 0])
        ft = fa['failure_types']
        types = ['Structure\nFailures', 'Quality\nFailures']
        counts = [ft['structure_failures']['count'], ft['quality_failures']['count']]
        colors_ft = ['#e74c3c', '#f39c12']

        bars = ax1.bar(types, counts, color=colors_ft, alpha=0.7, edgecolor='black')
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom', fontsize=11)

        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title('Failure Types', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Structure errors
        ax2 = fig.add_subplot(gs[0, 1])
        se = fa.get('structure_errors', {})

        if not se.get('no_structure_errors') and 'distribution' in se:
            error_types = [item['error_type'][:30] for item in se['distribution'][:8]]
            error_counts = [item['count'] for item in se['distribution'][:8]]

            bars = ax2.barh(error_types, error_counts, color='coral', alpha=0.7, edgecolor='black')
            for i, (bar, count) in enumerate(zip(bars, error_counts)):
                ax2.text(count, i, f' {count}', va='center', fontsize=9)

            ax2.set_xlabel('Count', fontsize=11)
            ax2.set_title('Top Structure Errors', fontsize=12, fontweight='bold')
            ax2.invert_yaxis()
        else:
            ax2.text(0.5, 0.5, 'No structure errors', ha='center', va='center',
                    fontsize=12, transform=ax2.transAxes)
            ax2.axis('off')

        # Quality issues
        ax3 = fig.add_subplot(gs[1, :])
        qi = fa.get('quality_issues', {})

        if not qi.get('no_quality_issues') and 'top_20_issues' in qi:
            issues = [item['issue'][:50] for item in qi['top_20_issues'][:10]]
            issue_counts = [item['count'] for item in qi['top_20_issues'][:10]]

            bars = ax3.barh(issues, issue_counts, color='salmon', alpha=0.7, edgecolor='black')
            for i, (bar, count) in enumerate(zip(bars, issue_counts)):
                ax3.text(count, i, f' {count}', va='center', fontsize=9)

            ax3.set_xlabel('Count', fontsize=11)
            ax3.set_title('Top 10 Quality Issues', fontsize=12, fontweight='bold')
            ax3.invert_yaxis()
        else:
            ax3.text(0.5, 0.5, 'No quality issues', ha='center', va='center',
                    fontsize=12, transform=ax3.transAxes)
            ax3.axis('off')

        plt.suptitle('Failure Analysis Overview', fontsize=14, fontweight='bold', y=0.995)
        output_path = self.output_dir / 'failure_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_score_correlation(self):
        """Plot correlation heatmap for score dimensions"""
        # This would require raw data, so we'll create a simplified version
        # showing the relative scores across dimensions

        dims = self.stats['quality_metrics'].get('score_dimensions', {})

        if not dims or len(dims) < 2:
            return None

        dim_names = [dim.replace('_score', '').replace('_', ' ').title() for dim in dims.keys()]
        means = [stats['mean'] for stats in dims.values()]

        # Create a simple matrix showing relative scores
        n = len(dim_names)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    # Simulate correlation based on mean similarity
                    matrix[i, j] = 1 - abs(means[i] - means[j]) / 10.0

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(matrix, cmap='coolwarm', vmin=0, vmax=1)

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(dim_names, rotation=45, ha='right')
        ax.set_yticklabels(dim_names)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Similarity Score', fontsize=11)

        # Add text annotations
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)

        ax.set_title('Score Dimension Similarity Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / 'score_correlation.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_validation_dimension_distributions(self):
        """Plot distribution histograms for each validation dimension"""
        dims = self.stats['quality_metrics'].get('score_dimensions', {})

        if not dims or len(dims) == 0:
            return None

        # Calculate number of rows needed
        n_dims = len(dims)
        n_cols = 3
        n_rows = (n_dims + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))

        # Flatten axes array for easier iteration
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()

        for idx, (dim_name, dim_stats) in enumerate(sorted(dims.items())):
            ax = axes_flat[idx]

            # Get histogram data
            bins = dim_stats['histogram']['bin_edges']
            counts = dim_stats['histogram']['counts']

            # Plot histogram
            ax.bar(bins[:-1], counts, width=np.diff(bins), align='edge',
                   color='steelblue', edgecolor='black', alpha=0.7)

            # Add mean and median lines
            ax.axvline(dim_stats['mean'], color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {dim_stats["mean"]:.2f}')
            ax.axvline(dim_stats['percentiles']['p50'], color='green', linestyle='--', linewidth=2,
                      label=f'Median: {dim_stats["percentiles"]["p50"]:.2f}')

            # Format dimension name
            display_name = dim_name.replace('_', ' ').title()
            ax.set_xlabel('Score', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(display_name, fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)

            # Add statistics text
            stats_text = f'n={dim_stats["count"]}\nσ={dim_stats["std"]:.2f}'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Hide unused subplots
        for idx in range(n_dims, len(axes_flat)):
            axes_flat[idx].axis('off')

        plt.suptitle('Validation Dimension Score Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / 'validation_dimension_distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_mc_answer_distribution(self):
        """Plot multiple choice correct answer distribution (A/B/C/D)"""
        mc_quality = self.stats['data_quality_insights'].get('mc_quality', {})

        if mc_quality.get('no_mc_questions'):
            return None

        answer_dist = mc_quality.get('correct_answer_distribution', [])

        if not answer_dist:
            return None

        # Extract data
        answers = [item['answer'] for item in answer_dist]
        counts = [item['count'] for item in answer_dist]
        percentages = [item['percentage'] for item in answer_dist]

        total_mc = mc_quality['total_mc_questions']
        balance_quality = mc_quality.get('answer_balance_quality', 'unknown')

        # Expected percentage if perfectly balanced
        expected_pct = 100.0 / len(answers) if answers else 0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Bar chart with expected line
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c'][:len(answers)]
        bars = ax1.bar(answers, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add count labels on bars
        for bar, count, pct in zip(bars, counts, percentages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Add expected line
        ax1.axhline(total_mc * expected_pct / 100, color='red', linestyle='--', linewidth=2,
                   label=f'Expected (balanced): {expected_pct:.1f}%')

        ax1.set_xlabel('Correct Answer', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title(f'Multiple Choice Correct Answer Distribution\n(n={total_mc}, balance: {balance_quality})',
                     fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Pie chart
        ax2.pie(percentages, labels=answers, autopct='%1.1f%%',
               startangle=90, colors=colors, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax2.set_title('Answer Proportion', fontsize=13, fontweight='bold')

        # Add balance assessment text
        max_deviation = max(abs(pct - expected_pct) for pct in percentages)
        balance_text = f'Max deviation from expected: {max_deviation:.1f}%'
        if max_deviation < 10:
            balance_color = 'green'
            balance_status = 'Well balanced'
        elif max_deviation < 20:
            balance_color = 'orange'
            balance_status = 'Moderately balanced'
        else:
            balance_color = 'red'
            balance_status = 'Imbalanced'

        fig.text(0.5, 0.02, f'{balance_status} - {balance_text}',
                ha='center', fontsize=11, color=balance_color, fontweight='bold')

        plt.tight_layout(rect=[0, 0.05, 1, 1])

        output_path = self.output_dir / 'mc_answer_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_dashboard(self):
        """Create comprehensive dashboard"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

        meta = self.stats['meta']
        dist = self.stats['distribution_analysis']
        qm = self.stats['quality_metrics']

        # 1. Overall stats (text)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        stats_text = f"""
        DATASET OVERVIEW

        Total Examples:    {meta['total_examples']:,}
        Valid:             {meta['total_valid_examples']:,}
        Failed:            {meta['total_failed_examples']:,}
        Success Rate:      {meta['overall_success_rate']:.1f}%

        Unique Strategies: {dist['question_strategies']['unique_values']}
        Unique Categories: {dist['categories']['unique_categories']:,}
        """
        ax1.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')

        # 2. Question types
        ax2 = fig.add_subplot(gs[0, 1])
        qt = dist['question_types']
        types = ['MC', 'Open']
        counts = [qt['multiple_choice']['count'], qt['open_ended']['count']]
        ax2.pie(counts, labels=types, autopct='%1.1f%%', startangle=90,
               colors=['#3498db', '#e74c3c'])
        ax2.set_title('Question Types', fontsize=11, fontweight='bold')

        # 3. Complexity distribution
        ax3 = fig.add_subplot(gs[0, 2])
        comp = dist['complexity']['distribution']
        if comp:
            comp_names = [item['value'] for item in comp]
            comp_counts = [item['count'] for item in comp]
            ax3.bar(comp_names, comp_counts, color=sns.color_palette("husl", len(comp_names)))
            ax3.set_title('Complexity', fontsize=11, fontweight='bold')
            ax3.tick_params(axis='x', labelsize=9)
        else:
            ax3.axis('off')

        # 4. Strategy distribution (top 8)
        ax4 = fig.add_subplot(gs[1, :])
        strat = dist['question_strategies']['distribution'][:8]
        if strat:
            strat_names = [item['value'][:25] for item in strat]
            strat_counts = [item['count'] for item in strat]
            bars = ax4.barh(strat_names, strat_counts,
                           color=sns.color_palette("husl", len(strat_names)))
            ax4.invert_yaxis()
            ax4.set_title('Top 8 Question Strategies', fontsize=11, fontweight='bold')
            ax4.set_xlabel('Count')
            for i, (bar, count) in enumerate(zip(bars, strat_counts)):
                ax4.text(count, i, f' {count}', va='center', fontsize=8)

        # 5. Score distribution
        ax5 = fig.add_subplot(gs[2, 0])
        scores = qm['validation_scores']
        if scores['count'] > 0:
            bins = scores['histogram']['bin_edges']
            counts = scores['histogram']['counts']
            ax5.bar(bins[:-1], counts, width=np.diff(bins), align='edge',
                   color='mediumseagreen', alpha=0.7)
            ax5.axvline(scores['mean'], color='red', linestyle='--', linewidth=2)
            ax5.set_title(f'Validation Scores (μ={scores["mean"]:.2f})',
                         fontsize=11, fontweight='bold')
            ax5.set_xlabel('Score')
            ax5.set_ylabel('Count')

        # 6. Response length
        ax6 = fig.add_subplot(gs[2, 1])
        rl = dist['response_length']
        if rl['count'] > 0:
            bins = rl['histogram']['bin_edges']
            counts = rl['histogram']['counts']
            ax6.bar(bins[:-1], counts, width=np.diff(bins), align='edge',
                   color='skyblue', alpha=0.7)
            ax6.axvline(rl['mean'], color='red', linestyle='--', linewidth=2)
            ax6.set_title(f'Response Length (μ={rl["mean"]:.0f} words)',
                         fontsize=11, fontweight='bold')
            ax6.set_xlabel('Word Count')
            ax6.set_ylabel('Count')

        # 7. Quality tiers
        ax7 = fig.add_subplot(gs[2, 2])
        trends = self.stats['data_quality_insights'].get('quality_trends', {})
        if trends and 'quality_tiers' in trends:
            tiers = trends['quality_tiers']
            tier_names = ['Excellent', 'Good', 'OK', 'Poor']
            tier_counts = [tiers['excellent'], tiers['good'], tiers['acceptable'], tiers['poor']]
            colors = ['#27ae60', '#3498db', '#f39c12', '#e74c3c']
            ax7.pie(tier_counts, labels=tier_names, autopct='%1.0f%%',
                   startangle=90, colors=colors)
            ax7.set_title('Quality Tiers', fontsize=11, fontweight='bold')
        else:
            ax7.axis('off')

        plt.suptitle('Data Construction Pipeline Dashboard', fontsize=16, fontweight='bold')

        output_path = self.output_dir / 'dashboard.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate visualizations for data statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'stats_json',
        type=str,
        help='Path to data_statistics.json file'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for visualizations (default: <stats_dir>/visualizations)'
    )

    args = parser.parse_args()

    if not HAS_VIZ_LIBS:
        print("\nERROR: Required libraries not installed")
        print("Install with: pip install matplotlib seaborn numpy")
        sys.exit(1)

    # Check if stats file exists
    stats_path = Path(args.stats_json)
    if not stats_path.exists():
        print(f"\nERROR: Statistics file not found: {stats_path}")
        print("\nGenerate statistics first with:")
        print("  python3 statistics_pipeline.py <output_dir>")
        sys.exit(1)

    # Generate visualizations
    viz = DataVisualizer(args.stats_json, args.output_dir)
    success = viz.generate_all_visualizations()

    if success:
        print("\nDone!")
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
