import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


INDEX_CSV_DEFAULT = "evaluate/results/index.csv"
OUTPUT_DIR_DEFAULT = "evaluate/summary"


@dataclass
class RunRecord:
    timestamp: str
    group: str
    checkpoint: str
    output_json: Optional[str]
    log_file: Optional[str]


def read_index(index_csv: str) -> List[RunRecord]:
    records: List[RunRecord] = []
    if not os.path.exists(index_csv):
        raise FileNotFoundError(f"未找到索引文件: {index_csv}")
    with open(index_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                RunRecord(
                    timestamp=row.get('timestamp', ''),
                    group=row.get('group', ''),
                    checkpoint=row.get('checkpoint', ''),
                    output_json=row.get('output_json') or None,
                    log_file=row.get('log_file') or None,
                )
            )
    return records


def load_metrics_from_result(json_path: str) -> Optional[Dict]:
    if not json_path or not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        summary = data.get('summary', {})
        # 取关键指标
        return {
            'overall_accuracy': summary.get('overall_stats', {}).get('accuracy'),
            'overall_errors': summary.get('overall_stats', {}).get('error_count'),
            'overall_avg_time': summary.get('overall_stats', {}).get('avg_response_time'),
            'knowledge_accuracy': summary.get('knowledge_questions', {}).get('accuracy'),
            'bridge_accuracy': summary.get('bridge_questions', {}).get('accuracy'),
            'multimodal_accuracy': summary.get('multimodal_questions', {}).get('accuracy'),
            'total_questions': summary.get('total_questions'),
            'model_name': data.get('metadata', {}).get('model_name'),
            'input_file': data.get('metadata', {}).get('input_file'),
            'processing_time': data.get('metadata', {}).get('total_processing_time'),
        }
    except Exception:
        return None


def build_dataframe(records: List[RunRecord]) -> pd.DataFrame:
    rows: List[Dict] = []
    for r in records:
        metrics = load_metrics_from_result(r.output_json) if r.output_json else None
        row = {
            'timestamp': r.timestamp,
            'group': r.group,
            'checkpoint': r.checkpoint,
            'output_json': r.output_json,
            'log_file': r.log_file,
        }
        if metrics:
            row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def save_group_tables(df: pd.DataFrame, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # 保存总表
    csv_path = os.path.join(out_dir, 'all_runs.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')

    # 每组单独保存
    for group, gdf in df.groupby('group'):
        slug = sanitize(group)
        gpath = os.path.join(out_dir, f'group_{slug}.csv')
        gdf.to_csv(gpath, index=False, encoding='utf-8')


def sanitize(name: str) -> str:
    return ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in name.strip())


def plot_group_bars(df: pd.DataFrame, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid')

    # 1) 按组绘制 overall_accuracy 柱状图
    for group, gdf in df.groupby('group'):
        gdf = gdf.copy()
        gdf = gdf.sort_values(by='overall_accuracy', ascending=False)
        plt.figure(figsize=(12, 5))
        ax = sns.barplot(
            data=gdf,
            x='checkpoint', y='overall_accuracy', color='#4C78A8'
        )
        ax.set_title(f'{group} - Overall Accuracy')
        ax.set_xlabel('Checkpoint')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.2%}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, xytext=(0, 3), textcoords='offset points')
        plt.tight_layout()
        out_png = os.path.join(out_dir, f'{sanitize(group)}__overall_accuracy.png')
        plt.savefig(out_png, dpi=200)
        plt.close()

    # 2) 三类题型准确率并排柱状图（按组）
    melted = df.melt(
        id_vars=['group', 'checkpoint'],
        value_vars=['knowledge_accuracy', 'bridge_accuracy', 'multimodal_accuracy'],
        var_name='metric', value_name='accuracy'
    )
    for group, gdf in melted.groupby('group'):
        plt.figure(figsize=(13, 5))
        ax = sns.barplot(
            data=gdf,
            x='checkpoint', y='accuracy', hue='metric'
        )
        ax.set_title(f'{group} - Accuracy by Question Type')
        ax.set_xlabel('Checkpoint')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
        plt.legend(title='Metric')
        plt.tight_layout()
        out_png = os.path.join(out_dir, f'{sanitize(group)}__type_accuracy.png')
        plt.savefig(out_png, dpi=200)
        plt.close()


def generate_markdown_report(df: pd.DataFrame, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    md_path = os.path.join(out_dir, 'REPORT.md')

    lines: List[str] = []
    lines.append('# 分组结果概览\n')
    lines.append(f'- 运行总数: {len(df)}\n')

    # 每个组输出一个简表：checkpoint 与 overall_accuracy
    for group, gdf in df.groupby('group'):
        lines.append(f'\n## {group}\n')
        gdf = gdf.sort_values(by='overall_accuracy', ascending=False)
        lines.append('| Checkpoint | Overall Acc | Knowledge | Bridge | Multimodal | Errors | Time(s) |\n')
        lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n')
        for _, row in gdf.iterrows():
            lines.append(
                f"| {row.get('checkpoint','')} | "
                f"{(row.get('overall_accuracy') or 0):.2%} | "
                f"{(row.get('knowledge_accuracy') or 0):.2%} | "
                f"{(row.get('bridge_accuracy') or 0):.2%} | "
                f"{(row.get('multimodal_accuracy') or 0):.2%} | "
                f"{int(row.get('overall_errors') or 0)} | "
                f"{float(row.get('processing_time') or 0):.2f} |\n"
            )

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, default=INDEX_CSV_DEFAULT, help='批量运行生成的索引 CSV 路径')
    parser.add_argument('--out', type=str, default=OUTPUT_DIR_DEFAULT, help='输出目录')
    args = parser.parse_args()

    records = read_index(args.index)
    df = build_dataframe(records)

    # 丢弃没有解析到 summary 的记录
    if 'overall_accuracy' in df.columns:
        df = df[~df['overall_accuracy'].isna()].copy()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    # 保存数据表
    save_group_tables(df, args.out)

    # 绘图
    plot_dir = os.path.join(args.out, 'plots')
    plot_group_bars(df, plot_dir)

    # 报告
    generate_markdown_report(df, args.out)

    print(f"汇总完成：\n- 表格: {args.out}\n- 图表: {plot_dir}\n- 报告: {os.path.join(args.out, 'REPORT.md')}")


if __name__ == '__main__':
    main()


