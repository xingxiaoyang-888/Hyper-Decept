"""
run_ablation.py — 消融实验统一入口

运行完整流程 (独立性验证 → 消融实验 → 可视化)：
  python -m ablation.run_ablation

仅运行独立性验证：
  python -m ablation.run_ablation --skip-ablation

仅运行消融实验 (跳过独立性)：
  python -m ablation.run_ablation --skip-independence

自定义数据路径：
  python -m ablation.run_ablation --db "path/to/data.db" --csv "path/to/labels.csv"

调整交叉验证参数：
  python -m ablation.run_ablation --repeats 3 --folds 5
"""

import argparse
import os
import sys

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="情感模块消融实验 & 独立性验证",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # 数据路径
    parser.add_argument("--db", type=str, default=None,
                        help="SQLite 数据库路径 (默认使用 main_detector.DB_FILE)")
    parser.add_argument("--csv", type=str, default=None,
                        help="CSV 标签文件路径 (默认使用 main_detector.CSV_FILE)")
    
    # 交叉验证
    parser.add_argument("--repeats", type=int, default=5,
                        help="RepeatedStratifiedKFold 重复次数 (default: 5)")
    parser.add_argument("--folds", type=int, default=5,
                        help="K 折数 (default: 5)")
    
    # 跳过选项
    parser.add_argument("--skip-independence", action="store_true",
                        help="跳过独立性验证，仅运行消融实验")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="跳过消融实验，仅运行独立性验证")
    
    # 输出
    parser.add_argument("--output", type=str, default="./ablation_results",
                        help="结果输出目录 (default: ./ablation_results)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print("\n" + "★" * 65)
    print("   情感模块消融实验与独立性验证工具")
    print("★" * 65)
    
    from ablation.ablation_experiment import run_ablation
    from ablation.ablation_plot import plot_all
    
    # 执行消融实验 (内部会先调用独立性验证)
    ablation_results = run_ablation(
        db_path=args.db,
        csv_path=args.csv,
        n_repeats=args.repeats,
        n_splits=args.folds,
        save_dir=args.output,
        skip_independence=args.skip_independence,
        skip_ablation=args.skip_ablation,
    )
    
    # 如果有消融结果，自动绘制柱状图
    if ablation_results:
        from ablation.ablation_plot import plot_ablation_bar_chart
        plot_ablation_bar_chart(
            ablation_results,
            save_path=os.path.join(args.output, "ablation_bar_chart.png")
        )
    
    print("\n" + "★" * 65)
    print(f"   全部完成！结果已保存至: {os.path.abspath(args.output)}")
    print("★" * 65 + "\n")


if __name__ == "__main__":
    main()