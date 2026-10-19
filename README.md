# Current Phase:Spatial-Allocation
"Algorithmic spatial allocation and coordinate determination for semantic event networks and cognitive topology."

## Micro—ACT-R (tentative)
> "ACT-R at a Micro-Scale"
## What makes this different from standard ACT-R?

Standard ACT-R operates on a macro level with production rules. This project zooms into the micro-level of cognitive processing.

## Overview
**Micro–ACT-R operates through a clear, fully explainable pipeline from input to reasoning and execution:**

First, text is acquired from the **Environment** and parsed in the **Input** stage into individual cognitive **particles** (Particle Encapsulation).

Next, during **Reasoning**, the system uses **Topological Mapping** to define relationships between particles and assigns explicit spatial coordinates via **Spatial Allocation**. These coordinates are logged directly into **Storage**.

The system then validates consistency between the category of knowledge and past experiences **(Match & Select)**, selects the appropriate action based on this consistency **(Execution)**, and finally produces the **Output**.

## Micro-ACT-R Pipeline Architecture
<img width="960" height="540" alt="image" src="https://github.com/user-attachments/assets/89f34fcc-b544-4fc1-b36d-fccdd794e3dc" />

1. [input-parser](https://github.com/ao-labs-123/input-parser)
2. [particle-encapsulation](https://github.com/ao-labs-123/particle-encapsulation)
3. [topological-mapper](https://github.com/ao-labs-123/topological-mapper)
4. [spatial-allocation](https://github.com/ao-labs-123/spatial-allocation)※Current Position

## Repository Structure
```text
Spatial-allocation/
├── README.md                  # 設計思想・仕様・使い方
├── requirements.txt           # 依存ライブラリ (numpy, networkx, pyvis など)
├── pyproject.toml             # パッケージ設定
│
├── data/                      # テスト用・入出力用のデータ置き場
│   └── topological_graph.json # topological-mapper から受け取る入力サンプル
│
├── src/                       # コアロジック
│   ├── __init__.py
│   ├── loader.py              # トポロジーJSONの読み込み・ノード/エッジ構造のパース
│   ├── allocator.py           # 空間配置アルゴリズム（力学モデル、座標計算、Z軸展開など）
│   └── exporter.py            # 座標付きグラフデータ（JSON等）の出力
│
├── visualize/                 # 可視化・確認用スクリプト
│   └── render_3d.py           # 3D/2Dでの空間配置確認用（Three.jsやPlotly等への連携）
│
└── tests/                     # 単体テスト
    └── test_allocator.py
```
