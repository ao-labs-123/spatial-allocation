# Current Phase:Spatial-Allocation
"Algorithmic spatial allocation and coordinate determination for semantic event networks and cognitive topology."

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
