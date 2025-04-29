# Half-edge data structure construction on GPU: 

[ハーフエッジデータ構造](https://en.wikipedia.org/wiki/Doubly_connected_edge_list)の構築をGPU上で行うコードサンプルです。\
\
This is a sample code to construct a [half-edge data structure](https://en.wikipedia.org/wiki/Doubly_connected_edge_list) (also known as doubly connected edge list) on GPU.

## 動作環境 / Confirmed Environment
現状以下の環境で動作を確認しています。\
I've confirmed that the program runs correctly on the following environment.

* Windows 11 (24H2) & Visual Studio 2022 (17.13.6)
* Ryzen 9 7950X, 64GB, RTX 4080 16GB
* NVIDIA Driver 576.02

動作させるにあたっては以下のライブラリが必要です。\
It requires the following libraries.

* CUDA 12.8.1 (cubd might also be built with bit older CUDA version 11.0-)\
  Note that CUDA has compilation issues with Visual Studio 2022 17.10.0.

## ライセンス / License
Released under the Apache License, Version 2.0 (see [LICENSE.md](LICENSE.md))

----
2025 [@Shocker_0x15](https://twitter.com/Shocker_0x15), [@bsky.rayspace.xyz](https://bsky.app/profile/bsky.rayspace.xyz)
