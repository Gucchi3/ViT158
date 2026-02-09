# TERNARY_VIT: CUDA カーネルと detach フォールバックの動作まとめ

このファイルは、TerLinearCUDA における
1) CUDA カーネル経路
2) STE + detach フォールバック経路
の違いと使い分けを、できるだけ分かりやすく整理したメモです。

---

## 全体像

- **CUDA カーネル経路**: int8 + ternary GEMM を専用 CUDA 拡張で実行。勾配は **STE (Straight-Through Estimator)** で近似。
- **フォールバック経路**: PyTorch 上で「量子化 → 逆量子化 → 線形演算」を行い、
	勾配は **STE (Straight-Through Estimator)** で近似。

---

## フォールバック経路 (STE + detach) の詳細

### 1. 量子化と逆量子化を float で行う

フォールバックでは、実際の線形演算は **float** で行います。
ただし、入力や重みは一度「量子化してから逆量子化」されます。

流れは次の通りです:

1. 入力 `x` を正規化 (`LayerNorm`) した後、8-bit 量子化・逆量子化。
2. 重み `w` を ternary ( -1, 0, +1 ) に量子化・逆量子化。
3. 逆量子化された `x` と `w` を float の `F.linear` に渡して乗算。

ここで重要なのは、**「実際に乗算するのは float に戻した値」**という点です。
つまり「int8 をそのまま掛ける」わけではなく、
**量子化された値を float に戻してから演算**しています。

### 2. detach とは何か

`detach()` は **計算グラフから切り離す** PyTorch の操作です。
これにより、量子化の「丸め・クリップ」などの非微分操作が
**勾配計算に影響しない**ようにできます。

実装上は次のような式になっています:

```
x_quant = x_norm + (x_q - x_norm).detach()
w_quant = w + (w_q - w).detach()
```

意味としては:

- **順伝播**: `x_q` や `w_q` を使う (量子化済み値)
- **逆伝播**: `x_norm` や `w` を使う (量子化を無視)

つまり **前向きは量子化、後ろ向きは元の値** という挙動になります。
これが STE (Straight-Through Estimator) です。

---

## CUDA カーネル経路の詳細

CUDA 経路では、以下を行います:

1. 入力 `x_norm` を **int8** に量子化 (absmax, per-token)。
2. 重み `w` を **ternary int8** に量子化 (absmean)。
3. 専用 CUDA kernel (`ternary_qkv_gemm`) で GEMM。
4. 出力を **float に逆量子化** して返す。

ここでの逆量子化は、

```
y_float = y_int * (alpha / x_scale)
```

という式に従っています。
この `alpha` は重みの absmean、`x_scale` は入力の absmax スケールです。

---

## 使い分けの条件

### 1. USE_CUDA_KERNEL

`config.json` の `USE_CUDA_KERNEL` が **1 のときだけ** CUDA カーネルをビルドします。
0 の場合は、そもそも JIT ビルドが走りません。

### 2. forward 時の分岐条件

CUDA 経路を使う条件は次の 2 つです:

- `_qkv_kernel` がロード済みであること（コンパイルが可能であるということ）
- `x_norm.is_cuda` が True であること

この 2 つを満たさない場合は、必ずフォールバックになります。

---

## まとめ

- **USE_CUDA_KERNEL=1** かつ **CUDA が利用可能** ならカーネル経路。
- それ以外は **STE + detach** のフォールバック。
- フォールバックは **量子化 → 逆量子化 → float 乗算**。
- `detach()` により、**順伝播は量子化値、逆伝播は元値** になる。

---

