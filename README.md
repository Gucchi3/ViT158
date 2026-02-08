# ViT158：CUDA カーネルビルドの既知問題

## 概要

本リポジトリでは、Windows 環境で PyTorch の CUDA 拡張（`qkv_kernel` など）をビルドする際、  
MSVC のバージョン不整合によりコンパイル・リンクエラーが発生し、実行に失敗する既知の問題があります。

---

## 現象

CUDA カーネルコンパイル時に、コンパイル／リンクエラーが発生し、`main.py` 実行時に失敗します。

典型的なエラーメッセージ例（抜粋）:

```text
qkv_kernel.o : error LNK2019: 未解決の外部シンボル __std_search_1 ...
qkv_kernel.o : error LNK2019: 未解決の外部シンボル __std_find_end_1 ...
qkv_kernel.o : error LNK2019: 未解決の外部シンボル __std_find_first_not_of_trivial_pos_1 ...
qkv_kernel.o : error LNK2019: 未解決の外部シンボル __std_find_last_not_of_trivial_pos_1 ...
qkv_kernel.pyd : fatal error LNK1120: N 件の未解決の外部参照
ninja: build stopped: subcommand failed.
```

---

## 環境

- **OS**: Windows 11
- **IDE**: Visual Studio 2022 Community
- **MSVC v143**:
  - 14.39.33519（14.3x 系）
  - 14.44.35207（14.4x 系）
- **Python**: 3.12
- **CUDA Toolkit**: 13.1
- **PyTorch**: CUDA 13.0 相当の wheel

---

## 原因

PyTorch の公式 Windows バイナリ（`torch_cpu.lib`, `torch_cuda.lib` など）は、  
**MSVC 14.3x 系 (例: 14.39)** でビルドされています。

一方、Visual Studio 2022 の自動更新などにより、環境側に **MSVC 14.4x 系 (例: 14.44.35207)** が導入され、  
そちらを使って CUDA 拡張（`qkv_kernel`）をビルドすると、

- **PyTorch ライブラリ**: MSVC 14.3x 系でビルド
- **CUDA 拡張**: MSVC 14.4x 系でビルド

という **異なる MSVC バージョンの組み合わせ** になり、  
C++ 標準ライブラリ (STL) の内部シンボル（`__std_search_1` など）が一致せず、LNK2019 が発生します。

### 技術的背景

- MSVC 17.10 以降（14.4x 系）では、STL のベクトル化アルゴリズム（`std::find_*`, `std::search` など）の内部実装が変更されました。
- これにより、14.3x でビルドされたライブラリと 14.4x でビルドしたコードを混在させると、内部シンボルの ABI 不一致が発生します。
- Visual Studio は自動更新や Windows Update 経由で、ユーザー操作なしに 17.10 / 14.4x に上がることがあります。

---

## 対処法（推奨手順）

### 手順の全体像

1. **MSVC 14.39 系がインストールされていることを確認**
2. **ビルド時に MSVC 14.39 系を明示的に選択**
3. **必要に応じて PyTorch 拡張のビルドキャッシュを削除**
4. 仮想環境を有効化し、`main.py` を実行

以下、各ステップの詳細です。

---

### ステップ 1: インストール済み MSVC バージョンの確認

コマンドプロンプトで以下を実行：

```bat
dir "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC"
```

**出力例:**

```text
2026/02/08  03:52    <DIR>          14.39.33519
2026/02/08  03:46    <DIR>          14.44.35207
```

`14.39.xxxxx` と `14.44.xxxxx` の両方が存在することを確認します。

**もし 14.39 系がない場合:**

Visual Studio Installer を開き、「個別のコンポーネント」タブから以下を追加インストール：

- **MSVC v143 - VS 2022 C++ x64/x86 ビルドツール (v14.39-17.9)**

---

### ステップ 2: x64 Native Tools Command Prompt で MSVC 14.39 を選択

スタートメニュー → Visual Studio 2022 → **x64 Native Tools Command Prompt for VS 2022** を起動。

以下のコマンドで **MSVC 14.39** を使用するように環境を初期化します：

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.39
```

---

### ステップ 3: 使用されているコンパイラのバージョン確認

初期化後、以下のコマンドで **現在有効な MSVC のバージョンとパス** を確認します：

```bat
echo %VCToolsVersion%
where cl
where link
```

**確認ポイント:**

- `echo %VCToolsVersion%` の結果が `14.39.33519` など 14.3x 系であること
- `where cl` の**先頭行**が以下のようになっていること：

  ```text
  C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64\cl.exe
  ```

- `where link` の**先頭行**が以下のようになっていること：

  ```text
  C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64\link.exe
  ```

※ 複数行表示される場合でも、**先頭に 14.39 側が来ているか**を必ず確認してください。

---

### ステップ 4: 仮想環境を有効化し、`main.py` を実行

プロジェクトルートに移動し、仮想環境を有効化してから `main.py` を実行します：

```bat
cd C:\Users\<user>\Desktop\ViT-1.58b\ViT158
.venv\Scripts\activate
python main.py
```

---

### ステップ 5: それでも LNK2019 が出る場合：キャッシュの削除

MSVC のバージョンを切り替えた後でも、  
PyTorch の CUDA 拡張ビルドキャッシュに **古いバージョンでコンパイルされた `.o` / `.pyd`** が残っていると、  
それが再利用されてエラーが続くことがあります。

その場合は、以下のフォルダを削除してから再度 `main.py` を実行してください。

#### CUDA 拡張のビルドキャッシュパス

```text
C:\Users\<user>\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\Local\torch_extensions\torch_extensions\Cache\py312_cu130\qkv_kernel
```

#### 削除コマンド（例）

```bat
rmdir /S /Q "C:\Users\<user>\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\Local\torch_extensions"
```

その後、再度:

```bat
python main.py
```

を実行します。

---

## 使用コマンド一覧（クイックリファレンス）

### 1. インストール済み MSVC バージョンの一覧

```bat
dir "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC"
```

### 2. MSVC 14.39 を使用するように環境を初期化

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.39
```

### 3. 使用中のツールセットバージョンを確認

```bat
echo %VCToolsVersion%
```

### 4. `cl` / `link` の実パスを確認

```bat
where cl
where link
```

### 5. CUDA 拡張ビルドキャッシュの削除（必要な場合）

```bat
rmdir /S /Q "C:\Users\<user>\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\Local\torch_extensions"
```

---

## 補足事項

### 環境の独立性について

- **x64 Native Tools Command Prompt** は、**ウィンドウごとに環境が独立**しています。
- ウィンドウを閉じると、`vcvarsall.bat` で設定した内容はリセットされます。
- 新しくウィンドウを開いたときは、**毎回必ず以下を実行**してください：

  ```bat
  call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.39
  ```

### ビルド成功の条件

上記の手順を守ることで、以下の組み合わせで統一されます：

- **PyTorch 本体**: MSVC 14.3x でビルド
- **CUDA 拡張**: MSVC 14.39 でビルド

これにより、`__std_search_1` などの STL 内部シンボルが一致し、LNK2019 エラーを防ぐことができます。

---

## トラブルシューティング

### Q1: `vcvarsall.bat` を実行してもエラーが出る

**A:** Visual Studio のインストールパスが異なる場合があります。以下で確認：

```bat
dir "C:\Program Files\Microsoft Visual Studio\2022"
```

`Community` 以外に `Professional` や `Enterprise` がある場合は、パスを適宜変更してください。

### Q2: `where cl` で 14.44 が先頭に来てしまう

**A:** 以下を確認：

1. `vcvarsall.bat` を実行したウィンドウで作業しているか
2. `echo %VCToolsVersion%` が 14.39 系を示しているか
3. 別のウィンドウやシェルで作業していないか

### Q3: キャッシュを削除しても解決しない

**A:** 以下を試してください：

1. コマンドプロンプトを完全に閉じる
2. 新しく **x64 Native Tools Command Prompt** を開く
3. `vcvarsall.bat ... -vcvars_ver=14.39` を再実行
4. キャッシュを再度削除
5. `python main.py` を実行

---

## 関連情報

- [Microsoft DevBlogs: MSVC Toolset Minor Version Number 14.40 in VS 2022 v17.10](https://devblogs.microsoft.com/cppblog/msvc-toolset-minor-version-number-14-40-in-vs-2022-v17-10/)
- [Boost Issue #914: No support for msvc-toolset 14.4x](https://github.com/boostorg/boost/issues/914)
- [GitHub Actions Issue #6629: Unresolved external symbols in C++ stdlib](https://github.com/actions/runner-images/issues/6629)

---

## まとめ

この問題の本質は、**PyTorch がビルドされた MSVC ツールセット（14.3x）と、ローカル環境の MSVC ツールセット（14.4x）のバージョン不整合**にあります。

解決には以下の 3 つが必要です：

1. **14.3x 系ツールセットの追加インストール**
2. **ビルド時に 14.39 を明示的に指定**
3. **古いビルドキャッシュの完全削除**

この手順により、PyTorch の CUDA 拡張を安定してビルドできるようになります。
