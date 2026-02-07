# ViT158
3値量子化ViT

## 現象
cudaカーネルコンパイル時に、コンパイル・リンカエラーが発生して、実行に失敗する。

### 環境
OS: Windows 11
Visual Studio 2022 Community
MSVC v143:
14.39.33519（14.3x 系）
14.44.35207（14.4x 系）
Python 3.12
CUDA Toolkit 13.1

PyTorch (CUDA 13.0 相当の wheel)
### 原因
おそらくPytorch側でコンパイルされたもの(14.39側)に対し、新しいversionのC++コンパイラ(14.4側)を用いてコンパイルしてリンクしようとするとエラーが発生する。  

### 対処法
C++コンパイラのほうのversionを14.4系ではなく14.39系を使用する。  
以下２番のコマンドを使用しx64 Native Tools Command Promptに入る。  
次に、１番のコマンドで14.39系のMSVCが入っていることを確認する。  
３番で14.39系にPathが使用されることを確認する。  
仮想環境をactivateし、main.pyを実行する。  
ここでエラーが出る場合は、その他１番のフォルダへ行き、qkv_kernelフォルダごと削除し、main.pyを再実行する。

### 使用コマンド
１．ダウンロードされているC++コンパイラのMVCC versionを確認するコマンド
```python
dir "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC"
```

２．x64 Native Tools Command Prompt for VS 2022を開くときに使用するC++コンパイラのversionを指定するコマンド
```python
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.39
```

３．Pathが通っているC++コンパイラのversionを確認するコマンド
```python
echo %VCToolsVersion%
```

４．clとlinkのPathを確認するコマンド
```python
where cl
where link
```

### その他
１．コンパイルした.O等のファイルが格納されているPath
```python
C:\Users\<user>\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\Local\torch_extensions\torch_extensions\Cache\py312_cu130\qkv_kernel
```