# model

このフォルダには複数のモデル実装が入っています。

新しいモデルを追加する場合は、model直下にモデル名のフォルダを作成し、
その中にモデル本体のPythonファイルを用意してください。
また、main.pyで `from model.xxx import ClassName` のように使えるよう、
各モデルフォルダに `__init__.py` を作成して公開クラス/関数を定義して下さい。

現在定義されているモデル:
- vit      : 通常のVision Transformer 
- ter_vit  : 3値量子化のVision Transformer (TerLinearを使用)
- tiny_cnn : 小規模CNN
