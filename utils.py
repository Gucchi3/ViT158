import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from timm.data.mixup import Mixup
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms_factory import create_transform
from timm.loss.cross_entropy import SoftTargetCrossEntropy
from timm.scheduler.cosine_lr import CosineLRScheduler
from torcheval.metrics import MulticlassAccuracy


class tools:
    """様々な汎用tool"""
    @staticmethod
    def init_setting(config_path, model=None):
        """、デバイス設定・シード固定・config読み込み・フォルダ作成"""
        # config読み込み
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f) 
        
        # シード固定
        seed = config["SEED"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # デバイス
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 学習用ユニークフォルダ作成（タイムスタンプ）
        os.makedirs(config["LOG_DIR"], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(config["LOG_DIR"], timestamp)
        os.makedirs(run_dir, exist_ok=True)
        config["RUN_DIR"] = run_dir

        return device, config
    
    @staticmethod
    def load_weight(model, device, config):
        """保存済み重みの読み込み（config設定に基づく）"""
        if config["LOAD_WEIGHT"] == 0:
            return model
        
        weight_path = config["LOAD_WEIGHT_PATH"]
        if not os.path.exists(weight_path):
            print(f"Error: Weight file not found at {weight_path}")
            exit()
        
        try:
            model.load_state_dict(torch.load(weight_path, map_location=device))
            print(f"Success: Loaded weights from {weight_path}")
        except Exception as e:
            print(f"Error: Failed to load weights — {e}")
            exit()
        
        return model

    @staticmethod
    def print_training_info(model, device, config):
        """学習開始時の設定情報を表示"""
        # パラメータ数計算
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # パッチ数・シーケンス長計算
        num_patches = (config["IMG_SIZE"] // config["PATCH_SIZE"]) ** 2
        seq_length = num_patches + 1  # +1 for cls_token
        
        # GPU情報
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
        else:
            gpu_name = "CPU"
        
        table = Table(title="Training Configuration", show_header=True, header_style="bold")
        table.add_column("Section", style="cyan", no_wrap=True)
        table.add_column("Key", style="green")
        table.add_column("Value", style="white")

        # 環境情報
        table.add_row("Environment", "Device", f"{device} ({gpu_name})")
        table.add_row("Environment", "PyTorch", torch.__version__)
        table.add_row("Environment", "CUDA Kernel", "Enabled" if config["USE_CUDA_KERNEL"] else "Disabled")
        table.add_row("Environment", "Seed", str(config["SEED"]))

        # モデル構造
        table.add_row("Model", "Image Size", str(config["IMG_SIZE"]))
        table.add_row("Model", "Patch Size", str(config["PATCH_SIZE"]))
        table.add_row("Model", "Num Patches", f"{num_patches} (+1 cls = {seq_length})")
        table.add_row("Model", "Embed Dim", str(config["EMBED_DIM"]))
        table.add_row("Model", "Depth", str(config["DEPTH"]))
        table.add_row("Model", "Num Heads", str(config["NUM_HEADS"]))
        table.add_row("Model", "MLP Ratio", str(config["MLP_RATIO"]))
        table.add_row("Model", "Total Params", f"{total_params:,} ({total_params/1e6:.2f}M)")
        table.add_row("Model", "Trainable", f"{trainable_params:,}")

        # 学習パラメータ
        table.add_row("Training", "Epochs", str(config["EPOCHS"]))
        table.add_row("Training", "Batch Size", str(config["BATCH_SIZE"]))
        table.add_row("Training", "Optimizer", config["OPTIMIZER"].upper())
        table.add_row("Training", "Learning Rate", str(config["LEARNING_RATE"]))
        table.add_row("Training", "Weight Decay", str(config["WEIGHT_DECAY"]))
        table.add_row("Training", "Warmup Epochs", str(config["WARMUP_EPOCHS"]))
        table.add_row("Training", "LR Min", str(config["LR_MIN"]))

        # データ拡張
        table.add_row("Augmentation", "Mixup α", str(config["MIXUP_ALPHA"]))
        table.add_row("Augmentation", "CutMix α", str(config["CUTMIX_ALPHA"]))
        table.add_row("Augmentation", "Label Smooth", str(config["LABEL_SMOOTHING"]))
        table.add_row("Augmentation", "AutoAugment", str(config["AUTO_AUGMENT"]))

        # 出力先
        table.add_row("Output", "Run Directory", str(config["RUN_DIR"]))

        print()
        print(table)
        print()

    @staticmethod
    def make_loader_cifar10(config):
        """CIFAR-10 の DataLoader を作成（vit-cifar100準拠の前処理）"""
        CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
        # auto_augmentの処理（"none"や空文字列はNoneに変換）
        auto_aug = config["AUTO_AUGMENT"]
        if auto_aug in ("none", "None", ""):
            auto_aug = None
        # transform定義
        train_transform = transforms.Compose([
            transforms.Resize(config["IMG_SIZE"], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(config["IMG_SIZE"]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(-30, 30)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN,CIFAR10_STD)
            
            # input_size=(config["IN_CHANS"], config["IMG_SIZE"], config["IMG_SIZE"]),
            # is_training=True,
            # color_jitter=config["COLOR_JITTER"],
            # auto_augment=auto_aug,
            # interpolation=config["INTERPOLATION"],
            # mean=CIFAR10_MEAN,
            # std=CIFAR10_STD,
            # re_prob=config["RE_PROB"],
            # re_mode=config["RE_MODE"],
            # re_count=config["RE_COUNT"],
        ])
        test_transform = transforms.Compose([
            transforms.Resize(config["IMG_SIZE"], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(config["IMG_SIZE"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ])
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        # Dataset作成
        train_set = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=test_transform)
        # Dataloader作成
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=config["BATCH_SIZE"], shuffle=True,
            num_workers=config["NUM_WORKERS"], pin_memory=True, drop_last=True,
            persistent_workers=True)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=config["BATCH_SIZE"], shuffle=False,
            num_workers=config["NUM_WORKERS"], pin_memory=True, drop_last=False,
            persistent_workers=True)
        # 使用するクラス定義
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
        # return
        return train_loader, test_loader, classes
    
    @staticmethod
    def save_curves(train_losses, test_losses, test_accs, config):
        """訓練曲線をプロットして保存"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(train_losses, label="train"); ax1.plot(test_losses, label="test")
            ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend(); ax1.grid()
            ax2.plot(test_accs, label="test acc", color="r")
            ax2.set_xlabel("epoch"); ax2.set_ylabel("acc"); ax2.legend(); ax2.grid()
            curves_path = os.path.join(config["RUN_DIR"], "curves.png")
            fig.savefig(curves_path)
            print(f"Curves saved at {curves_path}")
        except Exception as e:
            print("#" * 50)
            print(f"Failed to save curves: {e}")
            print("#" * 50)

    @staticmethod
    def save_model(model, config):
        """モデルの重みをpth形式で保存"""
        try:
            pth_path = os.path.join(config["RUN_DIR"], "model.pth")
            torch.save(model.state_dict(), pth_path)
            print(f"Model saved at {pth_path}")
        except Exception as e:
            print("#" * 50)
            print(f"Failed to save model: {e}")
            print("#" * 50)

    @staticmethod
    def save_onnx(model, config, device):
        """モデルをONNX形式でエクスポート"""
        model.eval()
        dummy_input = torch.randn(1, config["IN_CHANS"], config["IMG_SIZE"], config["IMG_SIZE"]).to(device)
        onnx_path = os.path.join(config["RUN_DIR"], "model.onnx")
        try:
            # ONNXエクスポート時の警告を抑制
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=FutureWarning)
                torch.onnx.export(
                    model,
                    dummy_input,
                    onnx_path,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                    opset_version=21,  # ONNXバージョン
                    do_constant_folding=True,  # 定数演算最適化（Conv+BN統合など）
                    export_params=True,  # 重みを含める
                    dynamo=False,  # 旧エクスポータを使用（dynamic_axes警告回避）
                )
            #? ※ "Applied XXX pattern rewrite rules" はONNXグラフ最適化の正常メッセージ
            print(f"ONNX model saved at {onnx_path}")
        except Exception as e:
            print("#" * 50)
            print(f"Failed to save ONNX file: {e}")
            print("#" * 50)

    @staticmethod
    def check_cuda_compilation_available():
        """CUDA拡張のコンパイルが可能かチェック"""
        if not torch.cuda.is_available():
            return 0, "CUDA not available"
        
        from torch.utils.cpp_extension import CUDA_HOME
        if not CUDA_HOME:
            return 0, "CUDA_HOME not set"
        
        import shutil
        if not shutil.which('nvcc'):
            return 0, "nvcc compiler not found"
        
        print("CUDA ready")
        return 1, "CUDA compilation available"

class TrainUtils:
    """学習の際に使用するUtils"""
    @staticmethod
    def train_one_epoch(model, device, loader, mixup_fn, criterion, optimizer):
        """1エポック分の学習"""
        model.train()
        loss_sum, count = 0.0, 0
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None, style="grey30", complete_style="bold green", finished_style="bold green"),
            MofNCompleteColumn(), TimeRemainingColumn(), TransferSpeedColumn(),
            transient=True, expand=True, redirect_stdout=True, redirect_stderr=True,
        ) as progress:
            #!##### ---- ループ開始 ---- #####!#
            task = progress.add_task("Train", total=len(loader))
            for data, label in loader:
                # データとラベルをデバイスに転送
                data, label = data.to(device), label.to(device)
                # # Mixup/CutMixによるデータ拡張を適用（有効な場合のみ）
                # if mixup_fn is not None:
                #     data, label = mixup_fn(data, label)
                # 勾配をゼロクリア
                optimizer.zero_grad()
                # モデルで推論
                output = model(data)
                # 損失を計算
                loss = criterion(output, label)
                # 誤差逆伝播
                loss.backward()
                # パラメータを更新
                optimizer.step()
                # バッチ損失を累積（平均損失算出用）
                loss_sum += loss.item() * data.size(0)
                # 総データ数をカウント
                count += data.size(0)
                # プログレスバーを更新
                progress.update(task, advance=1)
            #!##### ---- ループ終了 ---- #####!#
        return loss_sum / count

    @staticmethod
    @torch.no_grad()
    def evaluate(model, device, loader, criterion, metric):
        """テストデータでの評価"""
        model.eval()
        loss_sum, count = 0.0, 0
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None, style="grey30", complete_style="bold green", finished_style="bold green"),
            MofNCompleteColumn(), TimeRemainingColumn(), TransferSpeedColumn(),
            transient=True, expand=True, redirect_stdout=True, redirect_stderr=True,
        ) as progress:
            #!##### ---- ループ開始 ---- #####!#
            task = progress.add_task("Test", total=len(loader))
            for data, target in loader:
                # データとラベルをデバイスに転送
                data, target = data.to(device), target.to(device)
                # モデルで推論
                output = model(data)
                # 損失を計算
                loss = criterion(output, target)
                # バッチ損失を累積（平均損失算出用）
                loss_sum += loss.item() * data.size(0)
                # 総データ数をカウント
                count += data.size(0)
                # 評価指標を更新（正答率計算用）
                metric.update(output, target)
                # プログレスバーを更新
                progress.update(task, advance=1)
            #!##### ---- ループ終了 ---- #####!#
        acc = metric.compute().item()
        metric.reset()
        return loss_sum / count, acc

    @staticmethod
    def setup_training(model, config, device):
        """学習に必要な各要素をまとめて初期化"""
        # # mixup (両方0の場合は無効化)
        # if config["MIXUP_ALPHA"] > 0 or config["CUTMIX_ALPHA"] > 0:
        #     mixup_fn = Mixup(mixup_alpha=config["MIXUP_ALPHA"], cutmix_alpha=config["CUTMIX_ALPHA"], prob=config["MIXUP_PROB"], switch_prob=config["MIXUP_SWITCH_PROB"], mode=config["MIXUP_MODE"], label_smoothing=config["LABEL_SMOOTHING"], num_classes=config["NUM_CLASSES"],)
        #     train_criterion = SoftTargetCrossEntropy()
        # else:
        #     mixup_fn = None
        #     train_criterion = nn.CrossEntropyLoss(label_smoothing=config["LABEL_SMOOTHING"])
        mixup_fn = None
        train_criterion = nn.CrossEntropyLoss(label_smoothing=config["LABEL_SMOOTHING"])
        # 損失関数
        test_criterion = nn.CrossEntropyLoss()
        # オプティマイザ
        if config["OPTIMIZER"].lower() == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=config["LEARNING_RATE"], momentum=config["MOMENTUM"], weight_decay=config["WEIGHT_DECAY"], nesterov=config["NESTEROV"])
        else:  # adamw
            optimizer = optim.AdamW(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"], betas=tuple(config["OPTIMIZER_BETAS"]))
        # スケジューラ
        scheduler = CosineLRScheduler(optimizer, t_initial=config["EPOCHS"], lr_min=config["LR_MIN"], warmup_t=config["WARMUP_EPOCHS"], warmup_lr_init=config["WARMUP_LR_INIT"], warmup_prefix=config["WARMUP_PREFIX"])
        # 評価指標
        metric = MulticlassAccuracy(num_classes=config["NUM_CLASSES"], device=device)
        # 学習エポック数
        epochs = config["EPOCHS"]
        # return
        return mixup_fn, train_criterion, test_criterion, optimizer, scheduler, metric, epochs

    @staticmethod
    def class_accuracy(model, device, loader, classes):
        """クラス別正答率を計算して表示"""
        model.eval()
        correct_pred = {c: 0 for c in classes}
        total_pred = {c: 0 for c in classes}
        with torch.no_grad():
            for data, labels in loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                for lbl, pred in zip(labels, predicted):
                    if lbl == pred:
                        correct_pred[classes[lbl]] += 1
                    total_pred[classes[lbl]] += 1
        for cls, cnt in correct_pred.items():
            print(f"  {cls:>6s}: {100*cnt/total_pred[cls]:.1f}%")

