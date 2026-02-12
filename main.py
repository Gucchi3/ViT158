import sys, os
import time

import torch

from rich.prompt import Prompt
import pretty_errors
pretty_errors.activate()

from utils import tools, TrainUtils
from model.ter_vit import TernaryVisionTransformer
from model.tiny_cnn import tiny_cnn
from model.vit import ViT
from model.test import TestViT




def main():
    # ── 初期セッティング ──────────────────────────────────────────────────────────────────────────────
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    device, config = tools.init_setting(config_path)
    # # cuda checker
    # if (config["USE_CUDA_KERNEL"]):
    #     tools.check_cuda_compilation_available()
    # モデル初期化 
    model_instance = TestViT(image_size=config["IMG_SIZE"], patch_size=config["PATCH_SIZE"], num_classes=config["NUM_CLASSES"], dim=config["EMBED_DIM"], depth=config["DEPTH"], heads=config["NUM_HEADS"], mlp_dim=int(config["EMBED_DIM"] * config["MLP_RATIO"]), channels=config["IN_CHANS"], dim_head=config["DIM_HEAD"], dropout=config["DROPOUT"], emb_dropout=config["EMB_DROPOUT"], pool=config["POOL"]).to(device)
    #model_instance = tiny_cnn().to(device)
    
    # 重み読み込み（config.jsonのLOAD_WEIGHTで制御）
    model_instance = tools.load_weight(model_instance, device, config)
    # DataLoader
    train_loader, test_loader, classes = tools.make_loader_cifar10(config)
    # パラメータ等初期化
    mixup_fn, train_criterion, test_criterion, optimizer, scheduler, metric, epochs = TrainUtils.setup_training(model_instance, config, device)
    # 損失等LOG用変数
    train_losses, test_losses, test_accs = [], [], []
    # 学習開始、設定表示、タイム計測開始
    tools.print_training_info(model_instance, device, config)
    start = time.time()
    
    
    # ── 学習ループ ─────────────────────────────────────────────────────────────────────────────────────
    for epoch in range(epochs):
        # スケジューラ前進
        scheduler.step(epoch)
        # 1epoch学習
        train_loss = TrainUtils.train_one_epoch(model_instance, device, train_loader, mixup_fn, train_criterion, optimizer)
        # test
        test_loss, test_acc = TrainUtils.evaluate(model_instance, device, test_loader, test_criterion, metric)
        # 損失等情報をLOG変数に格納
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # 1epoch終了ごとにepoch終了時の損失等出力
        print(f"[Epoch {epoch+1}/{epochs}] train_loss: {train_loss:.4f}  test_loss: {test_loss:.4f}  test_acc: {test_acc*100:.2f}%")
    
    
    # ── 学習時間、ベストaccEpoch、ベストAcc出力 ───────────────────────────────────────────────────────────
    elapsed = time.time() - start
    print(f"\nFinished Training — {elapsed:.1f}s")
    print(f"Best Epoch: {test_accs.index(max(test_accs))+1}  Best Acc: {max(test_accs)*100:.2f}%")
    
    
    # ── クラス別正答率 ──────────────────────────────────────────────────────────────────────────────────
    TrainUtils.class_accuracy(model_instance, device, test_loader, classes)
    
    
    # ── model保存 ─────────────────────────────────────────────────────────────────────────────────────
    tools.save_model(model_instance, config)
    
    
    # ── ONNX保存 ──────────────────────────────────────────────────────────────────────────────────────
    tools.save_onnx(model_instance, config, device)
    
    
    # ── グラフ保存 ─────────────────────────────────────────────────────────────────────────────────────
    tools.save_curves(train_losses, test_losses, test_accs, config)
    
    
    # ── 学習情報保存 ───────────────────────────────────────────────────────────────────────────────────
    tools.save_training_info(model_instance, device, config)
    
    



if __name__ == "__main__":
    main()
