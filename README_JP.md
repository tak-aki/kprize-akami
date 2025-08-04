# kprize-akami

Kaggle のコンペティション [Konwinski Prize](https://www.kaggle.com/competitions/konwinski-prize) におけるソリューションコードです。

## 構成

[smly さんの作業フロー](https://ho.lc/blog/kaggle_code_submission/)を参考に、コード・ライブラリ・submit notebook を管理するための構成となっています。

```
.
├── codes  # 実際に動かすコードを配置。Kaggle 上ではデータセットとして扱われる
├── deps   # サブミットに必要なライブラリをファイルとして作成。kernel として Kaggle にアップ・実行することで submit notebook からインポート可能
└── sub    # submission 用の notebook を配置
```

## 環境構築

### 依存関係のインストール

```bash
uv sync
```

### 学習済み difficulty モデルのダウンロード

```bash
cd input
kaggle datasets download kami634/kprize-akami-difficulty-model
unzip kprize-akami-difficulty-model.zip -d kprize-akami-difficulty-model
```

## 学習環境

学習には以下のマシン環境を使用：

- **GPU**: NVIDIA A100 80GB
- **クラウド**: Google Cloud
- **インスタンスタイプ**: a2-ultragpu-1g（12 vCPU、6 コア、170 GB メモリ）

## ローカルでの difficulty モデルの学習

```bash
uv run python -m local.train.exp004.run exp=70b_003
```

## ローカルでの実行

```bash
uv run python -m local.main
```

## submit 手順

1. 変更した codes をデータセットに反映するために、`./codes` ディレクトリにて以下コマンドを実行

   ```bash
   kaggle d version -m 'update' -r zip
   ```

2. submit に必要なライブラリに変更がある場合は、`./deps/kprize-akami-deps.ipynb` を編集し、`./deps` ディレクトリにて以下コマンドを実行

   ```bash
   kaggle k push
   ```

3. submit notebook をアップ・実行するために、`./sub` ディレクトリにて以下コマンドを実行
   ```bash
   kaggle k push
   ```

## 参考資料

- 最終提出コード：https://www.kaggle.com/code/kami634/kprize-akami-sub-kami?scriptVersionId=226698839
