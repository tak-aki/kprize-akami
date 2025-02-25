# kprize-akami

## 構成
[smlyさんの作業フロー](https://ho.lc/blog/kaggle_code_submission/)を参考に、コード・ライブラリ・submit notebookを管理するための構成
1回smlyさんの記事を読んで理解する必要あり

.  
├── codes # 実際に動かすコードを配置。kaggle上ではデータセットとして扱われる。  
├── deps # サブミットに必要なライブラリをファイルとして作成する。kernelとしてkaggleにアップ・実行することでsubmit notebookからインポート可能になる。  
└── sub # submission用のnotebookをおく。


## submit手順
1. 変更したcodesをデータセットに反映するために、./codesにて以下コマンドを実行  
`codes % kaggle d version -m 'update' -r zip`
2. submitに必要なライブラリに変更がある場合は、./deps/kprize-akami-deps.ipyngを編集し、./depsにて以下コマンドを実行  
`deps % kaggle k push`
2. submit notebookをアップ・実行するために、./subにて以下コマンドを実行  
`sub % kaggle k push`