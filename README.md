# dol
DOL: Distributed Off-policy Learning

このリポジトリではDeepMind産の分散強化学習アルゴリズムである[Ape-X](https://openreview.net/pdf?id=H1Dy---0Z)を実装しています。

デモのような機能はつけていないのですがOpenAI Gym環境の離散行動空間の任意のタスクを学習することができます。

行動空間は内部の専用環境ラッパーにラップされて利用されます。

分散学習用のフレームワークとして[Ray](https://docs.ray.io/en/latest/index.html)を利用しています。

カリフォルニア大学産の近年注目を浴びているフレームワークですが原則Linuxでしか動作しません。
