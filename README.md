# dol
DOL: Distributed Off-policy Learning

このリポジトリではDeepMind産の分散強化学習アルゴリズムである[Ape-X](https://openreview.net/pdf?id=H1Dy---0Z)を実装しています。

デモのような機能はつけていないのですがOpenAI Gym環境の離散行動空間の任意のタスクを学習することができます。

行動空間は内部の専用環境ラッパーにラップされて利用されます。

分散学習用のフレームワークとして[Ray](https://docs.ray.io/en/latest/index.html)を利用しています。

カリフォルニア大学産の近年注目を浴びているフレームワークですが原則Linuxでしか動作しないのでこちらのフレームワークもそのような仕様になっています。

# how to use
ニューラルネットワークおよび任意のGym環境のインスタンスを作成するbuilderクラスを作成しインスタンスを作成します。

**Tensorflowのモデルは基本的にプロセスセーフではありません**ので__call__()メソッドが呼ばれると適切なネットワークが作成されるクラスのインスタンスを生成します。

```
class NNBuilder:
  def __call__(self):
    return tf.keras.models.Sequential([tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(2, activation='relu')])
```

同様にgym.spaces.Spaceのインスタンスを返すメソッドを実装したクラスのインスタンスを作成します。

```
class EnvBuilder:
  def __call__(self):
    return gym.Make('CartPole-v0')
```

最後にこれらを用いて実際に学習をするためApeXBuilderクラスのインスタンスを作成しstart()メソッドを呼び出します。


```
from dol.agent.apex.apex import ApeXBuilder
apex = ApeXBuilder(EnvBuilder(), NNBuilder())
apex.start()
```
