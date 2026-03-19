# AIGaming
Train small models to play some classic games.

（本项目主要以俄罗斯方块为例）

以下是详细过程：

## 1.导入必要的库

我们需要导入必要的库，请在终端上运行以下命令：

```bash
pip install -r requirements.txt
```

> 为确保环境导入时正常，建议对于需要用到cuda等的程序包（如Torch）先进行手动下载

## 2.确认游戏环境

请运行```GameCheck.py```文件，用以更新`Envs.txt`文件

这样做是为了确认我们拥有哪些游戏环境，以防在编写代码的过程中出现名称问题或者是没有出现想要的环境。

> 由于在现阶段的`gymnasium`库中没有发现俄罗斯方块的相关环境，因此本项目另外还需要导入`ale-py`的库（已包含在`requirements.txt`的文件中），并使用`ALE/Tetris-v5`作为正式训练的环境。

## 3. 确认训练环境

请运行`GameEnv.py`文件的第一部分，用以确认torch是否能够应用GPU进行训练（如果实在缺少GPU环境，用CPU训练也可以，只是训练速度会变慢）。

```python
'''
这一段是用来是检查torch环境，包括是否支持cuda。
'''
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
```

请运行`GameEnv.py`文件的第二部分，用以确认游戏环境是否能够正常运行（主要是为了确认是否能够可视化，用以在项目的最后进行观测）。

```Python
'''
这一段是用来检查gym环境，特别是ALE环境是否正确注册和可用。
'''
gym.register_envs(ale_py)
# env=gym.make('ALE/Tetris-v5', render_mode='human')
env = gym.make('ALE/Tetris-v5', continuous=True, render_mode="human")
#continuous=True表示环境将持续运行，直到游戏结束，而不是在每一步后重置。

obs,info=env.reset()
episode_over=False

while not episode_over:
    '''
    action=env.action_space.sample()是随机采样一个动作，这个动作是从环境的动作空间中随机选择的。
    这个动作将被传递给env.step(action)函数，以便环境根据这个动作更新状态并返回新的观察、奖励、是否结束等信息。
    terminated和truncated是两个布尔值，分别表示游戏是否因为达到终止条件（如游戏结束）而结束，或者因为达到截断条件（如时间限制）而结束。
    episode_over是一个布尔值，当游戏结束或截断时为True，循环将停止。
    '''
    action=env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated

env.close()
```

