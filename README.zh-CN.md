# backtest-receipt

`backtest-receipt` 是回测的研究纪律层，不是又一个统计函数库。带清洗（purge）和禁运（embargo）的交叉验证，以及组合清洗交叉验证（CPCV），交给 [purgedcv](https://github.com/eslazarev/purged-cross-validation)（MIT）。概率夏普比率（PSR）和紧缩夏普比率（DSR）在本仓库里根据 Bailey 与 López de Prado（2012、2014）自行实现。

这个库补的是那些包不做的流程：先把主张写下来再看结果，锁住样本外数据，把实际跑过的每个配置都计数，并生成一张 CI 可以拒绝的研究收据。

它不是回测引擎，不是数据客户端，也不计算回测过拟合概率（PBO）。它挡不住一个愿意亲手重写一张自洽收据的人。当 `check --study` 对着研究目录重算时，这种重写会对不上。

许可：MIT。Python 3.11 及以上。

## 安装

本包不发布到 PyPI。

```bash
pip install "git+https://github.com/tupils1/backtest-receipt.git"
```

本地开发：

```bash
pip install -e ".[dev]"
ruff check src tests examples
pytest
```

## 五分钟

```bash
python examples/synthetic_study.py --mode overfit --out /tmp/overfit-study
backtest-receipt check /tmp/overfit-study/receipt.json
echo $?   # 1：在噪声上网格搜索不会通过

python examples/synthetic_study.py --mode edge --out /tmp/edge-study
backtest-receipt check /tmp/edge-study/receipt.json
echo $?   # 0：事先只声明一个配置，序列本身带有漂移
```

两条命令都只用人工价格路径。示例里的均线交叉把信号滞后一根 bar。这是协议演示，不是某个策略的成绩。

## 协议

1. **预注册。** 假设、样本宇宙、参数网格、指标和通过标准写在一份 YAML 或一个 Python 对象里。内容哈希是规范 JSON 的 SHA-256，键的顺序不影响结果。叙述文字会被保存并计入哈希，但不进入夏普比率的计算。
2. **封存 holdout。** 时间落在声明边界上及其之后的行，加密为 Fernet 密文。密钥单独放在一个权限为 `0600` 的文件里，并被 gitignore。边界之前的 N 行是禁运行，不进入样本内，也不进入封存的 holdout。函数只把样本内的表交还你。
3. **记下每一个配置**，而且只用样本内的表。去重后的参数组合个数就是送进 DSR 的试验次数 `N`。同一组参数得到不同收益序列，记为冲突。网格之外的配置仍然会记下来，并让收据失败。
4. **只解锁一次**，并记录被选中配置的样本外收益。第二次解锁会被拒绝并写入日志。若传入 `acknowledge_repeat=True`，可以再次拿到数据，但这张收据会失败。
5. **开具收据**（`receipt.json` 和 `receipt.md`）。`backtest-receipt check` 只在收据内部自洽且结论为通过时返回 0。

```python
from backtest_receipt import PreRegistration, Study

spec = PreRegistration.from_mapping(
    {
        "study_id": "demo",
        "hypothesis": "滞后一根 bar 的均线符号，在这条序列上夏普比率为正。",
        "universe": "一条日频序列",
        "parameters": {"fast": [10, 20], "slow": [40, 80]},
        "metrics": ["sharpe", "mean_return"],
        "primary_metric": "sharpe",
        "periods_per_year": 252,
        "notes": "备注只作审计，不进入估计量。",
        "holdout": {"time_column": "date", "start": "2024-01-01", "embargo_periods": 5},
        "pass_criteria": {
            "min_deflated_sharpe": 0.95,
            "min_oos_sharpe_annualized": 0.0,
            "min_oos_observations": 30,
        },
    }
)
study = Study.create(spec, "studies/demo")
in_sample = study.seal_holdout(frame)
study.record_trial({"fast": 10, "slow": 40}, returns=run(in_sample, fast=10, slow=40))
study.evaluate_holdout(lambda holdout, params: run(holdout, **params))
receipt = study.issue_receipt()
```

`sharpe` 是年化夏普，只用来给试验排序，以及对照样本外的下限。PSR 和 DSR 用的是逐期夏普（均值除以样本标准差）。把年化数值喂进这两个概率会使结果失去意义。`periods_per_year` 不进入概率计算。

主指标越大越好。若你关心的是一个损失统计量，把它取负之后，用你已经声明过的名字传入。

### 通过标准

省略的数值门槛会用库的默认值，并写进规范化文档。哈希覆盖的是这份规范化文档。写成 `null` 会关掉对应的数值门槛。

| 字段 | 默认 | 含义 |
|---|---|---|
| `min_deflated_sharpe` | `0.95` | DSR 门槛。`null` 表示不使用。 |
| `min_probabilistic_sharpe` | 关闭 | 相对零基准的 PSR。 |
| `min_oos_sharpe_annualized` | `0.0` | 样本外年化夏普的下限。 |
| `max_sharpe_decay_annualized` | 关闭 | 样本内年化夏普减去样本外年化夏普的上限。 |
| `min_oos_observations` | `30` | 行数不足时状态是 `blocked`，不是统计上的失败。 |
| `require_single_holdout_unlock` | `true` | holdout 必须被解密一次并完成评分。 |
| `reject_post_result_amendments` | `true` | 封存、试验或解锁之后再改声明，研究失败。 |
| `reject_trials_outside_grid` | `true` | 未声明的配置使研究失败。 |
| `reject_conflicting_reruns` | `true` | 同一组参数得到了不同的收益序列。 |
| `require_return_series` | `true` | 夏普比率从存下来的收益序列重算。 |

`blocked` 表示流程还没做完，或者样本小到无法判断。`fail` 表示已经判断并拒绝，或者某条完整性规则被触发。两种情况 `check` 都返回 1。文件读不出来，或者和自己的门槛、指纹不一致（例如只改了 `passed`）时返回 2。

```bash
backtest-receipt hash prereg.yaml
backtest-receipt issue studies/demo
backtest-receipt check studies/demo/receipt.json
backtest-receipt check studies/demo/receipt.json --study studies/demo
```

单独的 `check` 确认收据内部自洽。`check --study` 会根据日志和封存文件重算结论。一张被整体改写、但自己仍然自洽的收据，必须对照研究目录才能发现。请把研究目录放进版本控制。哈希链能发现某一行被改过。它本身不能发现有人把日志末尾和所有副本一起删掉。

不要提交 `holdout.key`。封存文件里是密文和明文哈希。

### 样本内的交叉验证

```python
from backtest_receipt import make_combinatorial_purged_cv, make_purged_kfold

cv = make_purged_kfold(prediction_times, evaluation_times, n_splits=5, embargo="2D")
cpcv = make_combinatorial_purged_cv(
    prediction_times, evaluation_times, n_splits=6, n_test_groups=2
)
```

折不是试验。若你因为交叉验证分数留下了某个配置，需要自己把这个配置记下来。这些切分器不会打开 holdout。

Optuna 回调不需要本库去导入 Optuna：

```python
def objective(trial):
    lookback = trial.suggest_int("lookback", 5, 40)
    returns = run(in_sample, lookback)
    trial.set_user_attr("returns", returns)
    return float(sharpe(returns))

optuna_study.optimize(objective, callbacks=[study.optuna_callback()])
```

把收益序列放在 `user_attrs["returns"]`。只转发目标函数数值的回调仍会被记上一笔，收据随后会在「必须保存收益序列」这一关上失败。

## 估计量

逐期计算。夏普比率里的标准差是样本标准差。偏度和峰度是除以 `n` 的插入式中心矩。峰度不是超额峰度，正态分布的峰度是 3，不是 0。

```text
PSR(SR*) = Φ( (SR - SR*) * sqrt(n - 1) / sqrt(1 - skew * SR + ((kurtosis - 1) / 4) * SR^2) )
```

DSR 是把基准 `SR*` 换成 `N` 个独立试验的期望最大夏普之后的同一个概率：

```text
SR* = σ_trials * ( (1 - γ) Φ^{-1}(1 - 1/N) + γ Φ^{-1}(1 - 1/(N e)) )
```

`γ` 是欧拉-马斯刻若尼常数。`σ_trials` 是已记录的各个唯一配置的逐期夏普的样本标准差。只有一个试验时基准为 0，DSR 等于 PSR。`N` 是唯一参数组合的个数，并假定它们相互独立。搜索如果是成簇的，这样会计得偏严。本库不会悄悄把 `N` 换成有效试验次数。

文献：

- Bailey, D. H. and López de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier.* Journal of Risk, 15(2), 3–44.
- Bailey, D. H. and López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality.* The Journal of Portfolio Management, 40(5), 94–107.

## 持续集成

把 [`examples/github/workflows/oos-gate.yml`](examples/github/workflows/oos-gate.yml) 复制到提交收据的那个仓库。本库自己的 CI（[`.github/workflows/ci.yml`](.github/workflows/ci.yml)）只跑 ruff 和 pytest。它不会把包发到 PyPI，也不会打标签或做发布。

## 局限

- 威胁模型是诚实的研究者，或者一个本来会偷看并继续调参的代理。它不是用来对付一个掌握全部文件、并且愿意同时改写日志和一张自洽收据的人。
- 成功解锁之后、调用 `record_holdout_result` 之前如果进程崩溃，这一次干净的解密就用掉了。恢复时要 `acknowledge_repeat=True`，收据会失败。请用 `evaluate_holdout`，让解锁和记分发生在同一次调用里。
- 封存的表是密文内部的 JSON，面向研究规模的数据，不是行情仓库。
- 时间列必须是不带时区的。
- 默认门槛属于规范化文档。两个默认值不同的库版本，对一份不完整的 YAML 会算出不同的哈希。`Study.create` 会把解析后的 YAML 写下来，请提交那份文件。

English documentation: [README.md](README.md).
