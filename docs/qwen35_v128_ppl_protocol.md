# V128 + GDN Wo768 / full-attention Wo512 PPL

用户已要求追加此配置的 PPL。复用此前 evaluation.run_qwen35_hybrid evaluate 的 WikiText 和 C4 协议，数据为 results/q35_hybrid/data/windows.pt 中 wikitext、c4_eval，按 2048-token 窗口计算 next-token NLL，按预测 token 数加权后取 exp。保留逐窗口 mean NLL、总 NLL、预测 token 数及窗口数。PPL 为 teacher-forced 评分，不使用 GSM8K 对话生成或 thinking 模式。

使用扩展候选并以 KL 锚点 128 分配得到的 Two-sided 平均 V128 bank，以及对应冻结 V 重新校准的 GDN768/full512 Wo bank。

```bash
bash scripts/run_qwen35_v128_ppl.sh 2703776 \
  >> results/q35_hybrid/logs/v128_g768_f512_ppl.log 2>&1
```

此脚本先等待当前 V128 流水线进程结束，检查两个 bank 存在，然后使用 lowrank、本地 Qwen3.5 依赖 PYTHONPATH=results/q35_hybrid/deps:.、GPU 6、CPU 2 线程运行：

```bash
python -u -m evaluation.run_qwen35_hybrid evaluate \
  --bank results/q35_hybrid/banks_v128/c1_twosided_v128.pt \
  --wo-bank results/q35_hybrid/wo_v128_g768_f512/wo_bank.pt \
  --output results/q35_hybrid/ppl/twosided128_g768_f512.json
```

不改动正在运行的 GSM8K 流水线脚本，不覆盖旧结果。若上游未能产出两个 bank，等待脚本退出并记录原因。当前尚无此配置的 PPL 数值；启动后先处于等待状态。未提交或上传 GitHub。
