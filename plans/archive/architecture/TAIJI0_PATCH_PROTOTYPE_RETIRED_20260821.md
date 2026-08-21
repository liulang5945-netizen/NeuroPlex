# Taiji-0 补丁原型废止记录

2026-08-21，用户明确判断 `neuroplex.taiji` 仍像 Transformer/NeuroPlex 的补丁，并要求 Taiji 成为具有完整算法和代码设计的独立底座。

被废止原型包含：三个 `TaijiCell`、全局 priority/top-k、同维向量 field、每活动 cell 的精确 K/V fast memory，以及最终接回 NeuroPlex 的迁移设计。它通过了状态合同、一次关联和 20 关联留存，但 T5 显示两个固定赢家保存完全相同的 20 份记忆，第三个 cell 不参与。

旧实现从当前工作树删除，不再接受 T5-bis 活动稳态修补。可恢复提交：

- `52fcb5c`：Taiji-0 动力学原型；
- `9671ab7`：精确局部 K/V；
- `57e3fba`：T5 留存与固定赢家证据。

替代实现是仓库顶层 `taiji/` 的 Taiji Predictive Fabric。该实现拥有自己的 raw-byte 输入、递归预测状态、局部学习、运动输出、自由生成和 checkpoint，不调用 Legacy forward。
