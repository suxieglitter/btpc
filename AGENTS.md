# btpc

Barlow Twins 自监督 P 波初动极性分类（BTPC）研究代码仓库。当前主线工作：按 SRL 审稿意见修改并重投，工作票与纸面产物在旁边的 `../SRL_revision` 仓库（`.scratch/srl-resubmission/issues/`）；本仓库承载实验代码改动。

## Agent skills

### Issue tracker

Issues 以本地 markdown 文件形式放在 `.scratch/<feature>/issues/` 下，随仓库提交。见 `docs/agents/issue-tracker.md`。

### Triage labels

沿用五个默认分诊标签（needs-triage / needs-info / ready-for-agent / ready-for-human / wontfix），写在每张票的 `Status:` 行。见 `docs/agents/triage-labels.md`。

### Domain docs

单上下文：根目录一个 `CONTEXT.md` + `docs/adr/`。见 `docs/agents/domain.md`。
