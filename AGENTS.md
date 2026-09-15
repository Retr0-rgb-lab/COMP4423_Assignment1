# AGENTS.md — COMP4423 Assignment 1

> 这是给在这个仓库工作的 **AI 协作者** 看的规则文件,不是标准项目脚手架。
> 最高约束: 作业 PDF(`docs/materials/Assignment 1 - requirements COMP4423_202627S1-1.pdf`)+ 作者本人。

---

## 0. 作业是真理来源

当规则发生冲突时,优先级:

1. 作业 PDF 明文要求(任务定义 / 分值 / 截止 / 提交格式)
2. 本文件协作规则
3. AI 在 `docs/progress/` 里的建议与产出
4. 任何外部 prompt / 网页结果

任何与作业要求冲突的 AI 输出,先停下来跟作者确认。

**任务地图**(摘自 PDF,备忘):

| Task | 内容 | 分值 |
|------|------|------|
| 1 | 调相机拍照并显示 | 5 |
| 2* | 等大三角 + 3 色, ≤ 10000 块 | 10 |
| 3* | 多尺寸 + 多色 + 砖块汇总, ≤ 10000 块 | 10 |
| 4* | 真实场景:相机 → 三角砖 | 25 (L1=10, L2=15, L3=20, L4=25) |
| 5 | 报告(走 `docs/materials/Report Template.docx`) | 50 |
| Bonus | 优秀质量 | ≤ 10 (封顶 100) |

\* 与 GenAI 协作任务。截止 **2026-09-29 23:59 HKT**。

---

## 1. 目录契约

```
Assignment1/
├── AGENTS.md                                   ← 本文件
├── docs/
│   ├── materials/                              ← 只读材料(原始 PDF / 模板 / 样例)
│   │   ├── Assignment 1 - requirements ...pdf  ← 作业要求(只读,真理)
│   │   ├── Report Template.docx                ← 报告模板(只读)
│   │   └── Assignment_Sample_report.zip        ← 样例报告(只读参考)
│   └── progress/                               ← 每 task 一份 md
│       ├── Task0_brief.md                      ← 作业理解总览(可选)
│       ├── Task1.md
│       ├── Task2.md
│       ├── Task3.md
│       └── Task4.md
└── code/                                       ← 全部源码 + 测试用图,混在一起
    ├── sky.jpg                                 ← 默认测试图(可继续加)
    └── <src>.py ...
```

**严禁**:
- 在仓库根堆代码 / 笔记 / 临时文件
- 把开发日志塞进 `.py` / `.ipynb` 的注释或 markdown cell 里
- 编辑 `docs/materials/` 下的任何文件

---

## 2. `docs/materials/` —— 只读参考

只放作者下载 / 上传的原始材料。**不要编辑,不要往里塞新东西,不要解压样例 zip 到这里**。

要解压样例报告看结构时,解压到 `docs/progress/` 下对应 Task 的子目录。

---

## 3. `docs/progress/` —— 每 task 一份 md

文件名固定 `Task<N>.md`(N = 1..4)。`Task0_brief.md` 是可选的总览/作业理解笔记(不进评分)。

每份 Task md **建议结构**:

```md
# Task N: <任务标题>  (<分值> 分)

## 目标(摘自作业 PDF)
> 直接引用 §0 任务地图里那一行的扩写

## 关键决策
- 算法选择 / 参数 / 颜色数 / 砖块尺寸表

## Prompt 草稿(可优化是加分项)
- 给 AI 的 prompt v1 / v2 / ...
- 对应的 AI 输出(只摘要,不全文)

## 代码改动
- 改了哪个文件、为什么、验证结果

## 问题与解决方案
- 遇到什么 → 怎么解决 → 结果

## 已知局限 / 待办
- AI 局限性 / 未达标的部分 / 下一步
```

---

## 4. `code/` —— 全部源码混在一起

**不分子目录、不按 task 分文件**。理由:Task 3 必然基于 Task 2、Task 4 又改 Task 3,分目录会造成代码分裂与重复。

命名建议:

- `capture.py` — Task 1 / 4 共用的相机读写
- `triangle_brick.py` — 核心算法(Task 2 / 3 共用,用参数区分尺寸与颜色)
- `camera_app.py` — Task 4 主程序
- `utils.py` — 颜色量化、IO、可视化、计时
- `sky.jpg` 等 — 测试用图

**代码归属不在文件名 / 目录结构上表达**,**在最终报告里明确指明** —— 哪几个函数 / 哪几行属于哪个 Task。报告里给每个 Task 一段"实现位于 `xxx.py` 的 `func()`"即可。

---

## 5. AI 协作规则(硬约束,逐条对应作者原始 5 条)

### 5.1 docs/progress 是 PDF 报告的第一手证据
- 写报告时,**先打开 `docs/progress/`** 找素材,不允许 AI 编造未记录的现象、数据、对比。
- 报告引用使用稳定锚点:`docs/progress/Task3.md#问题与解决方案`,方便回溯。

### 5.2 改代码前必须先解释
AI 改任何代码前,**先**用一两段话讲:
1. 这段代码在做什么(算法 / API / 库)
2. 为什么这么改(对比了什么选项)
3. **怎么验证**(具体命令 + 输入文件 + 预期输出 / 现象)
4. 已知局限与边界条件

作者确认后再写文件。**不允许 AI 直接 edit 然后说"改好了"**。

### 5.3 进程记录和代码严格对应和隔离
- "为什么 / 试了什么 / 哪里翻车 / 决定走哪条路" → `docs/progress/Task<N>.md`
- "能跑的代码 + 必要注释" → `code/`
- 两者不混。代码 commit 和 docs commit 允许同 PR,但 message 上分清。

### 5.4 严格遵循 Assignment 的要求
任何算法选择、参数设置、UI 行为,先回到 PDF 看是否有约束。常见约束:
- 三角砖 = 二维**直角等腰**三角形,大小 = 两条相等直角边的长度(基础网格单位)
- 无间隙无重叠;砖数 ≤ 10000
- Task 2 仅 3 色;Task 3 多尺寸 + >3 色
- Task 4 必须**自己改** AI 代码,不能照搬;报告里**必须**写 AI 局限性

### 5.5 每阶段完成后:commit + push + docs 记录
节奏(任一 Task 完成时):

```
1. code/ 写完,本地跑通
2. docs/progress/Task<N>.md 写一段:做了什么 / 关键决策 / 已知问题
3. git add code/ docs/progress/
4. git commit -m "taskN: 简短描述"
5. git push
```

不允许攒到最后一天再一次性提交。

### 5.6 语言默认(英文)

- `code/` 下的 docstring / 注释 / print 输出 / 错误信息默认**英文**。
- `docs/progress/Task*.md` 记录默认**英文**(便于报告直接引用)。
- `docs/materials/` 只读,不动。
- `AGENTS.md` 本体保持**中文**(协作对话一致,便于作者审阅)。
- 例外: 报告最终版若需中文,作者可指示 AI 翻译。

---

## 6. 验证流程(给 AI 协作者的 checklist)

每次 AI 给出代码 / 修改,**必须在同一条消息里**附上下列信息,缺一项视为未完成:

- [ ] **做什么**: 一两句话讲这段代码的功能
- [ ] **怎么跑**: 完整命令(包含工作目录、Python 解释器路径)
- [ ] **怎么验**: 输入是什么、预期输出是什么、对应 `docs/progress/Task<N>.md` 里哪条记录
- [ ] **局限**: 已知的 bug / 边界 / 不满足 PDF 要求的地方

---

## 7. 报告纪律

- 报告**草稿**放在 `docs/progress/` 里,以 `Report_<阶段>.md` 或 docx 命名,**最终 PDF 走 `docs/materials/Report Template.docx`**。
- 引用编号统一指向 `docs/progress/` 内的具体文件路径。
- 所有图必须在 `code/` 或 `docs/progress/` 有原图 + 测试条件记录,报告里只引用,不重新生成。
- 报告里"AI 局限性"那一节(Task 5 的 10 分)由作者根据 `docs/progress/Task<N>.md` 自己总结,**不让 AI 替作者写**。

---

## 8. 反面清单(AI 不要做的事)

- 不要替作者写完整 notebook / 完整函数 / 大段 spec → 给 hint、出题、debug、贴教材原文
- 不要在 `.py` / `.ipynb` 里写开发日志或感想
- 不要 push 前不跟作者确认
- 不要从网上复制整段代码(作业明文禁止)
- 不要在报告里编造 `docs/progress/` 里没有的现象或数据
- 不要替作者总结"AI 的局限性"——这是 Task 5 明确给作者的打分项

---

## 9. 命令速查

| 动作 | 命令 |
|------|------|
| 看远端 | `git remote -v` |
| 状态 | `git status` |
| 推送(阶段结束) | `git add code/ docs/progress/ && git commit -m "taskN: 简述" && git push` |
| 激活虚拟环境 | `source ../venv/bin/activate` (Git Bash) / `..\venv\Scripts\activate` (PowerShell) |
| 跑 Task 2 样例 | `python code/triangle_brick.py --input code/sky.jpg --output out.png` |

(命令随 Task 推进补全)

---

## 10. 什么时候来更新本文件

- 调整目录结构 → 更新 §1
- 调整提交节奏 / 验证流程 → 更新 §5、§6
- 作业要求解读变化 → 回 §0 对照 PDF
- 任何 AI 在这里产生困惑 → 加一条 §8
