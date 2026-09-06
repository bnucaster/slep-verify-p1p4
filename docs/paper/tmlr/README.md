# TMLR 投稿稿（LaTeX）

英文正文 `main.tex` + 文献 `references.bib`，图引用 `../figures/figN.png`。

## 构建前置

需官方 TMLR 体例文件（本仓库未含，从公开仓库取）：

```
# https://github.com/JmlrOrg/tmlr-style-file
# 取 tmlr.sty 与 tmlr.bst 放到本目录（docs/paper/tmlr/）
```

模式开关在 `main.tex` 顶部：默认匿名投稿（`\usepackage{tmlr}`），
`[preprint]` 为署名预印，`[accepted]` 为定稿。

## 构建

```
cd docs/paper/tmlr
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## 投稿前待办（不影响编译，影响内容合规）

- **母论文引文**：`references.bib` 的 `slep_parent` 为占位（作者/年份/
  出处未核实，标 TODO）。投稿前须核实填齐，禁止凭记忆虚构。
- **图内标签英文化**：现有七图 `docs/paper/figures/` 为中文标签（草稿
  阶段产物）。英文投稿须把图内文字改英文——在 `scripts/make_paper_figures.py`
  加英文标签模式重生成，图注已是英文。
- **作者块**：匿名版占位，署名版填 `\author{\name ... \email ...}`。
- **仓库 URL**：匿名版复现声明里已隐去，定稿版补回
  `https://github.com/bnucaster/slep-verify-p1p4`。
- **文献字段复核**：`references.bib` 中非核心条目（估计器方法引用）字段
  按记忆填写，投稿前用 DBLP/CrossRef 核对。
