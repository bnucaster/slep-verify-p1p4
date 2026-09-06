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

已用官方 tmlr.sty/tmlr.bst 正式编译通过（TeX Live 2025）。两版产物：

- 匿名投稿版：默认 `\usepackage{tmlr}`，作者显示 "Anonymous authors"（9 页）。
- arXiv 先发版：把包行切到 `\usepackage[preprint]{tmlr}`（同时置 preprint+
  accepted，显作者、无 OpenReview 行，10 页）。arXiv 源码包见
  `slep_arxiv_source.zip`（main.tex 已切 preprint + main.bbl + 三样式 +
  figures/*.pdf，自洽编译，arXiv 侧无需联网抓文献）。

注：本目录的 tmlr.sty/tmlr.bst/fancyhdr.sty 与生成的 PDF/zip 均在
gitignore（第三方样式不入库，构建产物可重生）；zip 已含样式供 arXiv 上传。

## 投稿前待办（不影响编译，影响内容合规）

- **母论文引文**：`references.bib` 的 `slep_parent` 为占位（作者/年份/
  出处未核实，标 TODO）。投稿前须核实填齐，禁止凭记忆虚构。
- ~~图内标签英文化~~：已完成。七图英文标签 + Times New Roman 正文 +
  STIX 数学体，白底、300 dpi、矢量 PDF（`docs/paper/figures/figN.pdf`，
  main.tex 引用 PDF）+ PNG 预览。重生成：
  `.venv/Scripts/python.exe scripts/make_paper_figures.py`。
- **作者块**：匿名版占位，署名版填 `\author{\name ... \email ...}`。
- **仓库 URL**：匿名版复现声明里已隐去，定稿版补回
  `https://github.com/bnucaster/slep-verify-p1p4`。
- **文献字段复核**：`references.bib` 中非核心条目（估计器方法引用）字段
  按记忆填写，投稿前用 DBLP/CrossRef 核对。
