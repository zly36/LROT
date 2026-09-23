# data/ — LR 数据库来源文件

| 文件 | 说明 |
|---|---|
| `CellChatDB.mouse.rda` | CellChatDB 小鼠版原始文件（CellChat 官方发布），`tools/build_cellchat_db.py` 的输入 |
| `CellChatDB_mouse_lr.pkl` | 由上面 rda 解析得到的配体–受体对（本文 LR 数据库） |
| `go/` | **未随包**：GO 本体库 `go-basic.obo`、`mgi.gaf.gz`，请从 Gene Ontology 官方下载后放入 |

`go/` 下载地址（Gene Ontology 官方发布）：
* `https://purl.obolibrary.org/obo/go/go-basic.obo`
* `https://current.geneontology.org/annotations/mgi.gaf.gz`

用途：`experiments/go_enrichment.py`（图S3）。
