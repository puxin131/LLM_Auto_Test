# 详细设计文档（测试用例生成平台）

> 文档定位：面向研发与维护人员，描述当前实现的技术细节、数据结构、流程与扩展方案。  
> 适用版本：当前仓库（2026-04-09）  
> 关联文档：`DESIGN_REVIEW.md`（评审视角），`AGENTS.md`（协作与改造约束）

## 1. 设计目标与范围

### 1.1 目标
- 构建一个本地可运行的测试资产平台，支持多源知识入库、检索增强生成、审核入库闭环。
- 用最小改造实现多阶段能力增强：
  - 模块真源统一
  - 跨端双向链路
  - 场景分层生成
  - 前置人机协作约束
  - badcase 闭环与规则模板化

### 1.2 范围内
- Streamlit 页面（多页面导航与结果展示）
- 知识库资产入库、索引、向量检索
- 工作流编排（LangGraph + 线性回退）
- 规则引擎 + LLM 双通道约束校验
- 风险报告、badcase 事件、回放评测、模板自动调参

### 1.3 范围外
- 生产级分布式部署
- 多租户权限体系
- 高并发任务队列与异步调度
- 企业统一监控平台接入

---

## 2. 系统总览

## 2.1 目录与模块
- 页面层：
  - `app_streamlit.py`（唯一入口）
  - `apps/streamlit_views.py`（主页面逻辑）
- 平台编排层：
  - `apps/testcase_platform.py`
- 工作流层：
  - `workflow_graph.py`
- RAG 层：
  - `src/rag/generate_testcase.py`
  - `src/rag/kb_upsert.py`
  - `src/rag/analysis/*`
- 测试：
  - `tests/*`
- 数据：
  - `data/`（索引、队列、日志、评测样本等）

## 2.2 运行入口
- 命令：`streamlit run app_streamlit.py`
- 入口链路：
  - `app_streamlit.py` -> `apps.streamlit_views.main()`

---

## 3. 分层设计

## 3.1 页面层（`apps/streamlit_views.py`）
- 负责：
  - 页面导航与 UI 展示
  - 用户输入收集
  - 展示工作流结果（评审、约束、风险、链路、badcase）
- 不负责：
  - 核心业务规则计算
  - RAG 检索算法细节

## 3.2 编排层（`apps/testcase_platform.py`）
- 负责：
  - 资产构建、同步、日志
  - 审核队列持久化
  - 生成结果入库与幂等
  - 入库阻断策略（`risk_report` + 模板）
- 提供：
  - 供页面调用的统一工具函数

## 3.3 工作流层（`workflow_graph.py`）
- 负责：
  - 生成主流程状态机与回退
  - 推荐模式切换与锁定
  - 前置协作约束验证
  - 风险报告汇总
  - badcase 事件写入与回放

## 3.4 分析层（`src/rag/analysis`）
- 负责：
  - 证据锚点提取
  - 模块归因与影响分析
  - 双域契约、字段映射、覆盖矩阵
  - 双向链路提取
  - 约束 DSL 与合规评估
  - badcase 模板、回放、调参、回滚

---

## 4. 核心流程设计

## 4.1 知识入库流程
1. 页面收集多源输入（文件/文本/链接）
2. 平台层构建标准资产（`build_asset`）
3. `kb_upsert.ingest_assets`：
   - 解析文本
   - 切片
   - 写入向量库
   - 更新 `kb_index.json`
4. 返回入库摘要 + 记录操作日志

## 4.2 生成流程（工作流）
1. 读取检索上下文（支持 `approved_only/release/modules`）
2. 分类路由（`ui/api/fallback`）
3. 生成草稿（按 `generation_mode/recommended_mode`）
4. 约束校验（规则引擎 + 可选 LLM 审查）
5. 评审节点（评分、改写建议）
6. 失败重试或结束
7. 产出：
   - `final_testcases_md`
   - `review_result`
   - `risk_report`
   - `link_analysis`
   - `badcase_replay`

## 4.3 入库审核流程
1. 生成结果进入待审核队列或直接入库
2. 入库前执行 `P0` 阻断（模板可配置）
3. 幂等检测（按 content hash）
4. 成功后更新队列状态与日志

---

## 5. 数据结构设计

## 5.1 工作流输出（关键字段）
- `final_testcases_md`: 最终 markdown
- `review_result`: 评审结构（scores/comments/decision）
- `compliance_report`: 约束合规报告
- `risk_report`:
  - `overall_level/overall_score`
  - `severity_counts`
  - `items[]`（id/severity/category/title/reason/evidence/suggestions）
  - `history_adjustment`
- `link_edges/trace_refs/link_summary`: 双向链路结果
- `badcase_replay`:
  - `event_count/badcase_count/overall_bad_rate`
  - `alerts`
  - `rule_update_hints`
  - `rule_tuning`
  - `rule_template` 元信息

## 5.2 持久化文件
- `data/kb_index.json`: 资产索引
- `data/review_queue.json`: 审核队列
- `data/observation_log.jsonl`: 运行观测
- `data/badcase_events.jsonl`: badcase 样本事件
- `data/risk_rule_templates.json`: 风险规则模板（含历史版本）

---

## 6. 规则与闭环设计

## 6.1 约束合规（Phase C）
- 输入：`scope/exclusions/risk_focus/must_cover/...`
- 引擎：
  - 规则通道：term 匹配、阈值计算
  - LLM 通道：语义复核（可选）
- 输出：`pass/score/reasons/missing_items/rewrite_instructions`

## 6.2 风险报告（Phase B1/B2）
- 汇总信号：
  - 检索质量
  - 约束缺失
  - 缺口提示
  - 联动影响
  - 链路追踪
- 结果用于：
  - 页面解释
  - 入库阻断
  - badcase 事件标注

## 6.3 badcase 闭环（Phase D）
- 事件记录：每次工作流结束写入 badcase 事件
- 回放报告：按窗口统计 bad_rate 与高风险签名
- 模板自动调参：根据回放结果微调 `history_badcase` 规则
- 模板治理：
  - 历史记录
  - 最近变更查看
  - 一键回滚

---

## 7. 关键实现约束

## 7.1 兼容回退
- 若 LangGraph 不可用，自动进入线性工作流回退。
- 回退路径仍产出风险报告和 badcase 数据，保证数据一致性。

## 7.2 最小改造原则
- 页面只承载展示和输入收集。
- 业务规则收敛到平台/工作流/analysis 层。
- 风险策略模板化，避免散落硬编码。

## 7.3 幂等与安全
- 生成结果入库使用内容 hash 去重。
- `P0` 风险默认阻断入库，防止高风险内容沉淀。

---

## 8. 测试与验证

## 8.1 单测覆盖
- 工作流：`tests/test_workflow_graph.py`
- 入库与状态治理：`tests/test_kb_upsert.py`
- 约束合规：`tests/test_constraint_compliance.py`
- 链路提取：`tests/test_linkage_extractor.py`
- badcase 闭环：`tests/test_badcase_loop.py`

## 8.2 建议验证命令
```bash
python -m unittest tests.test_linkage_extractor \
  tests.test_constraint_compliance \
  tests.test_workflow_graph \
  tests.test_generate_testcase \
  tests.test_kb_upsert \
  tests.test_testcase_platform \
  tests.test_analysis_engines \
  tests.test_badcase_loop
```

---

## 9. 已知限制

- 模板自动调参为规则启发式，不是统计学习模型。
- badcase 回放依赖本地事件文件，文件被清理后历史权重会退化。
- 外部连接器（飞书/Figma）受网络与凭证限制，离线环境需降级策略。

---

## 10. 后续扩展建议

1. 模板治理增强：增加“人工审批后生效”开关，避免自动调参直接生效。  
2. 拆分大文件：将 `streamlit_views.py`/`testcase_platform.py` 按页面与服务拆分。  
3. 观测指标化：badcase 统计输出 Prometheus 兼容指标。  
4. 发布与基线：补 `README + requirements/pyproject + CI`。  

