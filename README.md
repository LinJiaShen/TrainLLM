# LLM Practice 🚀
學習與實作大型語言模型（LLM）的實踐專案，從零開始訓練、微調（Fine-tuning）、搭配 RAG（Retrieval-Augmented Generation），並探討如何設計 Prompt 以減少 Prompt Injection Attack。

---

## 📌 目標
這個專案的目標是：
- **理解 LLM 的運作原理**
- **從頭訓練一個 LLM**
- **學習如何 Fine-tune 現有的模型**
- **實作 Retrieval-Augmented Generation（RAG）**
- **研究 Prompt Engineering，並減少 Prompt Injection 攻擊風險**

---

## 📚 內容規劃
1. **LLM 基礎介紹**
   - Transformer 架構
   - Self-Attention 機制
   - 主要 LLM 模型（GPT, BERT, T5）
  
2. **從頭訓練 LLM**
   - 使用 Hugging Face 及 PyTorch 訓練 LLM
   - 構建數據集並進行預處理
   - 設計與訓練自己的 Transformer 模型

3. **Fine-tuning**
   - 微調現有的 LLM，如 LLaMA、GPT-3.5、Mistral
   - 使用 LoRA、Adapter 等技術提升效率
   - 在特定領域應用（如醫療、金融等）

4. **RAG（檢索增強生成）**
   - 結合 LLM 與向量資料庫（如 FAISS、Chroma）
   - 建立一個基於 RAG 的問答系統
   - 應對知識更新、提升回應準確性

5. **Prompt Engineering 與安全性**
   - 如何設計 Prompt 以提升 LLM 的回應質量
   - Prompt Injection Attack 原理與防禦策略
   - 限制 LLM 輸出有害內容的技術（如 RLHF）

---

## 🔧 環境安裝
你可以使用 Python 環境來執行本專案，推薦使用 `venv` 或 `conda` 建立虛擬環境。

### 1️⃣ 創建 Python 虛擬環境
```sh
python -m venv llm_env
source llm_env/bin/activate  # Mac/Linux
llm_env\Scripts\activate  # Windows
```
### 2️⃣ 安裝必要套件
```sh
pip install torch transformers datasets accelerate langchain faiss-cpu
```
### 3️⃣ 檢查安裝
```sh
python -c "import torch; import transformers; print('環境準備完成 ✅')"
```
---

## 🏗️ 專案結構
```bash
LLM_Practice/
│── README.md            # 本文件
│── requirements.txt     # 依賴套件清單
│── src/                 # 主要程式碼
│   ├── train_llm.py     # 訓練 LLM 的腳本
│   ├── finetune.py      # Fine-tuning 腳本
│   ├── rag_pipeline.py  # RAG 應用
│   └── prompt_security.py  # Prompt Injection 測試
│── data/                # 數據集存放
│── notebooks/           # Jupyter Notebook
│── models/              # 訓練好的模型
└── logs/                # 訓練記錄
```
---
## 📌 計畫進度
✅ 計畫整理 
🟡 撰寫 LLM 介紹 🔜 實作 LLM 訓練
