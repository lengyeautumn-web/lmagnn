# lmagnn
 🔍 AdvancedLogicOperators: Logical Message Propagation with Feature Fusion

## 📌 Overview
**AdvancedLogicOperators** is a PyTorch-based deep learning framework designed for **logical reasoning over structured and temporal data**, particularly tailored for **knowledge graph reasoning tasks**.

This repository implements a **logic-driven neural architecture** that integrates:
- temporal logical operators  
- attention-based feature fusion  
- multi-hop message propagation  

to enhance reasoning capability under **sparsity, incompleteness, and complex logical dependencies**, which are common challenges in real-world knowledge graphs.

---

## 🚀 Key Contributions

- 🔗 **Logical Message Propagation Mechanism**  
  Introduces a novel reasoning paradigm that embeds logical operators into neural message passing, enabling structured inference over paths.

- 🧠 **Key Feature Fusion Strategy**  
  Combines multi-source features (structural, temporal, and logical signals) to improve reasoning robustness and accuracy.

- ⏱ **Temporal Logical Modeling**  
  Supports temporal reasoning via advanced operators such as *until*, *eventually*, and *globally*.

- 🎯 **Improved Reasoning under Sparse Data**  
  Effectively handles incomplete knowledge graphs by leveraging logical constraints and attention mechanisms.

---

## 🧩 Model Architecture

The model consists of the following main components:

### 1. Logical Operators Module
Implements a variety of differentiable logical operators:

- **Temporal operators**:  
  `always`, `eventually`, `until`, `next`, `previous`, `release`  

- **Modal operators**:  
  `globally`, `strongly_eventually`, `weakly_always`  

- **Boolean operators**:  
  `and`, `or`, `not`, `implies`, `unless`  

These operators enable **logic-aware feature transformation** during reasoning.

---

### 2. Attention-based Feature Fusion
- Multi-head attention is used to:
  - capture dependencies across entities and relations  
  - dynamically weight important reasoning paths  

- Enhances **key feature aggregation** during propagation

---

### 3. Logical Message Propagation
- Extends traditional GNN-based propagation by:
  - injecting logical constraints into message passing  
  - modeling **multi-hop relational paths with logic guidance**

---

### 4. Regularization & Optimization
- Dropout + BatchNorm for stable training  
