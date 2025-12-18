# 🎯 THUYẾT TRÌNH: LEVEL 1 - OBJECT DETECTION PATH
## Kiến trúc Trích xuất Đặc trưng Không gian trong Q-Former Cải tiến

---

## 📋 NỘI DUNG THUYẾT TRÌNH

1. [Tổng quan kiến trúc 3 tầng](#1-tổng-quan-kiến-trúc-3-tầng)
2. [Vấn đề cần giải quyết](#2-vấn-đề-cần-giải-quyết)
3. [Kiến trúc Level 1 chi tiết](#3-kiến-trúc-level-1-chi-tiết)
4. [Cơ chế Attention-based Bounding Box](#4-cơ-chế-attention-based-bounding-box)
5. [Luồng xử lý dữ liệu](#5-luồng-xử-lý-dữ-liệu)
6. [Kết quả và Visualization](#6-kết-quả-và-visualization)

---

## 1. TỔNG QUAN KIẾN TRÚC 3 TẦNG

```
┌─────────────────────────────────────────────────────────────────┐
│                    Q-FORMER CẢI TIẾN                            │
│            (Hierarchical Multi-Path Reasoning)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   LEVEL 1   │───▶│   LEVEL 2   │───▶│   LEVEL 3   │         │
│  │   Object    │    │Scene Graph  │    │   Neural    │         │
│  │  Detection  │    │ Generation  │    │   State     │         │
│  │    Path     │    │   (GNN)     │    │  Machine    │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│        │                  │                  │                  │
│        ▼                  ▼                  ▼                  │
│   [Objects +        [Relations +        [Multi-hop             │
│    Bounding          Spatial            Reasoning              │
│    Boxes]            Graph]             Output]                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Vai trò của Level 1:
- **Nền tảng** cho toàn bộ pipeline xử lý
- Trích xuất **đặc trưng đối tượng** (object features)
- Xác định **vị trí không gian** (bounding boxes)
- Cung cấp input cho Level 2 (Scene Graph)

---

## 2. VẤN ĐỀ CẦN GIẢI QUYẾT

### 2.1 Hạn chế của Q-Former gốc (BLIP-2)

| Khía cạnh | Q-Former gốc | Vấn đề |
|-----------|--------------|--------|
| Spatial awareness | Implicit (ẩn) | Không biết object ở đâu |
| Object representation | Global queries | Không phân biệt objects |
| Position encoding | 1D từ ViT | Mất thông tin 2D |

### 2.2 Yêu cầu cho VQA nâng cao

```
❓ "Is the cat ON the table?"
     ↓
Cần biết: 
  • Vị trí của "cat" → Bounding Box 1
  • Vị trí của "table" → Bounding Box 2  
  • Quan hệ không gian giữa chúng
```

### 2.3 Mục tiêu thiết kế Level 1

✅ Trích xuất đặc trưng cho từng object riêng biệt  
✅ Xác định vị trí hình học (bounding box)  
✅ Không cần ground-truth boxes (unsupervised localization)  
✅ Có thể visualize để giải thích model  

---

## 3. KIẾN TRÚC LEVEL 1 CHI TIẾT

### 3.1 Các thành phần chính

```
┌──────────────────────────────────────────────────────────────────┐
│                    OBJECT DETECTION PATH                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────┐                                         │
│  │  2D Positional     │  ← Mã hóa vị trí (x,y) cho mỗi patch   │
│  │  Encoding          │                                         │
│  └─────────┬──────────┘                                         │
│            │                                                     │
│            ▼                                                     │
│  ┌────────────────────┐     ┌──────────────────┐                │
│  │   Image Features   │────▶│  Cross-Modal     │                │
│  │   (ViT patches)    │     │  Transformer     │                │
│  └────────────────────┘     └────────┬─────────┘                │
│                                      │                          │
│  ┌────────────────────┐              │                          │
│  │   Object Queries   │──────────────┘                          │
│  │   (Learnable)      │                                         │
│  └────────────────────┘                                         │
│            │                                                     │
│            ▼                                                     │
│  ┌────────────────────┐                                         │
│  │  Attention-based   │  ← Tính BB từ attention weights         │
│  │  Box Predictor     │                                         │
│  └─────────┬──────────┘                                         │
│            │                                                     │
│            ▼                                                     │
│  ┌────────────────────┐                                         │
│  │  Iterative Box     │  ← Tinh chỉnh qua nhiều layers          │
│  │  Refinement        │                                         │
│  └─────────┬──────────┘                                         │
│            │                                                     │
│            ▼                                                     │
│  ┌─────────────────────────────────────────┐                    │
│  │  Output: Object Features + Bounding Boxes                    │
│  │          + Confidence Scores + Attention Maps                │
│  └─────────────────────────────────────────┘                    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Chi tiết từng module

#### A. 2D Positional Encoding

```python
class PositionalEncoding2D:
    """
    Mã hóa vị trí 2D cho image patches.
    
    Ý tưởng: Mỗi patch trong ảnh có tọa độ (x, y) chuẩn hóa [0, 1]
    """
```

**Công thức:**
```
PE(x, y, 2i)   = sin(x · π / 10000^(2i/d))
PE(x, y, 2i+1) = cos(x · π / 10000^(2i/d))
PE(x, y, 2i+2) = sin(y · π / 10000^(2i/d))
PE(x, y, 2i+3) = cos(y · π / 10000^(2i/d))
```

**Minh họa grid 16×16:**
```
(0,0)  (1,0)  (2,0)  ... (15,0)
(0,1)  (1,1)  (2,1)  ... (15,1)
  :      :      :          :
(0,15) (1,15) (2,15) ... (15,15)

→ Normalize to [0, 1]:
(0.0, 0.0), (0.067, 0.0), ..., (1.0, 1.0)
```

#### B. Object Queries (Learnable)

```python
self.object_queries = nn.Parameter(torch.randn(1, 32, 768))
#                                        │   │    │
#                                        │   │    └─ Dimension
#                                        │   └────── 32 object slots
#                                        └────────── Batch dimension
```

**Vai trò:**
- Mỗi query học cách "tìm kiếm" một loại object
- Hoạt động như "anchor" để phát hiện objects
- Tương tự DETR object queries

#### C. Cross-Modal Transformer

```
Object Queries ──┐
                 ├──▶ Cross-Attention ──▶ Object Features
Image Patches ───┘         │
                          │
                    Attention Weights
                    (query attend vào patches nào)
```

**Công thức Cross-Attention:**
```
Attention(Q, K, V) = softmax(Q·K^T / √d) · V

Trong đó:
  Q = Object Queries      (32, 768)
  K = Image Patches       (256, 768)  
  V = Image Patches       (256, 768)
  
Output: 
  Object Features         (32, 768)
  Attention Weights       (32, 256) ← Quan trọng cho BB!
```

---

## 4. CƠ CHẾ ATTENTION-BASED BOUNDING BOX

### 4.1 Ý tưởng cốt lõi

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   "Bounding box được tính từ PHÂN PHỐI ATTENTION"          │
│                                                             │
│   Object Query attend vào patches nào → Vị trí object      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Công thức tính Bounding Box

**Bước 1: Tính tâm (Center) từ Weighted Mean**

```
          Σᵢ (attention_i × xᵢ)
cx = ─────────────────────────────
          Σᵢ attention_i

          Σᵢ (attention_i × yᵢ)  
cy = ─────────────────────────────
          Σᵢ attention_i
```

**Minh họa:**
```
Attention weights:        Patch coordinates:
┌─────────────────┐       ┌─────────────────┐
│ 0.1  0.1  0.1  │       │(0,0)(1,0)(2,0) │
│ 0.1 [0.3][0.2] │   ×   │(0,1)(1,1)(2,1) │
│ 0.05 0.1  0.05│       │(0,2)(1,2)(2,2) │
└─────────────────┘       └─────────────────┘

Center = weighted average ≈ (1.2, 0.8)
```

**Bước 2: Tính kích thước từ Variance**

```
Var(X) = E[X²] - E[X]²

width  = 2 × 2 × √Var(x)  ≈ 4 × std_x
height = 2 × 2 × √Var(y)  ≈ 4 × std_y
```

**Giải thích:**
- Variance đo độ "lan rộng" của attention
- Attention tập trung → Variance nhỏ → Object nhỏ
- Attention phân tán → Variance lớn → Object lớn

**Bước 3: Convert to [x, y, w, h]**

```python
# Top-left corner
x = cx - width / 2
y = cy - height / 2

# Final box (normalized [0, 1])
box = [x, y, width, height]
```

### 4.3 Box Refinement (Tinh chỉnh)

```python
# 2 layers refinement
for refine_layer in self.refine_layers:
    # Combine features với current box
    input = concat([object_features, current_box])
    
    # Predict small delta
    delta = refine_layer(input)  # → (Δx, Δy, Δw, Δh)
    
    # Apply với scaling nhỏ
    current_box = current_box + 0.1 × tanh(delta)
```

### 4.4 Confidence Score

```python
# Dựa trên Attention Entropy
entropy = -Σ(attention × log(attention))

# Entropy thấp = Attention tập trung = Confidence cao
# Entropy cao = Attention phân tán = Confidence thấp

confidence = MLP(features) - normalized_entropy
```

---

## 5. LUỒNG XỬ LÝ DỮ LIỆU

### 5.1 Input → Output Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT                                   │
├─────────────────────────────────────────────────────────────────┤
│  Image: (B, 3, 224, 224)                                       │
│  Question: "Is there a cat on the table?"                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VISION ENCODER (ViT)                         │
├─────────────────────────────────────────────────────────────────┤
│  Image → Patches (16×16 grid) → 256 patch embeddings           │
│  Output: (B, 257, 1024) ← Includes CLS token                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PROJECTION + 2D PE                           │
├─────────────────────────────────────────────────────────────────┤
│  1024 → 768 dimension                                          │
│  + Add 2D Positional Encoding                                  │
│  Output: (B, 256, 768) với position info                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 CROSS-MODAL TRANSFORMER                         │
├─────────────────────────────────────────────────────────────────┤
│  Object Queries: (B, 32, 768)                                  │
│  ↓                                                             │
│  Cross-Attention với Image Patches                             │
│  ↓                                                             │
│  Self-Attention với Text Embeddings                            │
│  ↓                                                             │
│  Feed-Forward Network                                          │
│  Output: Object Features (B, 32, 768)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              ATTENTION-BASED BOX PREDICTOR                      │
├─────────────────────────────────────────────────────────────────┤
│  Compute attention weights: (B, 32, 256)                       │
│  ↓                                                             │
│  Center = Σ(attn × coords)                                     │
│  Size = 4 × √Var(coords)                                       │
│  ↓                                                             │
│  Initial Box: (B, 32, 4)                                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ITERATIVE REFINEMENT                           │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: box = box + 0.1 × tanh(MLP(features, box))           │
│  Layer 2: box = box + 0.1 × tanh(MLP(features, box))           │
│  Clamp to [0, 1]                                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                  │
├─────────────────────────────────────────────────────────────────┤
│  • Object Features:     (B, 32, 768)                           │
│  • Spatial Info (boxes): (B, 32, 4)  ← [x, y, w, h]           │
│  • Confidence Scores:   (B, 32, 1)                             │
│  • Attention Maps:      (B, 32, 256) ← For visualization       │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Kích thước tensors qua từng bước

| Bước | Tensor | Shape | Ghi chú |
|------|--------|-------|---------|
| 1 | Input Image | (B, 3, 224, 224) | RGB image |
| 2 | ViT Output | (B, 257, 1024) | 256 patches + CLS |
| 3 | Projected Features | (B, 256, 768) | Remove CLS, project |
| 4 | + 2D Position | (B, 256, 768) | Add spatial info |
| 5 | Object Queries | (B, 32, 768) | Learnable |
| 6 | Cross-Attn Output | (B, 32, 768) | Object features |
| 7 | Attention Weights | (B, 32, 256) | For box prediction |
| 8 | Initial Boxes | (B, 32, 4) | From attention |
| 9 | Refined Boxes | (B, 32, 4) | Final output |

---

## 6. KẾT QUẢ VÀ VISUALIZATION

### 6.1 Ví dụ minh họa

```
Input Image:                    Detected Objects:
┌─────────────────────┐        ┌─────────────────────┐
│                     │        │  ┌───┐              │
│    🐱               │   →    │  │🐱│  Obj 0: 0.89 │
│         ┌─────┐     │        │  └───┘   ┌─────┐    │
│         │ 🍵  │     │        │          │ 🍵  │    │
│         └─────┘     │        │          └─────┘    │
│                     │        │          Obj 1: 0.76│
└─────────────────────┘        └─────────────────────┘
```

### 6.2 Attention Heatmap

```
Object Query 0 (Cat):           Object Query 1 (Table):
┌─────────────────────┐        ┌─────────────────────┐
│ ░░░░░░░░░░░░░░░░░░ │        │ ░░░░░░░░░░░░░░░░░░ │
│ ░░▓▓▓▓░░░░░░░░░░░░ │        │ ░░░░░░░░░░░░░░░░░░ │
│ ░░▓█████▓░░░░░░░░░ │        │ ░░░░░░░░▓▓▓▓▓░░░░░ │
│ ░░▓▓▓▓░░░░░░░░░░░░ │        │ ░░░░░░░░████████░░ │
│ ░░░░░░░░░░░░░░░░░░ │        │ ░░░░░░░░▓▓▓▓▓░░░░░ │
└─────────────────────┘        └─────────────────────┘
   Focused on cat                 Focused on table
```

### 6.3 Chạy Visualization

```bash
python scripts/visualize_bounding_boxes.py \
    --image_path images/cat_table.jpg \
    --question "Is the cat on the table?" \
    --output_dir visualizations \
    --top_k 5
```

**Output files:**
- `cat_table_bbox.png` - Ảnh với bounding boxes
- `cat_table_individual.png` - Từng object với attention map

---

## 7. SO SÁNH VỚI CÁC PHƯƠNG PHÁP KHÁC

| Phương pháp | Cần GT Boxes | Interpretable | VQA-friendly |
|-------------|--------------|---------------|--------------|
| Faster R-CNN | ✅ Yes | ❌ Black-box | ❌ Separate |
| YOLO | ✅ Yes | ❌ Black-box | ❌ Separate |
| DETR | ✅ Yes | ⚠️ Partial | ❌ Separate |
| **Ours (Level 1)** | ❌ No | ✅ Attention | ✅ Integrated |

### Ưu điểm của phương pháp đề xuất:

1. **Không cần supervision** cho bounding boxes
2. **Interpretable** - có thể visualize attention
3. **End-to-end** - tích hợp với VQA pipeline
4. **Lightweight** - không cần object detector riêng

---

## 8. TÓM TẮT

### 8.1 Đóng góp chính của Level 1

```
┌─────────────────────────────────────────────────────────────────┐
│                    ĐÓNG GÓP CHÍNH                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 2D Positional Encoding                                     │
│     → Mã hóa vị trí không gian rõ ràng cho patches             │
│                                                                 │
│  2. Attention-based Box Prediction                             │
│     → Tính bounding box từ attention distribution              │
│     → Không cần ground-truth supervision                       │
│                                                                 │
│  3. Iterative Refinement                                       │
│     → Tinh chỉnh boxes qua nhiều layers                        │
│                                                                 │
│  4. Visualization Capability                                    │
│     → Attention maps giải thích model decisions                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Slide kết thúc

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│           LEVEL 1: OBJECT DETECTION PATH                        │
│                                                                 │
│     "Trích xuất đặc trưng đối tượng với                        │
│      vị trí không gian từ attention patterns"                  │
│                                                                 │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐                       │
│  │ Object  │ → │Attention│ → │Bounding │                       │
│  │ Queries │   │ Weights │   │  Boxes  │                       │
│  └─────────┘   └─────────┘   └─────────┘                       │
│                                                                 │
│  → Feeds into Level 2: Scene Graph Generation                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📚 TÀI LIỆU THAM KHẢO

1. BLIP-2: Bootstrapping Language-Image Pre-training (2023)
2. DETR: End-to-End Object Detection with Transformers (2020)
3. Attention is All You Need (2017)
4. ViT: An Image is Worth 16x16 Words (2020)

---

## ❓ CÂU HỎI DỰ KIẾN TỪ HỘI ĐỒNG

**Q1: Tại sao không dùng object detector có sẵn như YOLO?**
> A: Phương pháp này tích hợp trực tiếp vào pipeline VQA, không cần pre-training riêng, và attention maps có thể giải thích được.

**Q2: Bounding boxes có chính xác không?**
> A: Đây là "soft localization" - xác định vùng quan tâm hơn là detection chính xác. Phù hợp cho VQA vì chỉ cần biết "ở đâu" để trả lời câu hỏi.

**Q3: Làm sao biết object queries học được gì?**
> A: Thông qua visualization attention maps - mỗi query sẽ có pattern attention khác nhau, tập trung vào các vùng khác nhau của ảnh.

---

*Document prepared for thesis defense presentation*
*Q-Former Improvement Project - 2024*
