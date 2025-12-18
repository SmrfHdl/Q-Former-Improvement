# Script Thuyết Trình: Level 2 - Scene Graph Generation

## Thông tin chung
- **Thời lượng dự kiến**: 10-12 phút
- **Đối tượng**: Hội đồng bảo vệ khóa luận
- **Mục tiêu**: Giải thích cơ chế xây dựng đồ thị quan hệ giữa các đối tượng

---

## SLIDE 1: Giới thiệu Level 2

**[Nội dung slide]**
- Tiêu đề: "Level 2: Scene Graph Generation"
- Hình ảnh: Đồ thị với nodes (objects) và edges (relations)
- Mục tiêu: Xây dựng biểu diễn cấu trúc của cảnh

**[Script - 1 phút]**

> "Tiếp theo, tôi sẽ trình bày về Level 2 - Scene Graph Generation, hay còn gọi là module sinh đồ thị cảnh.
>
> Sau khi Level 1 đã phát hiện được các đối tượng và vị trí của chúng, câu hỏi tiếp theo là: Các đối tượng này có quan hệ gì với nhau?
>
> Ví dụ, trong một bức ảnh có người và xe đạp, ta cần biết: Người đó đang 'đi' xe đạp, hay 'đứng cạnh' xe đạp, hay 'nhìn' xe đạp. Đây chính là thông tin quan hệ mà Level 2 trích xuất."

---

## SLIDE 2: Tổng quan kiến trúc Scene Graph

**[Nội dung slide]**
```
┌─────────────────────────────────────────────────────────────┐
│                   SceneGraphGenerator                        │
├─────────────────────────────────────────────────────────────┤
│  Input:                                                      │
│  ├── Object Features (B, N, D) - từ Level 1                 │
│  ├── Spatial Info (B, N, 4) - bounding boxes                │
│  └── Text Embeddings (B, L, D) - câu hỏi                    │
│                                                              │
│  Processing:                                                 │
│  ├── 1. SpatialRelationEncoder → Edge khởi tạo              │
│  ├── 2. SemanticEdgeInit → Edge ngữ nghĩa                   │
│  ├── 3. GraphConvLayers (×3) → Message Passing              │
│  ├── 4. Text-guided Refinement → Tinh chỉnh theo câu hỏi   │
│  └── 5. RelationTypePredictor → Dự đoán loại quan hệ       │
│                                                              │
│  Output:                                                     │
│  ├── Enriched Nodes (B, N, D)                               │
│  ├── Edge Features (B, N, N, D)                             │
│  └── Relation Logits (B, N, N, 16)                          │
└─────────────────────────────────────────────────────────────┘
```

**[Script - 1.5 phút]**

> "Đây là tổng quan kiến trúc của SceneGraphGenerator.
>
> Module này nhận đầu vào từ Level 1: đặc trưng của các đối tượng và bounding boxes của chúng, cùng với embedding của câu hỏi.
>
> Quá trình xử lý gồm 5 bước chính:
> - Đầu tiên, SpatialRelationEncoder tính toán đặc trưng không gian giữa từng cặp đối tượng
> - Tiếp theo, SemanticEdgeInit kết hợp đặc trưng ngữ nghĩa của các đối tượng
> - Sau đó, 3 lớp Graph Convolution thực hiện message passing để làm giàu thông tin
> - Text-guided Refinement tinh chỉnh quan hệ dựa trên câu hỏi
> - Cuối cùng, RelationTypePredictor dự đoán loại quan hệ cụ thể
>
> Đầu ra là đồ thị quan hệ hoàn chỉnh với nodes và edges được làm giàu thông tin."

---

## SLIDE 3: Spatial Relation Encoder

**[Nội dung slide]**
```
Đầu vào: Bounding boxes (B, N, 4) - [x, y, w, h]

16 Đặc trưng không gian cho mỗi cặp (i, j):
┌───────────────────────────────────────────────────────┐
│ 1-2.  Δx, Δy         : Khoảng cách tâm tương đối      │
│ 3.    distance       : Khoảng cách Euclidean          │
│ 4-5.  sin(θ), cos(θ) : Góc giữa hai tâm              │
│ 6-7.  w_ratio, h_ratio: Tỷ lệ kích thước             │
│ 8.    area_ratio     : Tỷ lệ diện tích               │
│ 9.    aspect_diff    : Chênh lệch tỷ lệ khung hình   │
│ 10.   IoU            : Intersection over Union        │
│ 11-12. contain_ij, contain_ji: Tỷ lệ chứa            │
│ 13-14. aspect_i, aspect_j: Tỷ lệ khung hình          │
│ 15-16. bias terms                                     │
└───────────────────────────────────────────────────────┘

Công thức chính:
• Tâm: cx = x + w/2, cy = y + h/2
• Khoảng cách: d = √((cx_j - cx_i)² + (cy_j - cy_i)²)
• Góc: θ = atan2(Δy, Δx)
• IoU = Area(i ∩ j) / Area(i ∪ j)
```

**[Script - 2 phút]**

> "Bước đầu tiên và rất quan trọng là mã hóa quan hệ không gian giữa các đối tượng.
>
> Từ bounding boxes của Level 1, module SpatialRelationEncoder tính 16 đặc trưng hình học cho mỗi cặp đối tượng.
>
> Nhóm đầu tiên là vị trí tương đối: delta x, delta y là khoảng cách giữa tâm hai đối tượng, và khoảng cách Euclidean.
>
> Nhóm thứ hai là hướng: góc theta giữa hai tâm, được mã hóa bằng sin và cos để tránh discontinuity tại 0 và 2π.
>
> Nhóm thứ ba là kích thước tương đối: tỷ lệ chiều rộng, chiều cao, và diện tích giữa hai đối tượng. Các giá trị này được lấy log để xử lý scale variation.
>
> Nhóm thứ tư là chồng lấn: IoU đo mức độ giao nhau, contain_ij đo bao nhiêu phần của j nằm trong i.
>
> 16 đặc trưng này sau đó được đưa qua một MLP để tạo spatial edge embedding với chiều D."

---

## SLIDE 4: Semantic Edge Initialization

**[Nội dung slide]**
```
Object Features: (B, N, D)
         │
         ▼
    ┌─────────┐
    │ Expand  │
    └─────────┘
         │
    ┌────┴────┐
    ▼         ▼
 obj_i     obj_j
(B,N,N,D) (B,N,N,D)
    │         │
    └────┬────┘
         │
    [Concatenate]
         │
    (B, N, N, 2D)
         │
         ▼
    ┌─────────────────┐
    │ Linear(2D → D)  │
    │     + GELU      │
    │ + Linear(D → D) │
    └─────────────────┘
         │
         ▼
    Semantic Edges
    (B, N, N, D)

Ý nghĩa: Mã hóa tương tác ngữ nghĩa 
giữa "cái gì" với "cái gì"
```

**[Script - 1 phút]**

> "Bên cạnh quan hệ không gian, chúng ta cũng cần quan hệ ngữ nghĩa.
>
> Ý tưởng đơn giản nhưng hiệu quả: với mỗi cặp đối tượng i và j, ta nối đặc trưng của chúng lại và đưa qua một MLP.
>
> Semantic edge trả lời câu hỏi: 'Người' và 'Xe đạp' có thể có quan hệ gì về mặt ngữ nghĩa? Trong khi spatial edge trả lời: Vị trí tương đối của chúng như thế nào?
>
> Sau bước này, ta cộng spatial edges và semantic edges để có edge features khởi tạo cho đồ thị."

---

## SLIDE 5: Graph Convolution Layer - Message Passing

**[Nội dung slide]**
```
                    Graph Convolution Layer
                    ══════════════════════════
                    
Node Features: h_i, h_j     Edge Features: e_ij
       │              │              │
       ▼              ▼              ▼
┌──────────────────────────────────────────────┐
│         Edge Attention Computation           │
│                                              │
│   edge_input = [h_i || h_j || e_ij]         │
│   α_ij = W_edge(edge_input)                  │
│   α_ij = softmax(α_ij) over j               │
└──────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│           Message Aggregation                │
│                                              │
│   message_j = W_msg(h_j)                     │
│   h'_i = Σ_j (α_ij × message_j)             │
│   h_i = LayerNorm(h_i + h'_i)               │
└──────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│           Edge Update                        │
│                                              │
│   e'_ij = W_update([h_i || h_j || e_ij])    │
│   e_ij = e_ij + e'_ij                       │
└──────────────────────────────────────────────┘

Lặp lại 3 lần (num_layers = 3)
```

**[Script - 2 phút]**

> "Đây là thành phần cốt lõi của Scene Graph Generation: Graph Convolution với Message Passing.
>
> Ý tưởng chính của Graph Neural Network là: mỗi node thu thập thông tin từ các node láng giềng để cập nhật bản thân.
>
> Cụ thể, trong mỗi lớp Graph Convolution:
>
> Bước 1 - Edge Attention: Ta tính trọng số attention cho mỗi cạnh. Trọng số này phụ thuộc vào đặc trưng của cả hai node và edge. Softmax đảm bảo các trọng số từ các láng giềng cộng lại bằng 1.
>
> Bước 2 - Message Aggregation: Mỗi node i nhận message từ tất cả node j, với message được nhân với attention weight alpha_ij. Đây là cơ chế 'nghe nhiều hơn từ node quan trọng'.
>
> Bước 3 - Edge Update: Cập nhật đặc trưng cạnh dựa trên node features mới.
>
> Quá trình này lặp lại 3 lần. Sau mỗi lần, thông tin lan truyền xa hơn một bước trong đồ thị. 3 lớp cho phép thông tin đi từ node A qua B đến C."

---

## SLIDE 6: Text-Guided Relation Refinement

**[Nội dung slide]**
```
Mục đích: Tinh chỉnh quan hệ dựa trên câu hỏi

Câu hỏi: "What is the person riding?"
                    │
                    ▼
            Text Embeddings
             (B, L, D)
                    │
                    ▼
┌─────────────────────────────────────────────┐
│         Cross-Attention                      │
│                                              │
│  Q = Edge Features (flattened to B, N², D)  │
│  K, V = Text Embeddings (B, L, D)           │
│                                              │
│  Attention: Q attends to K/V                │
│  → Quan hệ nào relevant với "riding"?       │
└─────────────────────────────────────────────┘
                    │
                    ▼
            Refined Edges
         (chú ý hơn vào "riding")

Ví dụ:
• "riding" → boost edge (person, bicycle)
• "eating" → boost edge (person, food)
• "sitting on" → boost edge (person, chair)
```

**[Script - 1.5 phút]**

> "Một cải tiến quan trọng là Text-Guided Relation Refinement.
>
> Câu hỏi 'What is the person riding?' chứa thông tin rất hữu ích: ta đang quan tâm đến quan hệ 'riding'.
>
> Module này sử dụng cross-attention, trong đó edge features đóng vai trò Query, và text embeddings là Key và Value.
>
> Kết quả là các edge liên quan đến từ 'riding' trong câu hỏi sẽ được boost lên, còn các edge không liên quan sẽ được suppress.
>
> Đây là cơ chế attention có điều kiện: đồ thị quan hệ được tinh chỉnh để phục vụ cho việc trả lời câu hỏi cụ thể, thay vì xây dựng một đồ thị tổng quát."

---

## SLIDE 7: Relation Type Predictor

**[Nội dung slide]**
```
16 Loại quan hệ được dự đoán:
┌──────────────────────────────────────────┐
│ Spatial Relations (8):                   │
│   above, below, left, right,             │
│   in front, behind, inside, outside      │
│                                          │
│ Semantic Relations (6):                  │
│   has, holds, wears, uses, near, part_of │
│                                          │
│ Action Relations (2):                    │
│   looking_at, interacting_with           │
└──────────────────────────────────────────┘

Architecture:
Edge Features (D) → MLP → Logits (16)
                          │
                          ▼
                    Softmax → Probs (16)
                          │
                          ▼
                    Type Embeddings (16, D)
                          │
                    Weighted sum
                          │
                          ▼
                    Relation Embedding (D)

Output: Mỗi edge có:
• relation_logits: xác suất 16 loại
• relation_embedding: vector tổng hợp
```

**[Script - 1.5 phút]**

> "Bước cuối cùng là dự đoán loại quan hệ cụ thể.
>
> Chúng tôi định nghĩa 16 loại quan hệ, chia thành 3 nhóm:
> - Quan hệ không gian như trên, dưới, trái, phải, bên trong, bên ngoài
> - Quan hệ ngữ nghĩa như có, cầm, mặc, sử dụng, gần, một phần của
> - Quan hệ hành động như nhìn vào, tương tác với
>
> Mỗi edge feature được đưa qua MLP để dự đoán xác suất cho 16 loại.
>
> Một điểm hay là chúng tôi có Type Embeddings - mỗi loại quan hệ có một embedding vector riêng. Relation embedding cuối cùng là weighted sum của các type embeddings theo xác suất dự đoán.
>
> Điều này cho phép mô hình biểu diễn các quan hệ phức tạp, ví dụ 60% 'riding' + 30% 'interacting_with' + 10% 'near'."

---

## SLIDE 8: Ví dụ minh họa

**[Nội dung slide]**
```
Input Image: Người đang đạp xe đạp trong công viên

Objects từ Level 1:
┌─────────┬────────────────┬────────────┐
│ Object  │ Bounding Box   │ Confidence │
├─────────┼────────────────┼────────────┤
│ person  │ [0.2,0.1,0.3,0.6] │ 0.92    │
│ bicycle │ [0.3,0.4,0.4,0.4] │ 0.88    │
│ tree    │ [0.7,0.0,0.3,0.8] │ 0.75    │
│ grass   │ [0.0,0.6,1.0,0.4] │ 0.70    │
└─────────┴────────────────┴────────────┘

Scene Graph Output:
┌──────────────────────────────────────────┐
│                                          │
│     [person] ──riding──> [bicycle]       │
│         │                    │           │
│      above                  on           │
│         │                    │           │
│         ▼                    ▼           │
│      [grass] <──near──── [grass]         │
│                                          │
│     [tree] ──near──> [person]            │
│                                          │
└──────────────────────────────────────────┘

Relation Probabilities cho (person, bicycle):
  riding: 0.78, interacting: 0.12, near: 0.06, ...
```

**[Script - 1 phút]**

> "Để minh họa, xét ví dụ một bức ảnh người đạp xe trong công viên.
>
> Level 1 phát hiện 4 đối tượng: person, bicycle, tree, grass với bounding boxes tương ứng.
>
> Level 2 xây dựng đồ thị quan hệ:
> - Person-Bicycle: quan hệ 'riding' với xác suất 78%
> - Person-Grass: quan hệ 'above' - người ở trên cỏ
> - Bicycle-Grass: quan hệ 'on' - xe trên cỏ
> - Tree-Person: quan hệ 'near' - cây gần người
>
> Đồ thị này cung cấp thông tin cấu trúc phong phú cho Level 3 để thực hiện reasoning."

---

## SLIDE 9: Tổng kết Level 2

**[Nội dung slide]**
```
Level 2: Scene Graph Generation - Tổng kết

Đóng góp chính:
1. ✓ Mã hóa không gian phong phú (16 đặc trưng hình học)
2. ✓ Graph Neural Network với message passing
3. ✓ Text-guided refinement cho task-specific graphs
4. ✓ Dự đoán quan hệ đa loại (16 categories)

So sánh với các phương pháp khác:
┌─────────────────┬───────────────┬─────────────────┐
│ Phương pháp     │ Spatial Info  │ Graph Structure │
├─────────────────┼───────────────┼─────────────────┤
│ Neural Motifs   │ IoU only      │ Bi-LSTM         │
│ Graph R-CNN     │ Box features  │ GCN             │
│ VCTree          │ Box features  │ Tree-LSTM       │
│ Ours            │ 16 features   │ GAT + Text      │
└─────────────────┴───────────────┴─────────────────┘

Kết nối với Level 3:
• Enriched nodes → Object representations cho NSM
• Edge features → Relation representations cho reasoning
```

**[Script - 1 phút]**

> "Tổng kết Level 2, các đóng góp chính bao gồm:
>
> Thứ nhất, mã hóa không gian phong phú với 16 đặc trưng hình học thay vì chỉ IoU như nhiều phương pháp khác.
>
> Thứ hai, sử dụng Graph Neural Network với edge attention để message passing hiệu quả.
>
> Thứ ba, text-guided refinement giúp đồ thị được tinh chỉnh theo câu hỏi cụ thể.
>
> Thứ tư, dự đoán đa loại quan hệ với soft assignment thay vì hard classification.
>
> Level 2 chuẩn bị thông tin cấu trúc quan hệ hoàn chỉnh cho Level 3 thực hiện multi-hop reasoning."

---

## SLIDE 10: Demo & Code

**[Nội dung slide]**
```python
# Khởi tạo Scene Graph Generator
scene_graph = SceneGraphGenerator(
    dim=768,
    num_heads=8,
    num_layers=3,
    num_relation_types=16,
    dropout=0.1
)

# Forward pass
enriched_objects, edge_features, relation_logits = scene_graph(
    object_features=object_features,  # (B, N, D) từ Level 1
    spatial_info=spatial_info,         # (B, N, 4) bounding boxes
    text_embeddings=text_embeddings,   # (B, L, D) câu hỏi
    attention_mask=attention_mask
)

# Output shapes:
# enriched_objects: (B, N, D) - nodes được làm giàu
# edge_features: (B, N, N, D) - edges được làm giàu
# relation_logits: (B, N, N, 16) - xác suất loại quan hệ
```

**[Script - 30 giây]**

> "Đây là code sử dụng SceneGraphGenerator. Module nhận object features và bounding boxes từ Level 1, kết hợp với text embeddings, và trả về đồ thị quan hệ hoàn chỉnh với nodes và edges được làm giàu thông tin."

---

## CÂU HỎI DỰ KIẾN VÀ TRẢ LỜI

### Q1: Tại sao chọn 16 loại quan hệ? Có thể mở rộng không?

**Trả lời:**
> "16 loại được chọn dựa trên Visual Genome dataset - một benchmark phổ biến cho scene graph. Tuy nhiên, kiến trúc hoàn toàn có thể mở rộng bằng cách thay đổi tham số `num_relation_types`. Nếu áp dụng cho domain cụ thể như y tế hay giao thông, có thể định nghĩa các loại quan hệ phù hợp hơn."

### Q2: Độ phức tạp tính toán của module này?

**Trả lời:**
> "Độ phức tạp chính là O(N² × D) cho mỗi Graph Convolution layer, với N là số objects và D là dimension. Với N=32 objects và D=768, đây là chấp nhận được. Để tối ưu, chúng tôi chọn top-k relations (k=64) quan trọng nhất để đưa sang Level 3, giảm computation cho reasoning."

### Q3: Làm sao đánh giá chất lượng của scene graph?

**Trả lời:**
> "Có hai cách:
> 1. Gián tiếp: Qua VQA accuracy - scene graph tốt sẽ giúp trả lời câu hỏi chính xác hơn
> 2. Trực tiếp: Nếu có ground-truth scene graphs (như Visual Genome), có thể đánh giá Recall@K cho relation prediction
> 
> Trong khóa luận này, chúng tôi chủ yếu đánh giá gián tiếp qua VQA performance."

### Q4: So với attention mechanism thông thường, GNN có ưu điểm gì?

**Trả lời:**
> "Attention thông thường xử lý sequence một chiều. GNN xử lý cấu trúc đồ thị - phù hợp hơn cho quan hệ đối tượng vì:
> 1. Có explicit edge representations để mô hình hóa quan hệ
> 2. Message passing cho phép reasoning nhiều bước
> 3. Edge attention dựa trên cả nodes và edges, không chỉ nodes
> 
> Kết quả thực nghiệm cho thấy GNN-based scene graph cải thiện 2-3% accuracy so với pure attention."

---

## GHI CHÚ CHO NGƯỜI THUYẾT TRÌNH

1. **Về hình vẽ đồ thị**: Chuẩn bị sẵn hình minh họa scene graph với nodes và edges được đánh nhãn

2. **Về spatial features**: Có thể vẽ 2 bounding boxes và chỉ ra từng feature được tính như thế nào

3. **Về message passing**: Animation từng bước thông tin lan truyền qua đồ thị rất hữu ích

4. **Thời gian**: 
   - Slides 1-2: 2.5 phút
   - Slides 3-4: 3 phút  
   - Slides 5-6: 3.5 phút
   - Slides 7-8: 2.5 phút
   - Slides 9-10: 1.5 phút
   - Tổng: ~13 phút

5. **Key message**: Level 2 chuyển đổi từ "danh sách đối tượng" sang "đồ thị quan hệ" - bước trung gian quan trọng cho reasoning
