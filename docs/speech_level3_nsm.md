# Script Thuyết Trình: Level 3 - Neural State Machine

## Thông tin chung
- **Thời lượng dự kiến**: 12-15 phút
- **Đối tượng**: Hội đồng bảo vệ khóa luận
- **Mục tiêu**: Giải thích cơ chế multi-hop reasoning với Neural State Machine

---

## SLIDE 1: Giới thiệu Level 3

**[Nội dung slide]**
- Tiêu đề: "Level 3: Neural State Machine - Multi-hop Reasoning"
- Hình ảnh: Biểu đồ các bước reasoning lặp lại
- Mục tiêu: Trả lời câu hỏi phức tạp qua nhiều bước suy luận

**[Script - 1 phút]**

> "Cuối cùng, tôi sẽ trình bày Level 3 - Neural State Machine, module thực hiện multi-hop reasoning.
>
> Sau khi Level 1 phát hiện đối tượng và Level 2 xây dựng đồ thị quan hệ, câu hỏi còn lại là: Làm sao tổng hợp thông tin để trả lời câu hỏi?
>
> Với câu hỏi đơn giản như 'What color is the car?', một bước attention là đủ. Nhưng với câu hỏi như 'What is the person next to the red car holding?', ta cần:
> 1. Tìm red car
> 2. Tìm person next to red car  
> 3. Xác định person đó holding cái gì
>
> Đây chính là multi-hop reasoning mà Level 3 giải quyết."

---

## SLIDE 2: Tổng quan kiến trúc Neural State Machine

**[Nội dung slide]**
```
┌─────────────────────────────────────────────────────────────────┐
│                    Neural State Machine                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│   │  Hop 1  │───>│  Hop 2  │───>│  Hop 3  │───>│  Hop 4  │     │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘     │
│        │              │              │              │           │
│        ▼              ▼              ▼              ▼           │
│   ┌─────────────────────────────────────────────────────┐      │
│   │           Mỗi hop gồm 3 units:                      │      │
│   │   1. Control Unit: Quyết định tìm gì tiếp theo     │      │
│   │   2. Read Unit: Trích xuất thông tin từ đồ thị     │      │
│   │   3. Write Unit: Cập nhật bộ nhớ                   │      │
│   └─────────────────────────────────────────────────────┘      │
│                                                                  │
│   State được duy trì:                                           │
│   • Control State c_t: "Đang tìm gì?"                          │
│   • Memory State m_t: "Đã biết gì?"                            │
│                                                                  │
│   Đầu ra: final_output = MLP([c_T || m_T])                     │
└─────────────────────────────────────────────────────────────────┘
```

**[Script - 1.5 phút]**

> "Neural State Machine lấy cảm hứng từ MAC Network, được thiết kế cho compositional reasoning.
>
> Kiến trúc gồm 4 reasoning hops, mỗi hop là một bước suy luận. Mỗi hop có 3 đơn vị:
> - Control Unit quyết định ta đang tìm kiếm thông tin gì
> - Read Unit trích xuất thông tin từ knowledge base (objects và relations)
> - Write Unit cập nhật memory với thông tin vừa đọc được
>
> Hai state được duy trì xuyên suốt:
> - Control state c_t mã hóa 'đang tìm gì' ở bước t
> - Memory state m_t mã hóa 'đã thu thập được gì'
>
> Sau 4 hops, control và memory được kết hợp để tạo output cuối cùng cho các task heads."

---

## SLIDE 3: Control Unit - Quyết định tìm gì

**[Nội dung slide]**
```
                    Control Unit
                    ════════════════
                    
Input:
• prev_control c_{t-1}: Control state trước (B, D)
• question_embeddings Q: Embedding câu hỏi (B, L, D)

Processing:
┌────────────────────────────────────────────────────┐
│                                                    │
│   Query = c_{t-1}                                  │
│   Key = Value = Q                                  │
│                                                    │
│   attended_q = Attention(Query, Key, Value)        │
│                                                    │
│   c_t = MLP([c_{t-1} || attended_q])              │
│                                                    │
└────────────────────────────────────────────────────┘

Output: c_t - Control state mới (B, D)

Ý tưởng: Dựa trên context hiện tại, attend vào câu hỏi
để xác định từ/phrase nào cần focus ở bước tiếp theo.

Ví dụ câu hỏi: "What is the person next to red car holding?"
• Hop 1: Focus "red car" → tìm red car
• Hop 2: Focus "person next to" → tìm người cạnh
• Hop 3: Focus "holding" → tìm vật đang cầm
```

**[Script - 2 phút]**

> "Control Unit hoạt động như một 'program counter' - quyết định bước tiếp theo của chương trình reasoning.
>
> Đầu vào là control state trước đó và embedding của câu hỏi. Output là control state mới.
>
> Cơ chế: Control state trước làm Query, câu hỏi làm Key và Value. Attention sẽ chọn phần nào của câu hỏi cần focus ở bước này.
>
> Ví dụ với câu hỏi 'What is the person next to red car holding?':
> - Ở hop 1, control có thể focus vào 'red car' - ta cần tìm đối tượng xe màu đỏ trước
> - Ở hop 2, sau khi đã tìm thấy red car, control shift sang 'person next to' - tìm người gần xe
> - Ở hop 3, control focus vào 'holding' - xác định người đó đang cầm gì
>
> Đây là cách model 'đọc' câu hỏi theo từng bước, thay vì xử lý toàn bộ một lần."

---

## SLIDE 4: Read Unit - Trích xuất thông tin

**[Nội dung slide]**
```
                    Read Unit
                    ════════════════
                    
Input:
• control c_t: "Đang tìm gì?" (B, D)
• object_features O: Objects từ Level 1 (B, N, D)
• relation_features R: Relations từ Level 2 (B, K, D)
• memory m_{t-1}: "Đã biết gì?" (B, D)

Processing:
┌────────────────────────────────────────────────────────┐
│                                                        │
│   // Attend vào objects                               │
│   obj_info = CrossAttention(c_t → O)                  │
│                                                        │
│   // Attend vào relations                             │
│   rel_info = CrossAttention(c_t → R)                  │
│                                                        │
│   // Kết hợp object và relation info                  │
│   combined = MLP([obj_info || rel_info])              │
│                                                        │
│   // Gate với memory (chọn lọc thông tin mới)         │
│   gate = σ(W[combined || memory])                     │
│   read_out = gate × combined + (1-gate) × memory      │
│                                                        │
└────────────────────────────────────────────────────────┘

Output: 
• read_out (B, D): Thông tin trích xuất được
• obj_attn (B, 1, N): Attention weights trên objects
• rel_attn (B, 1, K): Attention weights trên relations
```

**[Script - 2 phút]**

> "Read Unit thực hiện việc 'đọc' thông tin từ knowledge base dựa trên control signal.
>
> Knowledge base ở đây gồm 2 phần:
> - Object features từ Level 1: đại diện cho các đối tượng
> - Relation features từ Level 2: đại diện cho quan hệ giữa các đối tượng
>
> Read Unit sử dụng control state làm query để attention vào cả objects và relations. Kết quả được kết hợp qua MLP.
>
> Một điểm quan trọng là gating mechanism với memory. Không phải tất cả thông tin mới đều hữu ích. Gate quyết định bao nhiêu phần trăm thông tin mới được giữ lại, bao nhiêu phần trăm memory cũ được giữ nguyên.
>
> Ví dụ: Nếu hop 1 đã tìm được 'red car' và lưu vào memory, hop 2 khi tìm 'person next to', gate sẽ giữ lại thông tin về car trong memory đồng thời thêm thông tin về person."

---

## SLIDE 5: Write Unit - Cập nhật Memory

**[Nội dung slide]**
```
                    Write Unit
                    ════════════════
                    
Input:
• memory m_{t-1}: Memory state trước (B, D)
• read_output r_t: Thông tin vừa đọc (B, D)
• control c_t: Control signal hiện tại (B, D)

Processing:
┌────────────────────────────────────────────────────────┐
│                                                        │
│   // Tính gate: Cập nhật bao nhiêu?                   │
│   gate = σ(W_gate([m_{t-1} || r_t || c_t]))           │
│                                                        │
│   // Tính update candidate                            │
│   update = MLP([r_t || c_t])                          │
│                                                        │
│   // Gated update                                     │
│   m_t = gate × update + (1 - gate) × m_{t-1}         │
│                                                        │
│   m_t = LayerNorm(m_t)                               │
│                                                        │
└────────────────────────────────────────────────────────┘

Output: m_t - Memory state mới (B, D)

Ý tưởng: Memory như "working memory" của con người
• Lưu giữ thông tin relevant
• Quên thông tin không cần thiết
• Cập nhật với thông tin mới
```

**[Script - 1.5 phút]**

> "Write Unit cập nhật memory với thông tin vừa đọc được.
>
> Đây là gated update tương tự như LSTM/GRU. Gate quyết định memory cũ được giữ lại bao nhiêu và thông tin mới được ghi vào bao nhiêu.
>
> Gate phụ thuộc vào 3 yếu tố:
> - Memory hiện tại: Đã có thông tin gì rồi?
> - Read output: Thông tin mới là gì?
> - Control: Thông tin này có relevant với mục tiêu hiện tại không?
>
> Memory ở đây hoạt động như working memory của con người:
> - Giữ lại thông tin cần thiết cho reasoning
> - Loại bỏ thông tin không liên quan
> - Tích lũy kiến thức qua các bước
>
> Sau 4 hops, memory chứa tất cả thông tin đã thu thập, sẵn sàng cho việc trả lời câu hỏi."

---

## SLIDE 6: Reasoning Cell - Một bước hoàn chỉnh

**[Nội dung slide]**
```
                    Reasoning Cell (1 Hop)
                    ══════════════════════════
                    
    c_{t-1}     m_{t-1}      Q         O         R
       │           │         │         │         │
       ▼           │         ▼         │         │
  ┌─────────────┐  │    Question       │         │
  │   CONTROL   │──┼───Embeddings      │         │
  │    UNIT     │  │         │         │         │
  └──────┬──────┘  │         │         │         │
         │         │         │         │         │
         ▼         │         │         ▼         ▼
        c_t        │         │      Objects  Relations
         │         │         │         │         │
         ▼         ▼         │         ▼         ▼
  ┌─────────────────────────────────────────────────┐
  │                    READ UNIT                     │
  │   CrossAttn(c_t → O) + CrossAttn(c_t → R)       │
  │   + Gate with m_{t-1}                           │
  └───────────────────────┬─────────────────────────┘
                          │
                          ▼
                      read_out
                          │
         ┌────────────────┼────────────────┐
         │                │                │
         ▼                ▼                ▼
        c_t           read_out          m_{t-1}
         │                │                │
         ▼                ▼                ▼
  ┌─────────────────────────────────────────────────┐
  │                   WRITE UNIT                     │
  │   gate = σ(W[m || r || c])                      │
  │   m_t = gate × MLP([r || c]) + (1-gate) × m     │
  └───────────────────────┬─────────────────────────┘
                          │
                          ▼
                        m_t
                          
Output: c_t, m_t, attention_weights
```

**[Script - 1.5 phút]**

> "Đây là flow hoàn chỉnh của một reasoning hop.
>
> Bắt đầu từ control và memory state trước đó:
> 1. Control Unit attend vào câu hỏi để cập nhật control - xác định cần tìm gì
> 2. Read Unit dùng control mới để attend vào objects và relations - trích xuất thông tin
> 3. Write Unit kết hợp thông tin đọc được với memory cũ - tích lũy kiến thức
>
> Sau mỗi hop, ta có control và memory mới, sẵn sàng cho hop tiếp theo.
>
> Attention weights được lưu lại để visualization - ta có thể thấy model đang 'nhìn' vào đối tượng nào ở mỗi bước."

---

## SLIDE 7: Multi-hop Reasoning Flow

**[Nội dung slide]**
```
Câu hỏi: "What is the person next to the red car holding?"

┌─────────────────────────────────────────────────────────────────┐
│ HOP 1: Tìm "red car"                                            │
├─────────────────────────────────────────────────────────────────┤
│ Control: Focus "red car" trong câu hỏi                         │
│ Read: Attend mạnh vào object có attribute "red" + "car"        │
│       → obj_attn = [0.1, 0.1, 0.7, 0.05, 0.05] (car có idx=2)  │
│ Write: Memory cập nhật với embedding của car                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ HOP 2: Tìm "person next to"                                     │
├─────────────────────────────────────────────────────────────────┤
│ Control: Shift focus sang "person next to"                     │
│ Read: Attend vào relations (car, ?) với type "near"            │
│       + Attend vào objects có type "person"                    │
│       → rel_attn peaks at (car, person) edge                   │
│ Write: Memory += person info, giữ car info                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ HOP 3: Tìm "holding what"                                       │
├─────────────────────────────────────────────────────────────────┤
│ Control: Focus "holding"                                        │
│ Read: Từ person trong memory, attend vào relations             │
│       với type "holding"                                        │
│       → rel_attn peaks at (person, bag) edge                   │
│ Write: Memory += bag info (đây là answer!)                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ HOP 4: Consolidate                                              │
├─────────────────────────────────────────────────────────────────┤
│ Control: Focus tổng thể câu hỏi "What ... holding?"            │
│ Read: Verify bag is correct answer                             │
│ Write: Finalize memory                                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    Answer: "bag" ✓
```

**[Script - 2 phút]**

> "Hãy trace qua một ví dụ cụ thể để thấy multi-hop reasoning hoạt động như thế nào.
>
> Câu hỏi: 'What is the person next to the red car holding?'
>
> Hop 1: Control focus vào 'red car'. Read Unit attend mạnh vào đối tượng car trong danh sách objects. Memory lưu embedding của car.
>
> Hop 2: Control shift sang 'person next to'. Lúc này Read Unit sử dụng relation features - tìm edge có type 'near' bắt đầu từ car. Attention cao nhất ở edge (car, person). Memory cập nhật thêm person info.
>
> Hop 3: Control focus vào 'holding'. Read Unit tìm relation từ person với type 'holding'. Kết quả là edge (person, bag). Memory ghi nhận bag là answer candidate.
>
> Hop 4: Consolidation - verify và finalize.
>
> Final output được tính từ control và memory cuối cùng, chứa đủ thông tin để trả lời 'bag'."

---

## SLIDE 8: Hierarchical Reasoning Path

**[Nội dung slide]**
```
            HierarchicalReasoningPath (Level 3 complete)
            ═══════════════════════════════════════════════
            
Input từ Level 1 & 2:
┌──────────────────────┬──────────────────────┬────────────────┐
│ object_features      │ relation_features    │ image_features │
│ (B, N, D)           │ (B, K, D)            │ (B, P, D)      │
│ Enriched nodes      │ Top-k edges          │ Raw patches    │
└──────────────────────┴──────────────────────┴────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Neural State Machine                       │
│                      (4 hops)                                │
│                          │                                   │
│                          ▼                                   │
│                    nsm_output (B, D)                        │
└─────────────────────────────────────────────────────────────┘
              │
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│               Global Transformer                             │
│   global_queries ──CrossAttn──> hierarchical_features       │
│                          │                                   │
│                          ▼                                   │
│                  global_pooled (B, D)                       │
└─────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Feature Fusion                            │
│        fused = MLP([nsm_output || global_pooled])           │
└─────────────────────────────────────────────────────────────┘
              │
    ┌─────────┼─────────┬─────────────┐
    │         │         │             │
    ▼         ▼         ▼             ▼
┌───────┐ ┌───────┐ ┌───────┐   ┌───────────┐
│  ITM  │ │  LM   │ │Answer │   │  Global   │
│ Head  │ │ Head  │ │ Head  │   │ Features  │
└───────┘ └───────┘ └───────┘   └───────────┘
    │         │         │             │
    ▼         ▼         ▼             ▼
  2-way    Vocab     Binary        (B,Q,D)
  logits   logits    logits       for ITC
```

**[Script - 2 phút]**

> "HierarchicalReasoningPath là module hoàn chỉnh của Level 3, bao gồm NSM và các task heads.
>
> NSM nhận object features và relation features từ Level 1 và 2. Output là vector nsm_output chứa kết quả reasoning.
>
> Song song đó, Global Transformer với learnable queries attend vào hierarchical features để capture global context - những patterns có thể bị miss bởi step-by-step reasoning.
>
> Hai outputs được fuse lại:
> - NSM output: detailed reasoning
> - Global features: holistic understanding
>
> Fused features được đưa vào 3 task heads:
> - ITM Head: Binary classification cho Image-Text Matching
> - LM Head: Language modeling cho text generation
> - Answer Head: Binary yes/no cho VQA
>
> Global features còn được dùng cho ITC loss - contrastive learning giữa image và text."

---

## SLIDE 9: Task Heads và Losses

**[Nội dung slide]**
```
╔═══════════════════════════════════════════════════════════════╗
║                      Task Heads                                ║
╠═══════════════════════════════════════════════════════════════╣
║                                                                ║
║  1. ITM Head (Image-Text Matching)                            ║
║     ─────────────────────────────────                         ║
║     Input: fused (B, D)                                       ║
║     Output: logits (B, 2) → matched/not-matched              ║
║     Loss: CrossEntropy với hard negatives                     ║
║     Ý nghĩa: Image-text có match không?                       ║
║                                                                ║
║  2. LM Head (Language Modeling)                               ║
║     ─────────────────────────────────                         ║
║     Input: updated_text (B, L, D)                             ║
║     Output: logits (B, L, Vocab)                              ║
║     Loss: CrossEntropy với teacher forcing                    ║
║     Ý nghĩa: Generate text mô tả image                        ║
║                                                                ║
║  3. Answer Head (VQA)                                         ║
║     ─────────────────────────────────                         ║
║     Input: fused (B, D)                                       ║
║     Output: logits (B, 1) → yes/no probability               ║
║     Loss: BCE với label smoothing                             ║
║     Ý nghĩa: Trả lời yes/no questions                         ║
║                                                                ║
╠═══════════════════════════════════════════════════════════════╣
║                   Auxiliary Losses                             ║
╠═══════════════════════════════════════════════════════════════╣
║  • loss_itc: Contrastive (image ↔ text alignment)            ║
║  • loss_object: BCE trên object confidence (Level 1)         ║
║  • loss_relation: KL-div trên relation types (Level 2)       ║
╚═══════════════════════════════════════════════════════════════╝

Total Loss = 2.0×answer + 1.0×itc + 1.0×itm + 0.1×relation + 0.05×object
```

**[Script - 1.5 phút]**

> "Hệ thống có 3 task heads chính và các auxiliary losses.
>
> ITM Head phân loại image-text pair có match hay không. Training với hard negatives - những cặp tương tự nhưng không match, buộc model phân biệt tinh vi hơn.
>
> LM Head cho image-grounded text generation. Dùng teacher forcing khi training.
>
> Answer Head cho binary VQA với label smoothing để tránh overconfident predictions.
>
> Auxiliary losses từ Level 1 và 2:
> - Object loss khuyến khích phát hiện đối tượng
> - Relation loss regularize relation predictions với uniform prior
>
> Weights được tuned: answer loss quan trọng nhất (2.0), ITC và ITM cân bằng (1.0), auxiliary losses thấp hơn."

---

## SLIDE 10: Attention Visualization

**[Nội dung slide]**
```
Visualization của NSM qua các hops:

Câu hỏi: "Is the dog on the couch?"

                    Hop 1              Hop 2              Hop 3              Hop 4
                ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
                │          │      │          │      │          │      │          │
   Objects:     │ 🐕 0.8   │      │ 🐕 0.3   │      │ 🐕 0.7   │      │ 🐕 0.4   │
                │ 🛋️ 0.1   │      │ 🛋️ 0.6   │      │ 🛋️ 0.2   │      │ 🛋️ 0.3   │
                │ 🪴 0.1   │      │ 🪴 0.1   │      │ 🪴 0.1   │      │ 🪴 0.3   │
                │          │      │          │      │          │      │          │
                └──────────┘      └──────────┘      └──────────┘      └──────────┘
                     ↓                 ↓                 ↓                 ↓
   Control      "dog"             "on the"          "couch"           [verify]
   Focus:       
                     ↓                 ↓                 ↓                 ↓
   Memory:      dog_embed         dog + spatial     dog + couch       final state
                                  relation          + "on" relation

   
Relations:     
  (dog,couch):  0.1              0.7               0.8               0.5
  (dog,plant):  0.1              0.1               0.1               0.2
  (couch,plant):0.1              0.2               0.1               0.3

→ High attention on (dog, couch) with "on" relation → Answer: YES
```

**[Script - 1.5 phút]**

> "Một ưu điểm của NSM là interpretability - ta có thể visualize attention qua các hops.
>
> Với câu hỏi 'Is the dog on the couch?':
>
> Hop 1: Object attention cao nhất ở dog (0.8). Control focus vào từ 'dog'. Memory bắt đầu ghi nhận dog.
>
> Hop 2: Attention shift sang couch (0.6). Control focus 'on the'. Quan trọng hơn, relation attention giữa (dog, couch) tăng lên 0.7.
>
> Hop 3: Attention cân bằng hơn giữa dog và couch. Relation (dog, couch) attention đạt 0.8. Memory có đủ thông tin về spatial relationship.
>
> Hop 4: Consolidation - verify thông tin đã thu thập.
>
> Kết quả: High attention trên edge (dog, couch) với relation type 'on' → Answer YES.
>
> Visualization này giúp debug và explain predictions của model."

---

## SLIDE 11: Tổng kết Level 3

**[Nội dung slide]**
```
Level 3: Neural State Machine - Tổng kết

Đóng góp chính:
1. ✓ Multi-hop reasoning với 4 bước
2. ✓ Control-Read-Write architecture
3. ✓ Kết hợp object và relation reasoning
4. ✓ Interpretable attention patterns

So sánh với các phương pháp khác:
┌──────────────┬─────────────┬──────────────┬────────────────┐
│ Phương pháp  │ Multi-hop   │ Relations    │ Interpretable  │
├──────────────┼─────────────┼──────────────┼────────────────┤
│ ViLBERT      │ ✗           │ Implicit     │ Partial        │
│ LXMERT       │ ✗           │ Implicit     │ Partial        │
│ MAC Network  │ ✓ (12 hops) │ ✗            │ ✓              │
│ NSM (ours)   │ ✓ (4 hops)  │ Explicit     │ ✓              │
└──────────────┴─────────────┴──────────────┴────────────────┘

Key insight:
• Ít hops hơn MAC (4 vs 12) nhưng hiệu quả hơn
• Vì đã có structured relations từ Level 2
• Relations explicit → reasoning dễ hơn
```

**[Script - 1 phút]**

> "Tổng kết Level 3, các đóng góp chính:
>
> Thứ nhất, multi-hop reasoning cho phép xử lý câu hỏi phức tạp yêu cầu nhiều bước suy luận.
>
> Thứ hai, kiến trúc Control-Read-Write rõ ràng và modular.
>
> Thứ ba, tận dụng cả object và relation information từ các levels trước.
>
> Thứ tư, attention patterns có thể visualize để interpret predictions.
>
> So với MAC Network nguyên bản cần 12 hops, NSM của chúng tôi chỉ cần 4 hops. Lý do: Level 2 đã cung cấp explicit relations, giảm workload cho reasoning. Model không cần 'học' quan hệ implicitly nữa."

---

## SLIDE 12: Tích hợp 3 Levels

**[Nội dung slide]**
```
           TỔNG QUAN KIẾN TRÚC 3 CẤP
           ═══════════════════════════
           
    Image                          Question
      │                                │
      ▼                                ▼
┌──────────────┐              ┌──────────────┐
│ Vision       │              │ Text         │
│ Encoder      │              │ Encoder      │
│ (CLIP ViT)   │              │ (CLIP/BERT)  │
└──────┬───────┘              └──────┬───────┘
       │                             │
       ▼                             │
╔══════════════════════════════════════════════════════╗
║  LEVEL 1: Object Detection                           ║
║  • 2D Positional Encoding                            ║
║  • Cross-attention: queries → image                 ║
║  • Attention-based box prediction                    ║
║                                                      ║
║  Output: objects (B,N,D), boxes (B,N,4)             ║
╚══════════════════════════════════════════════════════╝
                        │
                        ▼
╔══════════════════════════════════════════════════════╗
║  LEVEL 2: Scene Graph Generation                     ║
║  • Spatial Relation Encoder                          ║
║  • Graph Convolution (3 layers)                      ║
║  • Text-guided refinement                            ║
║                                                      ║
║  Output: nodes (B,N,D), edges (B,N,N,D)             ║
╚══════════════════════════════════════════════════════╝
                        │
                        ▼
╔══════════════════════════════════════════════════════╗
║  LEVEL 3: Neural State Machine                       ║
║  • Multi-hop reasoning (4 hops)                      ║
║  • Control-Read-Write per hop                        ║
║  • Task heads (ITM, LM, Answer)                      ║
║                                                      ║
║  Output: predictions for VQA, ITM, generation       ║
╚══════════════════════════════════════════════════════╝
                        │
                        ▼
                    Answer
```

**[Script - 1 phút]**

> "Cuối cùng, đây là tổng quan kiến trúc 3 cấp hoàn chỉnh.
>
> Level 1 chuyển đổi image thành structured objects với bounding boxes.
>
> Level 2 xây dựng scene graph với nodes và edges được làm giàu thông tin.
>
> Level 3 thực hiện multi-hop reasoning trên đồ thị để trả lời câu hỏi.
>
> Mỗi level có nhiệm vụ riêng biệt, output của level dưới là input của level trên. Kiến trúc này cho phép reasoning có cấu trúc và interpretable."

---

## CÂU HỎI DỰ KIẾN VÀ TRẢ LỜI

### Q1: Tại sao chọn 4 hops? Có thể thay đổi không?

**Trả lời:**
> "4 hops được chọn dựa trên thực nghiệm. VQA questions thường yêu cầu 2-4 bước reasoning. Tuy nhiên, `num_hops` là hyperparameter có thể tune. Trong ablation study, chúng tôi thấy 4 hops là sweet spot - 2 hops quá ít cho complex questions, 6+ hops không cải thiện thêm mà tăng computation."

### Q2: Memory có thể bị overflow không? Làm sao xử lý?

**Trả lời:**
> "Memory có kích thước cố định D (dimension). Nó không 'lưu' thông tin theo nghĩa truyền thống mà encode thông tin thành distributed representation. Gating mechanism tự động 'quên' thông tin ít quan trọng để nhường chỗ cho thông tin mới. Tương tự working memory của con người - có capacity limit nhưng adapt theo task."

### Q3: So với Transformer thuần, ưu điểm của NSM là gì?

**Trả lời:**
> "Transformer xử lý toàn bộ input một lần với self-attention. NSM xử lý tuần tự với explicit control flow.
>
> Ưu điểm:
> 1. Interpretability: Biết model đang focus vào đâu ở mỗi bước
> 2. Compositional: Dễ xử lý câu hỏi có nhiều sub-questions
> 3. Memory: Tích lũy thông tin qua các bước thay vì one-shot
>
> Nhược điểm: Sequential processing chậm hơn parallel transformer. Trade-off tùy application."

### Q4: Làm sao train NSM hiệu quả khi không có ground-truth cho intermediate steps?

**Trả lời:**
> "Đây là thách thức chính. Chúng tôi sử dụng end-to-end training với final answer supervision. Gradient flows back qua tất cả hops. Các auxiliary losses (object, relation) cung cấp supervision gián tiếp cho intermediate representations.
>
> Một hướng cải tiến là sử dụng GQA dataset có scene graph annotations để supervise Level 2, giúp representations tốt hơn cho Level 3."

### Q5: Computational cost của 3 levels so với baseline?

**Trả lời:**
> "Xấp xỉ 1.5-2x so với Q-Former baseline:
> - Level 1: +20% (object queries + box prediction)
> - Level 2: +40% (GNN message passing O(N²))
> - Level 3: +30% (4 reasoning hops)
>
> Tuy nhiên, accuracy gain 3-5% justify overhead này. Có thể optimize bằng cách giảm num_objects, num_hops cho deployment."

---

## GHI CHÚ CHO NGƯỜI THUYẾT TRÌNH

1. **Về diagram NSM**: Chuẩn bị animation cho Control-Read-Write flow

2. **Về multi-hop example**: Trace qua từng hop với highlighting rõ ràng

3. **Về attention visualization**: Có thể show real examples từ model outputs

4. **Thời gian**:
   - Slides 1-2: 2.5 phút
   - Slides 3-5: 5 phút (core NSM)
   - Slides 6-7: 3.5 phút
   - Slides 8-9: 3 phút
   - Slides 10-12: 3 phút
   - Tổng: ~17 phút (có thể rút gọn bằng cách skip slide 8, 10)

5. **Key message**: NSM biến multi-hop reasoning thành interpretable step-by-step process

6. **Tích hợp 3 levels**: Nhấn mạnh mỗi level có vai trò rõ ràng:
   - Level 1: "What objects?"
   - Level 2: "How related?"
   - Level 3: "Answer what?"
