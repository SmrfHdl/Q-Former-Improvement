# 🎤 SCRIPT THUYẾT TRÌNH BẰNG LỜI
## Level 1: Object Detection Path - Trích xuất Đặc trưng Không gian

---

## 📍 SLIDE 1: GIỚI THIỆU TỔNG QUAN

**[Bắt đầu - Giọng tự tin, chậm rãi]**

"Kính thưa Hội đồng, trong phần này em xin trình bày về **Level 1 - Object Detection Path**, đây là tầng đầu tiên và cũng là nền tảng của kiến trúc Q-Former cải tiến mà em đề xuất.

Trước khi đi vào chi tiết, em xin nhắc lại rằng kiến trúc của em gồm 3 tầng:
- **Tầng 1** - Object Detection Path - trích xuất đặc trưng đối tượng
- **Tầng 2** - Scene Graph Generation - xây dựng đồ thị quan hệ
- **Tầng 3** - Neural State Machine - suy luận đa bước

Hôm nay em sẽ tập trung giải thích tầng 1, vì đây là tầng quyết định chất lượng của toàn bộ pipeline."

---

## 📍 SLIDE 2: VẤN ĐỀ CẦN GIẢI QUYẾT

**[Giọng nhấn mạnh vấn đề]**

"Vậy tại sao chúng ta cần Level 1 này?

Trong Q-Former gốc của BLIP-2, có một hạn chế lớn, đó là model **không biết các object nằm ở đâu trong ảnh**. Nó chỉ học được đặc trưng toàn cục, mà không có khái niệm về vị trí không gian cụ thể.

Em xin lấy một ví dụ để Hội đồng dễ hình dung:

Giả sử có câu hỏi: *'Con mèo có đang NẰM TRÊN bàn không?'*

Để trả lời câu hỏi này, model cần phải:
- **Thứ nhất**: Biết con mèo ở đâu
- **Thứ hai**: Biết cái bàn ở đâu  
- **Thứ ba**: Hiểu quan hệ không gian 'nằm trên' giữa hai vật thể

Nhưng Q-Former gốc không làm được điều này vì nó thiếu thông tin về **vị trí hình học** của các object.

Đây chính là lý do em thiết kế Level 1 - để trích xuất được **bounding box** cho từng đối tượng trong ảnh."

---

## 📍 SLIDE 3: THÁCH THỨC KỸ THUẬT

**[Giọng suy tư, đặt vấn đề]**

"Tuy nhiên, việc trích xuất bounding box trong bài toán VQA đặt ra một thách thức lớn.

Các phương pháp object detection truyền thống như YOLO hay Faster R-CNN đều **cần dữ liệu gán nhãn bounding box** để huấn luyện. Nhưng trong bài toán VQA, chúng ta chỉ có cặp ảnh-câu hỏi-câu trả lời, **không có ground-truth boxes**.

Vậy câu hỏi đặt ra là: *Làm sao để xác định vị trí object mà không cần supervision?*

Và đây là điểm sáng tạo chính trong thiết kế của em: **Sử dụng attention weights để tính toán bounding box**."

---

## 📍 SLIDE 4: Ý TƯỞNG CỐT LÕI

**[Giọng hào hứng, giải thích ý tưởng]**

"Ý tưởng của em xuất phát từ một quan sát đơn giản:

Khi một object query thực hiện cross-attention với image patches, nó sẽ **chú ý nhiều hơn vào những patches chứa object mà nó quan tâm**.

Nói cách khác, **attention weights cho chúng ta biết query đang 'nhìn' vào đâu trong ảnh**.

Từ đó, em đề xuất: *Thay vì dùng MLP để predict bounding box trực tiếp, hãy TÍNH TOÁN bounding box từ phân phối attention*.

Cụ thể:
- **Tâm của box** = Vị trí trung bình có trọng số của các patches, với trọng số là attention weights
- **Kích thước box** = Độ phân tán của attention, đo bằng variance

Cách tiếp cận này có 2 ưu điểm lớn:
1. **Không cần supervision** - vì attention tự động học trong quá trình training
2. **Có thể giải thích được** - vì chúng ta biết model đang nhìn vào đâu"

---

## 📍 SLIDE 5: KIẾN TRÚC CHI TIẾT

**[Giọng rõ ràng, đi từng bước]**

"Bây giờ em xin trình bày kiến trúc chi tiết của Level 1.

**Thành phần thứ nhất: 2D Positional Encoding**

Đây là bước quan trọng đầu tiên. Trong Vision Transformer gốc, positional encoding chỉ là 1D, nghĩa là các patches được đánh số 1, 2, 3, 4... mà không có thông tin về vị trí 2D thực sự trong ảnh.

Em đã thiết kế một module **2D Positional Encoding** sử dụng công thức sinusoidal tương tự Transformer, nhưng mã hóa riêng cho tọa độ x và tọa độ y. Mỗi patch bây giờ sẽ biết chính xác nó nằm ở vị trí nào trong grid 16 nhân 16.

**Thành phần thứ hai: Object Queries**

Em sử dụng 32 learnable queries, tương tự như DETR. Mỗi query đóng vai trò như một 'slot' để phát hiện một object trong ảnh. Trong quá trình training, các queries sẽ tự động học cách specializing cho các loại object khác nhau.

**Thành phần thứ ba: Cross-Modal Transformer**

Đây là nơi diễn ra tương tác chính. Object queries thực hiện cross-attention với image patches đã được thêm positional encoding. Kết quả là:
- **Object features**: Đặc trưng của từng object
- **Attention weights**: Ma trận cho biết mỗi query attend vào patches nào

**Thành phần thứ tư: Attention-based Box Predictor**

Đây là module em tự thiết kế. Nó nhận attention weights và tọa độ patches làm input, rồi tính toán bounding box theo công thức mà em sẽ giải thích ở slide sau."

---

## 📍 SLIDE 6: CÔNG THỨC TÍNH BOUNDING BOX

**[Giọng chậm, giải thích toán học]**

"Em xin giải thích công thức tính bounding box.

**Bước 1: Tính tâm của box**

Tâm được tính bằng **weighted average** của tọa độ patches:

*Center x bằng tổng của attention nhân x, chia cho tổng attention*
*Center y bằng tổng của attention nhân y, chia cho tổng attention*

Ví dụ đơn giản: Nếu query attend mạnh nhất vào patch ở góc trên trái, thì center sẽ gần góc trên trái. Nếu attend đều vào giữa ảnh, center sẽ ở giữa.

**Bước 2: Tính kích thước box**

Kích thước được tính từ **variance** của phân phối attention:

*Width bằng 4 nhân căn bậc hai của variance theo x*
*Height bằng 4 nhân căn bậc hai của variance theo y*

Tại sao lại là variance? Vì:
- Attention tập trung vào một vùng nhỏ → Variance nhỏ → Object nhỏ
- Attention phân tán rộng → Variance lớn → Object lớn

Hệ số 4 được chọn để bao phủ khoảng 95% diện tích attention, tương tự quy tắc 2-sigma trong thống kê.

**Bước 3: Box refinement**

Sau khi có initial box, em còn áp dụng 2 layers refinement. Mỗi layer nhận object features và current box, rồi predict một delta nhỏ để tinh chỉnh. Điều này giúp boxes chính xác hơn sau khi model đã học được context."

---

## 📍 SLIDE 7: CONFIDENCE SCORE

**[Giọng giải thích logic]**

"Ngoài bounding box, em còn tính **confidence score** cho mỗi object.

Ý tưởng ở đây là sử dụng **entropy của attention distribution**.

Nếu attention tập trung vào một vùng cụ thể, entropy sẽ thấp, điều này cho thấy model tự tin rằng có object ở đó.

Ngược lại, nếu attention phân tán đều khắp nơi, entropy sẽ cao, cho thấy model không chắc chắn.

Công thức cụ thể:
*Entropy bằng âm tổng của attention nhân log attention*

Confidence cuối cùng là output của một MLP trừ đi normalized entropy. Cách này kết hợp được cả thông tin từ features lẫn từ attention pattern."

---

## 📍 SLIDE 8: LUỒNG XỬ LÝ DỮ LIỆU

**[Giọng tóm tắt flow]**

"Em xin tóm tắt luồng xử lý dữ liệu qua Level 1:

1. **Input**: Ảnh 224 nhân 224 pixels và câu hỏi text

2. **Vision Encoder**: ViT chia ảnh thành grid 16 nhân 16, tức 256 patches, mỗi patch là một vector 1024 chiều

3. **Projection**: Project xuống 768 chiều và thêm 2D positional encoding

4. **Cross-Modal Transformer**: 32 object queries attend vào 256 patches, output là 32 object features

5. **Box Prediction**: Từ attention weights, tính ra 32 bounding boxes

6. **Refinement**: Tinh chỉnh boxes qua 2 layers

**Output cuối cùng bao gồm:**
- 32 object features, mỗi cái 768 chiều
- 32 bounding boxes, format x, y, width, height chuẩn hóa từ 0 đến 1
- 32 confidence scores
- Attention maps để visualization

Tất cả output này sẽ được truyền sang Level 2 để xây dựng Scene Graph."

---

## 📍 SLIDE 9: SO SÁNH VỚI CÁC PHƯƠNG PHÁP KHÁC

**[Giọng tự tin, so sánh]**

"So với các phương pháp object detection truyền thống, cách tiếp cận của em có những ưu điểm sau:

**So với YOLO và Faster R-CNN:**
- Họ cần ground-truth boxes để train, em thì không
- Họ là separate module, em tích hợp end-to-end vào VQA pipeline

**So với DETR:**
- DETR cũng dùng object queries nhưng vẫn cần box supervision
- Em tính box từ attention, không cần supervision

**Điểm mạnh nhất** của phương pháp em là:
1. **Unsupervised localization** - không cần dữ liệu gán nhãn box
2. **Interpretable** - có thể visualize attention maps
3. **End-to-end trainable** - optimize cùng với task VQA chính
4. **Lightweight** - không thêm computational overhead đáng kể"

---

## 📍 SLIDE 10: VISUALIZATION VÀ DEMO

**[Giọng giới thiệu kết quả]**

"Em xin trình bày một số kết quả visualization.

Ở hình bên trái, các em có thể thấy bounding boxes được vẽ lên ảnh. Mỗi màu tương ứng với một object query khác nhau. Số bên cạnh là confidence score.

Ở hình bên phải là attention heatmap. Màu đỏ là vùng được attend nhiều, màu xanh là attend ít. Có thể thấy attention tập trung vào các object chính trong ảnh.

Điều này chứng minh rằng model thực sự học được cách **định vị object** thông qua attention mechanism, mà không cần explicit supervision.

Em cũng đã viết một script visualization để Hội đồng có thể tự chạy và kiểm chứng kết quả."

---

## 📍 SLIDE 11: KẾT LUẬN LEVEL 1

**[Giọng tổng kết]**

"Tóm lại, Level 1 - Object Detection Path đóng góp những điểm sau:

**Thứ nhất**, thiết kế **2D Positional Encoding** giúp mã hóa vị trí không gian rõ ràng cho từng image patch.

**Thứ hai**, phương pháp **Attention-based Box Prediction** cho phép tính toán bounding box từ attention distribution mà không cần ground-truth supervision.

**Thứ ba**, cơ chế **Iterative Refinement** giúp tinh chỉnh boxes để có kết quả chính xác hơn.

**Thứ tư**, khả năng **Visualization** qua attention maps giúp giải thích được quyết định của model.

Output của Level 1, bao gồm object features và bounding boxes, sẽ là input cho Level 2 - nơi chúng ta xây dựng Scene Graph để mô hình hóa quan hệ không gian giữa các objects.

Em xin kết thúc phần trình bày về Level 1. Hội đồng có câu hỏi nào không ạ?"

---

## 📍 PHẦN TRẢ LỜI CÂU HỎI DỰ KIẾN

### Câu hỏi 1: "Bounding boxes có chính xác không? Đã đánh giá bằng metric nào?"

**[Giọng thành thật, giải thích]**

"Dạ, em xin trả lời câu hỏi của Thầy/Cô.

Thực ra, mục tiêu của em không phải là object detection chính xác theo nghĩa truyền thống. Em không đánh giá bằng mAP hay IoU như các bài toán detection.

Mục tiêu của em là **soft spatial localization** - tức là xác định được vùng không gian mà object chiếm trong ảnh, đủ để phục vụ cho reasoning ở các level sau.

Để đánh giá, em sử dụng:
1. **Qualitative evaluation**: Visualization cho thấy boxes cover đúng objects
2. **Downstream task performance**: VQA accuracy tăng chứng tỏ spatial information hữu ích
3. **Attention coherence**: Attention maps tập trung vào đúng vùng liên quan

Em cho rằng với bài toán VQA, quan trọng là model hiểu được 'ở đâu' để trả lời câu hỏi, chứ không nhất thiết phải có pixel-perfect bounding box."

---

### Câu hỏi 2: "Tại sao dùng 32 object queries? Số này có ý nghĩa gì?"

**[Giọng giải thích]**

"Dạ, số 32 được em chọn dựa trên các yếu tố sau:

Thứ nhất, theo các nghiên cứu trước về scene understanding, một ảnh thông thường chứa khoảng 5-15 objects đáng chú ý. 32 là đủ để cover với buffer.

Thứ hai, em đã thử nghiệm với 16, 32, và 64 queries. Kết quả cho thấy 32 là điểm cân bằng tốt giữa coverage và computational cost.

Thứ ba, 32 cũng là số được sử dụng trong Q-Former gốc của BLIP-2, nên em giữ để fair comparison.

Trong thực tế, không phải tất cả 32 queries đều tìm được object. Những queries không tìm được object sẽ có confidence score thấp và bị filter out trong các bước sau."

---

### Câu hỏi 3: "Attention-based box có điểm yếu gì?"

**[Giọng khách quan, nhận xét hạn chế]**

"Dạ, em xin thừa nhận một số hạn chế:

**Thứ nhất**, với objects rất nhỏ chiếm ít patches, variance có thể không ổn định, dẫn đến box không chính xác.

**Thứ hai**, khi có nhiều objects giống nhau ở gần nhau, một query có thể attend vào nhiều objects cùng lúc, làm box bị lớn hơn thực tế.

**Thứ ba**, method này dựa vào giả định rằng attention tập trung vào đúng object. Nếu attention bị noisy, box sẽ sai.

Để giảm thiểu các vấn đề này, em đã thêm:
- Box refinement layers để học correction
- Clamp giá trị width/height có minimum threshold
- Confidence score dựa trên entropy để filter boxes không đáng tin

Em nghĩ đây là trade-off hợp lý để đổi lấy việc không cần box supervision."

---

### Câu hỏi 4: "Có thể áp dụng cho bài toán object detection thuần túy không?"

**[Giọng thận trọng]**

"Dạ, câu hỏi rất hay ạ.

Về lý thuyết, có thể adapt phương pháp này cho object detection, nhưng em nghĩ nó sẽ **không competitive** với các detector chuyên dụng vì:

1. Thiếu explicit classification head cho object categories
2. Không có NMS hay post-processing tinh vi
3. Không optimize trực tiếp cho detection metrics

Phương pháp của em được thiết kế specifically cho VQA, nơi mục tiêu chính là **hiểu scene để trả lời câu hỏi**, không phải detect chính xác từng object.

Tuy nhiên, ý tưởng attention-based localization có thể là hướng nghiên cứu thú vị cho **weakly-supervised object detection**, nơi chúng ta có image-level labels nhưng không có box annotations."

---

## 📝 LƯU Ý KHI THUYẾT TRÌNH

1. **Giọng nói**: Chậm rãi, rõ ràng, có nhấn nhá ở các điểm quan trọng
2. **Eye contact**: Nhìn vào Hội đồng, không đọc slides
3. **Thời gian**: Khoảng 10-12 phút cho Level 1, để dành thời gian cho Q&A
4. **Pointer**: Dùng laser pointer khi giải thích diagram
5. **Tự tin**: Đây là công trình của mình, mình hiểu rõ nhất

---

*Chúc bạn bảo vệ thành công! 🎓*
