# K4 — Ngày 1: Bài Tập & Phản Ánh

## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature

Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)

> Khi tăng temperature, độ đa dạng và tính ngẫu nhiên của câu trả lời tăng rõ rệt: ở mức 0.0 câu trả lời mang tính xác định, ngắn gọn và tập trung vào các sự thật phổ biến nhất (như xuất khẩu cà phê Robusta, hang Sơn Đoòng); ở mức 0.5 - 1.0 câu văn trở nên tự nhiên, phong phú và khai thác nhiều chủ đề văn hóa/ẩm thực khác nhau; đến mức 1.5, mô hình ưu tiên chọn các token có xác suất thấp dẫn đến câu văn bay bổng quá đà, cấu trúc ngữ pháp lủng củng và có dấu hiệu bịa đặt thông tin

### Câu 1.2 — Chọn temperature cho sản phẩm

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**

> Đặt temperature ở mức thấp từ 0.0 đến 0.2 nếu khách hàng đỏi hỏi tính chính xác trong câu trả lời theo đúng tài liệu của doanh nghiệp để tránh cung cấp sai thông tin

### Câu 1.3 — Đánh đổi chi phí

Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**

> chi phí: 10000 _ 3 = 30000 lượt/ngày
> -> 30000 _ 350 = 10.500.000 token/ngày, chi phí in/output cho gpt-4o hiện tại là $2.5/$10 / 1M token,gpt 4o-mini có chi phí $0.15/$0.6 / 1M token, vậy chi phí gpt-4o sẽ cao hơn gpt-4o-mini khoảng 16.67 lần
> Trường hợp xứng đáng dùng gpt-4o: khi cần độ chính xác cao, khả năng hiểu ngữ cảnh phức tạp, hoặc tạo ra các phản hồi sáng tạo, ví dụ như viết nội dung marketing, phân tích dữ liệu, hoặc hỗ trợ lập trình...
> Trường hợp nên dùng gpt-4o-mini: khi yêu cầu phản hồi nhanh, chi phí thấp, hoặc các tác vụ đơn giản như trả lời câu hỏi cơ bản, tra cứu thông tin, hoặc hỗ trợ khách hàng với các vấn đề phổ biến

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona

Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:

- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)

> Với prompt “giáo viên tiểu học”, phản hồi thường ngắn, dùng từ vựng đơn giản và ví dụ gần gũi như một cuốn sổ chung mà nhiều người cùng giữ bản sao. Với prompt “chuyên gia tài chính”, phản hồi có thể dài và sâu hơn, sử dụng các thuật ngữ như sổ cái phân tán, hàm băm, cơ chế đồng thuận, tính bất biến và smart contract.
> System prompt định hướng vai trò, đối tượng người đọc, mức độ chi tiết, phong cách diễn đạt và loại ví dụ mà model lựa chọn, nhưng không bảo đảm tuyệt đối mọi phản hồi đều khác nhau hoàn toàn.

### Câu 2.2 — tiktoken vs đếm từ

Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**

> Số token theo tiktoken là 150, trong khi ước lượng số từ / 0.75 là 133.33, chênh nhau khoảng 12.5%. Tiếng Việt thường tốn nhiều token hơn tiếng Anh cùng độ dài vì tiếng Việt có nhiều từ ghép và dấu câu, cũng như các ký tự đặc biệt (như dấu sắc, huyền, hỏi, ngã, nặng) được mã hóa thành nhiều token hơn trong mô hình ngôn ngữ

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming

**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)

> Streaming quan trọng nhất khi người dùng cần phản hồi nhanh, chẳng hạn như trong các ứng dụng chat trực tiếp, nơi mà việc hiển thị từng phần của câu trả lời ngay lập tức giúp cải thiện trải nghiệm người dùng.
> Non-streaming lại phù hợp hơn trong các tình huống mà độ chính xác và toàn vẹn của phản hồi là quan trọng, chẳng hạn như khi gửi dữ liệu nhạy cảm hoặc khi cần phân tích toàn bộ câu trả lời trước khi hiển thị cho người dùng

### Câu 3.2 — Vì sao backoff theo cấp số nhân?

**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**

> Exponential backoff giúp giảm tải cho server bằng cách tăng dần thời gian chờ giữa các lần retry, từ đó giảm số lượng yêu cầu đồng thời và tránh tình trạng khi hàng nghìn client cùng retry với delay cố định giống nhau.
> Nếu tất cả client retry với delay cố định, server có thể bị quá tải và dẫn đến tình trạng từ chối dịch vụ (DoS), trong khi exponential backoff giúp phân tán các yêu cầu theo thời gian, tăng khả năng thành công của các retry và cải thiện hiệu suất tổng thể của hệ thống.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona

**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**

> Persona: Bạn là trợ giảng lập trình Python thân thiện của VinUni. Hãy luôn trả lời ngắn gọn, súc tích bằng tiếng Việt, kèm ví dụ minh họa khi cần. Nếu câu hỏi không rõ ràng, hãy yêu cầu người dùng cung cấp thêm thông tin. Tránh đưa ra thông tin sai lệch hoặc không xác thực.
> Lựa chọn từ ngữ: "trả lời ngắn gọn, súc tích" giúp người dùng nhanh chóng nắm bắt thông tin mà không bị quá tải
> "kèm ví dụ minh họa" giúp người dùng hiểu rõ hơn về cách áp dụng kiến thức vào thực tế.

### Câu 4.2 — Hạn chế & cải thiện

**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**

> Hạn chế lớn nhất là không có bộ nhớ dài hạn, dẫn đến việc trợ lý không thể ghi nhớ các cuộc trò chuyện trước đó và cung cấp phản hồi dựa trên ngữ cảnh lâu dài. Cải thiện: triển khai một cơ chế lưu trữ lịch sử trò chuyện trong cơ sở dữ liệu, cho phép trợ lý truy xuất thông tin từ các cuộc trò chuyện trước đó khi cần thiết. Cách triển khai: sử dụng một cơ sở dữ liệu NoSQL để lưu trữ các đoạn hội thoại, với mỗi người dùng có một ID duy nhất, và khi người dùng gửi câu hỏi mới, trợ lý sẽ truy vấn cơ sở dữ liệu để lấy thông tin liên quan từ các cuộc trò chuyện trước đó.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
