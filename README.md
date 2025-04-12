# Nghiên cứu và xây dựng hệ thống gợi ý sản phẩm sử dụng Deep Matrix Factorization

## Giới thiệu

Dự án "Nghiên cứu và xây dựng hệ thống gợi ý sản phẩm sử dụng thuật toán Deep Matrix Factorization" là đồ án thực tập tốt nghiệp nhằm xây dựng một hệ thống gợi ý thông minh kết hợp nhiều phương pháp Machine Learning hiện đại. Hệ thống này sử dụng Deep Matrix Factorization (DMF) làm nền tảng chính, kết hợp với K-Nearest Neighbors (KNN) và Popularity-Based Model để tạo ra trải nghiệm gợi ý sản phẩm cá nhân hóa và chính xác.

Trong bối cảnh thương mại điện tử, việc gợi ý sản phẩm phù hợp giúp người dùng tiết kiệm thời gian tìm kiếm và tăng khả năng tìm thấy sản phẩm ưng ý. Hệ thống được thiết kế để xử lý hiệu quả các thách thức như vấn đề cold-start (người dùng mới), cập nhật theo thời gian thực, và cá nhân hóa sâu dựa trên hành vi người dùng.

## Tính năng chính

- **Gợi ý cá nhân hóa**: Sử dụng mô hình Deep Matrix Factorization để học biểu diễn phi tuyến tính của người dùng và sản phẩm
- **Xử lý tình huống cold-start**: Kết hợp Popularity-Based Model để gợi ý sản phẩm phổ biến cho người dùng mới
- **Tính toán tương đồng**: Sử dụng KNN và độ tương đồng cosine để tìm sản phẩm gần nhất trong không gian đặc trưng
- **Học tăng dần (Incremental Learning)**: Cập nhật mô hình theo thời gian thực khi có tương tác mới của người dùng
- **Phân cụm người dùng**: Áp dụng thuật toán K-Means để phân nhóm người dùng theo hành vi tương tự
- **Tích hợp MongoDB**: Lưu trữ và quản lý dữ liệu tương tác người dùng hiệu quả
- **Cá nhân hóa**: Kết hợp thông tin danh mục, từ khóa và lịch sử tìm kiếm để tăng độ chính xác gợi ý
- **Giao diện web thân thiện**: Xây dựng bằng Flask, cung cấp trải nghiệm người dùng tốt với thiết kế responsive

## Công nghệ sử dụng

- **Python 3.8+**: Ngôn ngữ lập trình chính
- **TensorFlow 2.x/Keras**: Xây dựng và huấn luyện mô hình Deep Matrix Factorization
- **Scikit-learn**: Triển khai thuật toán KNN, K-Means và xử lý dữ liệu
- **MongoDB**: Cơ sở dữ liệu NoSQL để lưu trữ thông tin người dùng, sản phẩm và tương tác
- **Flask**: Framework web Python nhẹ để xây dựng giao diện người dùng
- **Pandas/NumPy**: Thư viện xử lý và phân tích dữ liệu
- **Bootstrap**: Framework CSS để thiết kế giao diện web responsive
- **Matplotlib/Seaborn**: Thư viện trực quan hóa dữ liệu
- **Bcrypt**: Mã hóa mật khẩu người dùng
- **Jupyter Notebook**: Môi trường phát triển và thử nghiệm mô hình

## Cấu trúc dự án

```
N47_HeThongGoiYSanPham/
│
├── data/
│   ├── __pycache__/                # Python cache
│   └── database.py                 # Module kết nối và tương tác với MongoDB
│
├── static/
│   ├── css/
│   │   └── style.css               # CSS tùy chỉnh cho giao diện web
│   ├── img/
│   │   └── placeholder.png         # Ảnh mặc định cho sản phẩm không có hình
│   └── js/
│       └── main.js                 # JavaScript xử lý các tương tác của người dùng
│
├── templates/
│   ├── auth/                       # Templates cho chức năng
│   │   ├── login.html              # Trang đăng nhập
│   │   └── signup.html             # Trang đăng ký
│   ├── cart.html                   # Trang giỏ hàng
│   ├── category.html               # Trang hiển thị sản phẩm theo danh mục
│   ├── home.html                   # Trang chủ
│   ├── layout.html                 # Template chung cho tất cả các trang
│   ├── product_detail.html         # Trang chi tiết sản phẩm
│   ├── require_login.html          # Trang yêu cầu đăng nhập
│   └── search_results.html         # Trang kết quả tìm kiếm
│
├── utils/
│   ├── __pycache__/                # Python cache
│   ├── __init__.py                 # Khởi tạo package utils
│   ├── data_processing.py          # Module xử lý dữ liệu
│   └── recommendation.py           # Module chứa logic gợi ý sản phẩm
│
├── amazon_cleaned.csv              # Dữ liệu Amazon đã được làm sạch và cân bằng
├── amazon_recommender_metadata_updated.pkl  # Metadata của mô hình
├── amazon_recommender_model_updated.keras   # Mô hình DMF đã huấn luyện
│
├── app.py                          # Ứng dụng Flask chính
│
├── best_model.keras                # Mô hình tốt nhất được lưu từ quá trình huấn luyện
│
├── cart_categories_cntt@gmail.com.json             # Lưu danh mục giỏ hàng của người dùng
│
├── clean_data.ipynb                # Notebook xử lý, làm sạch và cân bằng dữ liệu
│
├── example.py                      # File ví dụ tạo dữ liệu metadata mẫu
│
├── Link amazon reviews 2023.txt           # Link đến bộ dữ liệu Amazon Reviews 2023
├── Link amazon sales dataset (16 columns).txt  # Link đến bộ dữ liệu Amazon Sales Dataset from Kaggle
├── Link kết nối MongoDB.txt               # Link kết nối MongoDB Compass
│
├── MongoDB và Fine-Tuning.ipynb    # Notebook triển khai hệ thống học tăng dần
│
├── product_embeddings.npy          # File lưu vector embedding của sản phẩm
│
├── True First DMF.ipynb            # Notebook triển khai mô hình Deep Matrix Factorization
│
└── user_embeddings.npy             # File lưu vector embedding của người dùng
```

## Các mô hình sử dụng

### 1. Deep Matrix Factorization (DMF)
![Kiến trúc mô hình Deep Matrix Factorization](dmf_architecture.jpg)
- **Kiến trúc**: Mạng nơ-ron sâu với lớp embedding 32 chiều cho cả người dùng và sản phẩm, sau đó là các lớp kết nối đầy đủ (dense layers) với kích thước giảm dần 256→128→64→1
- **Regularization**: Sử dụng L2 regularization (0.01) cho các lớp embedding
- **Batch Normalization**: Áp dụng sau mỗi lớp dense để ổn định quá trình huấn luyện
- **Activation**: Sử dụng LeakyReLU (alpha=0.0001) thay vì ReLU để tránh vấn đề "dying neurons"
- **Dropout**: Tỷ lệ 0.2 áp dụng cho hai lớp dense đầu tiên để giảm overfitting
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam với learning rate thấp (0.00001) và clipnorm=1.0

### 2. K-Nearest Neighbors (KNN)
- Sử dụng từ thư viện scikit-learn với tham số metric="cosine"
- Tìm các sản phẩm có vector embedding gần nhất trong không gian đặc trưng
- Kết hợp nhiều yếu tố khi tính điểm tổng hợp:
  - Độ tương đồng cosine (30%)
  - Điểm phổ biến (10%)
  - Điểm danh mục (20-30%)
  - Điểm từ khóa (30%)

### 3. Popularity-Based Model
- Xếp hạng sản phẩm dựa trên tích của số lượng đánh giá và điểm đánh giá trung bình
- Chuẩn hóa điểm phổ biến về khoảng [0, 1]
- Sử dụng chủ yếu cho người dùng mới hoặc khi thiếu dữ liệu

### 4. Thuật toán K-Means 
- Phân cụm người dùng hoặc sản phẩm dựa trên vector embedding
- Sử dụng để phân tích hành vi và tạo insight từ dữ liệu

## Quy trình gợi ý sản phẩm

1. **Người dùng mới**:
   - Sử dụng Popularity-Based Model để gợi ý các sản phẩm phổ biến nhất
   - Nếu có thông tin về từ khóa tìm kiếm hoặc danh mục quan tâm, hệ thống kết hợp với điểm phổ biến

2. **Người dùng đã có lịch sử**:
   - Thu thập thông tin từ nhiều nguồn (từ khóa tìm kiếm gần đây, danh mục sản phẩm đã xem)
   - Sử dụng vector embedding của người dùng (đã học từ mô hình Deep Matrix Factorization)
   - Kết hợp KNN để tìm các sản phẩm tương đồng nhất
   - Tính điểm tổng hợp dựa trên nhiều tiêu chí
   - Đưa ra danh sách đề xuất sản phẩm phù hợp nhất

3. **Cập nhật mô hình**:
   - **Batch Fine-Tuning**: Định kỳ cập nhật mô hình với dữ liệu tương tác mới tích lũy
   - **Cập nhật Embeddings theo thời gian thực**: Cập nhật ngay lập tức vector embedding dựa trên từng tương tác mới của người dùng
   - **Mở rộng lớp Embedding**: Tự động mở rộng kích thước lớp embedding khi có người dùng hoặc sản phẩm mới

## Cài đặt và sử dụng

### Yêu cầu

- Python 3.8 đến 3.12
- MongoDB
- TensorFlow 2.x

### Cài đặt

1. Clone repository:
```bash
git clone https://github.com/yourusername/N47_HeThongGoiYSanPham.git
cd N47_HeThongGoiYSanPham
```

2. Cấu hình kết nối MongoDB Compass:
   - Sử dụng chuỗi kết nối:
   ```
   mongodb+srv://nhoktk000:Quan26012004@cluster.ap9ga.mongodb.net/ecommerce_db?retryWrites=true&w=majority
   ```

3. Khởi chạy ứng dụng:
```bash
python app.py
```

### Huấn luyện mô hình từ đầu

1. Xử lý và làm sạch dữ liệu:
   - Chạy notebook `clean_data.ipynb`

2. Huấn luyện mô hình DMF:
   - Chạy notebook `True First DMF.ipynb`

3. Cấu hình học tăng dần:
   - Chạy notebook `MongoDB và Fine-Tuning.ipynb`

### Sử dụng mô hình đã huấn luyện

1. Đảm bảo các file sau đã tồn tại:
   - `amazon_recommender_model_updated.keras`
   - `amazon_recommender_metadata_updated.pkl`
  
hoặc:
   - `amazon_recommender_model.keras`
   - `amazon_recommender_metadata.pkl`
Là hai files mô hình đã huấn luyện cho file True First DMF.ipynb

2. Khởi chạy ứng dụng web:
```bash
python app.py
```

3. Truy cập ứng dụng web qua trình duyệt:
```
http://localhost:5000
```

## Hướng dẫn sử dụng

1. **Đăng ký/Đăng nhập**: Tạo tài khoản mới hoặc đăng nhập vào hệ thống
2. **Xem sản phẩm**: Duyệt qua danh sách sản phẩm hoặc tìm kiếm theo từ khóa
3. **Gợi ý cá nhân hóa**: Xem các sản phẩm được gợi ý dựa trên lịch sử tương tác của bạn
4. **Giỏ hàng**: Thêm sản phẩm vào giỏ hàng và xem tổng tiền
5. **Tìm kiếm**: Tìm sản phẩm theo từ khóa, tên hoặc danh mục

## Tài liệu chi tiết

### API Endpoints

- **GET /**: Trang chủ
- **GET /product/<product_id>**: Chi tiết sản phẩm
- **GET /category/<category>**: Sản phẩm theo danh mục
- **GET /search?q=<query>**: Tìm kiếm sản phẩm
- **GET /cart**: Xem giỏ hàng
- **POST /add_to_cart/<product_id>**: Thêm sản phẩm vào giỏ hàng
- **GET /remove_from_cart/<product_id>**: Xóa sản phẩm khỏi giỏ hàng
- **POST /update_cart**: Cập nhật số lượng sản phẩm trong giỏ hàng
- **GET /login**: Trang đăng nhập
- **POST /login**: Xử lý đăng nhập
- **GET /signup**: Trang đăng ký
- **POST /signup**: Xử lý đăng ký
- **GET /logout**: Đăng xuất

### Cấu trúc cơ sở dữ liệu MongoDB

- **Collection users**: Lưu thông tin người dùng
  - username: Tên người dùng
  - email: Email (dùng làm ID đăng nhập)
  - password: Mật khẩu đã mã hóa
  - created_at: Thời gian tạo tài khoản
  - cart_history: Lịch sử giỏ hàng
  - view_history: Lịch sử xem sản phẩm

- **Collection user_interactions**: Lưu tương tác người dùng-sản phẩm
  - user_id: ID người dùng
  - product_id: ID sản phẩm
  - interaction_type: Loại tương tác (view, add_to_cart, purchase, rate)
  - rating: Điểm đánh giá (nếu có)
  - timestamp: Thời gian tương tác

- **Collection products**: Lưu thông tin sản phẩm
  - product_id: ID sản phẩm
  - product_name: Tên sản phẩm
  - keywords: Từ khóa liên quan
  - category: Danh mục sản phẩm
  - created_at: Thời gian tạo

- **Collection embeddings**: Lưu vector embedding
  - type: Loại (user hoặc product)
  - user_id/product_id: ID người dùng/sản phẩm
  - embedding: Vector embedding

- **Collection browsing_history**: Lưu lịch sử duyệt web
  - user_id: ID người dùng
  - product_id: ID sản phẩm
  - timestamp: Thời gian xem
  - category: Danh mục sản phẩm (nếu có)

- **Collection metadata_collection**: Lưu metadata của hệ thống
  - key: Khóa metadata
  - value: Giá trị tương ứng

## Tác Giả

- **[Nhóm 47 - Thực tập tốt nghiệp CNTT - FitVAA ]**

## Giấy phép

Dự án này được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.

## Lời Cảm Ơn

- Cảm ơn **[Giảng viên Tiến sĩ Trần Nguyên Bảo]** cho sự hỗ trợ và định hướng

## Tài liệu tham khảo

- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Keras Documentation](https://keras.io/guides/)
- [Deep Matrix Factorization with TensorFlow](https://st1990.hatenablog.com/entry/2019/03/16/231554)
- [Amazon Sales Dataset on Kaggle](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)
- [Building Recommendation System Using KNN](https://aurigait.com/blog/recommendation-system-using-knn/)
- [What Are Recommendation Systems in Machine Learning](https://www.analyticssteps.com/blogs/what-are-recommendation-systems-machine-learning)
- [Amazon Reviews 2023 Dataset](https://amazon-reviews-2023.github.io/)
