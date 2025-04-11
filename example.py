import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# --- Tạo các đối tượng giả cho metadata ---

# Ví dụ: một danh sách các ID sản phẩm
product_ids = ["P1", "P2", "P3", "P4", "P5"]

# Tạo một LabelEncoder và huấn luyện nó trên các ID sản phẩm.
# LabelEncoder sẽ chuyển đổi các ID dạng chuỗi thành các số nguyên
product_encoder = LabelEncoder()
product_encoder.fit(product_ids)

# Xây dựng một ánh xạ ngược: từ chỉ số đến ID sản phẩm.
# Điều này giúp chúng ta có thể lấy lại ID sản phẩm từ chỉ số số nguyên
idx_to_product = {idx: pid for idx, pid in enumerate(product_encoder.classes_)}

# Tạo một ánh xạ đơn giản từ ID sản phẩm đến tên sản phẩm.
# Điều này cho phép hiển thị tên thân thiện với người dùng thay vì ID kỹ thuật
product_to_name = {
    "P1": "Ultra HD Television",
    "P2": "Wireless Headphones",
    "P3": "Coffee Maker",
    "P4": "Smartphone",
    "P5": "Electric Kettle"
}

# Tạo một bộ chuẩn hóa đánh giá dựa trên các giá trị từ 1 đến 5.
# MinMaxScaler sẽ chuẩn hóa giá trị đánh giá vào khoảng [0,1]
ratings = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
rating_scaler = MinMaxScaler().fit(ratings)

# Tạo ánh xạ danh mục giả:
# Để đơn giản, chúng ta ánh xạ các chỉ số được mã hóa vào các danh mục.
# Ví dụ: chỉ số 0 và 1 thuộc "Electronics", 2 thuộc "Home", v.v.
category_mapping = {
    0: "Electronics",
    1: "Electronics",
    2: "Home",
    3: "Electronics",
    4: "Kitchen"
}

# Tạo các vector nhúng (embeddings) giả.
# Giả sử chiều của vector nhúng là 16.
# Các vector nhúng này đại diện cho đặc trưng của sản phẩm trong không gian ngữ nghĩa
product_embeddings = np.random.rand(len(product_ids), 16)
# Đối với vector nhúng người dùng, giả sử chúng ta có 10 người dùng giả.
# Các vector này đại diện cho sở thích và hành vi của người dùng
user_embeddings = np.random.rand(10, 16)

# --- Đóng gói metadata vào một từ điển ---
# Tất cả thông tin được gom vào một từ điển để dễ dàng sử dụng và lưu trữ

metadata = {
    "product_encoder": product_encoder,
    "idx_to_product": idx_to_product,
    "product_to_name": product_to_name,
    "rating_scaler": rating_scaler,
    "category_mapping": category_mapping,
    "user_embeddings": user_embeddings,
    "product_embeddings": product_embeddings
}

# --- Lưu metadata vào một tệp pickle ---
# Sử dụng pickle để lưu đối tượng Python vào tệp nhị phân để sử dụng sau này
with open("amazon_recommender_metadata_updated.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("Metadata đã được lưu thành công.")