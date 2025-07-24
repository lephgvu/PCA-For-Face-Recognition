# --------------- Phát hiện khuôn mặt và chuẩn hóa ảnh khuôn mặt ---------------
# Thư viện cần thiết
import cv2
import numpy as np

# 1. Đọc ảnh và chuyển sang ảnh xám
img = cv2.imread('D:/python_project/PCA_For_Face_Recognition/test.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Khởi tạo bộ phát hiện khuôn mặt (Phương pháp Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3. Phát hiện khuôn mặt trong ảnh xám 
faces = face_cascade.detectMultiScale(
                                        gray, 
                                        scaleFactor=1.2, # Tỷ lệ thu phóng trong khoảng 1.1-1.3, ảnh sẽ giảm 20% mỗi lần, giúp phát hiện khuôn mặt nhỏ hơn nhưng chậm
                                        minNeighbors=6, # Số lượng vùng lân cận cần thiết để xác định một khuôn mặt, giá trị từ 3-6, giá trị cao hơn sẽ giảm số lượng phát hiện nhưng tăng độ chính xác
                                        minSize=(20, 20) # Kích thước tối thiểu của khuôn mặt, giá trị này có thể điều chỉnh tùy theo kích thước khuôn mặt trong ảnh
)
# Kết quả trả về danh sách các tọa độ (x, y, w, h) của các khuôn mặt phát hiện được


# 4. Khởi tạo các biến cần thiết
common_size = (50, 50)        # Chuẩn hóa kích thước khuôn mặt về một kích thước chung
face_vectors = []             # Danh sách lưu các vector khuôn mặt đã chuẩn hóa
bounding_boxes = []           # Danh sách lưu vị trí khuôn mặt

# 5. Lặp qua từng khuôn mặt
for (x, y, w, h) in faces:
    face = img[y:y+h, x:x+w]                                   # Cắt khuôn mặt từ ảnh gốc
    face_resized = cv2.resize(face, common_size)               # Resize về kích thước chuẩn
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY) # Chuyển sang xám
    face_vector = face_gray.flatten().astype(np.float32)       # Flatten (làm phẳng) thành vector 1 chiều
    
    face_mean = np.mean(face_vector)                           # Tính trung bình của vector khuôn mặt
    face_normalized = face_vector - face_mean                  # Chuẩn hóa bằng cách trừ trung bình

    # Lưu trữ vector khuôn mặt đã chuẩn hóa và bounding box
    face_vectors.append(face_normalized)
    bounding_boxes.append((x, y, w, h))

# 6. Tạo ma trận X từ các vector khuôn mặt
if face_vectors:
    X = np.stack(face_vectors)  # Ma trận X có shape (n_samples, 2500)
    print(f"Matrix X shape: {X.shape}")  # In kích thước ma trận
else:
    print("No faces detected.")

# 7. Hiển thị ảnh với khung khuôn mặt
for (x, y, w, h) in bounding_boxes:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow('Detected Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --------------- Sử dụng PCA để giảm chiều dữ liệu khuôn mặt ---------------
# Bước 1: Thực hiện SVD trên ma trận X đã chuẩn hóa
U, S, Vt = np.linalg.svd(X, full_matrices=False)
# Vt.shape = (2500, 2500) → mỗi hàng của Vt là 1 eigenface

# Bước 2: Chọn k chiều chính (eigenfaces)
k = 50  # số chiều PCA giữ lại
eigenfaces = Vt[:k, :]  # ma trận k x 2500, mỗi hàng là 1 eigenface

# Bước 3: Biểu diễn các khuôn mặt trong không gian PCA (mỗi khuôn mặt sẽ được biểu diễn bằng vector 50 chiều)
X_pca = X @ eigenfaces.T  # Mỗi khuôn mặt ban đầu (2500 chiều) bây giờ đã được biểu diễn bằng vector 50 chiều 

# Bước 4: Trích xuất khuôn mặt đầu tiên từ không gian PCA
approx_face0 = X_pca[0] @ eigenfaces  # vector 1x2500
approx_face0_image = approx_face0.reshape(50, 50)

# Bước 5: Hiển thị ảnh khuôn mặt đầu tiên được tái tạo từ PCA
import matplotlib.pyplot as plt
plt.imshow(approx_face0_image, cmap='gray')
plt.title('Reconstructed Face from PCA')   
plt.axis('off')
plt.show()
# --------------- Kết thúc PCA và hiển thị ảnh khuôn mặt đã tái tạo ---------------
