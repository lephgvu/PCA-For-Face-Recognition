# Import thư viện cần thiết 
import cv2
import numpy as np

# Đọc ảnh từ đường dẫn
img = cv2.imread('D:/test.jpg') 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Chuyển đổi ảnh sang thang độ xám

# Khởi tạo bộ phát hiện khuôn mặt (Viola-Jones với Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Phát hiện khuôn mặt
faces = face_cascade.detectMultiScale(gray, 
                                      scaleFactor=1.2, # Tỷ lệ thu phóng trong khoảng 1.1-1.3, ảnh sẽ giảm 10% mỗi lần, giúp phát hiện khuôn mặt nhỏ hơn nhưng chậm
                                      minNeighbors=6, # Số lượng vùng lân cận cần thiết để xác định một khuôn mặt, giá trị từ 3-6, giá trị cao hơn sẽ giảm số lượng phát hiện nhưng tăng độ chính xác  
                                      minSize=(20, 20)) # Kích thước tối thiểu của khuôn mặt, giá trị này có thể điều chỉnh tùy theo kích thước khuôn mặt trong ảnh

# Kích thước chuẩn
common_size = (100, 100)
face_images = []
threshold_energy = 1e6  # hoặc thử 2e6, 5e5...

# Duyệt qua từng khuôn mặt
for (x, y, w, h) in faces:
    face = img[y:y+h, x:x+w]
    face_resized = cv2.resize(face, common_size)
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    face_vector = face_gray.flatten().astype(np.float32)
    
    # Chuẩn hóa (trừ trung bình)
    face_mean = np.mean(face_vector)
    face_normalized = face_vector - face_mean
    energy = np.linalg.norm(face_normalized) ** 2
    face_images.append((face_normalized, energy, (x, y, w, h)))

# Đếm số khuôn mặt với năng lượng vượt ngưỡng
num_detected_faces = sum(energy > threshold_energy for (_, energy, _) in face_images)

print(f"Number of faces recognized: {num_detected_faces}")

# Hiển thị kết quả
for (_, energy, (x, y, w, h)) in face_images:
    if energy > threshold_energy:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
