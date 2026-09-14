import cv2
import numpy as np

# 1. 解码视频并挑出最清晰的一帧
cap = cv2.VideoCapture("test.mp4")
scores, frames = [], []
idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    scores.append((idx, score))
    frames.append(frame)
    idx += 1
cap.release()

# 找出最高分的那一帧作为基准
scores.sort(key=lambda x: x[1], reverse=True)
base_idx = scores[0][0]
base_frame = frames[base_idx]
cv2.imwrite("best_frame.jpg", base_frame)

# 2. 初始化 SIFT 特征对齐工具
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()
kp_base, des_base = sift.detectAndCompute(cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY), None)
h, w, _ = base_frame.shape
aligned_frames = [base_frame]

# 3. 取前 10 个最清晰的帧，全部扭正并对齐
top_indices = [x[0] for x in scores[1:10]]

for i in top_indices:
    cur = frames[i]
    kp_cur, des_cur = sift.detectAndCompute(cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY), None)
    matches = bf.knnMatch(des_base, des_cur, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) >= 20:
        pts_base = np.float32([kp_base[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_cur = np.float32([kp_cur[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(pts_cur, pts_base, cv2.RANSAC, 5.0)
        if H is not None:
            aligned_frames.append(cv2.warpPerspective(cur, H, (w, h)))

# 4. 多帧取中位数，输出降噪高清大图
fused = np.median(aligned_frames, axis=0).astype(np.uint8)

# 边界保护性裁切：削掉四周因透视拉伸留下的黑边与过渡带
pad = 30
h, w, _ = fused.shape
fused_clean = fused[pad:h-pad, pad:w-pad]

# 保存裁切后的纯净成片（注意写入的是 fused_clean）
cv2.imwrite("enhanced_result.jpg", fused_clean)
print(f"处理完成！融合了 {len(aligned_frames)} 帧画面，已生成 enhanced_result.jpg")
