import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# 事前学習済みResNetをロード
resnet = models.resnet50(pretrained=True)
resnet.eval()  # 評価モードに設定

# 特徴量抽出用に最終分類層を除外
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])

# 前処理の定義
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 入力サイズにリサイズ
    transforms.ToTensor(),          # テンソルに変換
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
])

# 画像の前処理
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # バッチ次元を追加
    return image


# 特徴量の抽出
def extract_features(image_path, model):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():  # 勾配計算を無効化
        features = model(image_tensor)
    return features.squeeze().numpy()  # ベクトルをフラット化


def feature_matching(target_image, db_image, detector_type='AKAZE', ratio=0.75, resize_scale=1.0):
    # 解像度の変更（指定されたスケールでリサイズ）
    if resize_scale != 1.0:
        target_image = cv2.resize(target_image, (0, 0), fx=resize_scale, fy=resize_scale)
        db_image = cv2.resize(db_image, (0, 0), fx=resize_scale, fy=resize_scale)
    
    # 画像の前処理（グレースケール変換）
    target_image_grey = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    db_image_grey = cv2.cvtColor(db_image, cv2.COLOR_BGR2GRAY)

    # 特徴点検出器の選択
    if detector_type == 'SIFT':
        detector = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
        index_params = dict(algorithm = 0, trees = 1)  # KDTreeを使用
    elif detector_type == 'AKAZE':
        detector = cv2.AKAZE_create(descriptor_size=0, threshold=0.001)
        index_params= dict(algorithm = 6, table_number = 6, key_size = 12, multi_probe_level = 1) # LSHを使用
    elif detector_type == 'ORB':
        detector = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
        index_params= dict(algorithm = 6, table_number = 6, key_size = 12, multi_probe_level = 1) # LSHを使用
    else:
        raise ValueError("Unknown detector type")

    keypoints1, descriptors1 = detector.detectAndCompute(target_image_grey, None)
    keypoints2, descriptors2 = detector.detectAndCompute(db_image_grey, None)
    
    if descriptors1 is None or descriptors2 is None:
        return 0  # マッチがない場合
    
    # ディスクリプタの型変換（SIFTの場合はfloat32、AKAZE・ORBの場合はuint8）
    if detector_type == 'SIFT':
        descriptors1 = descriptors1.astype(np.float32)
        descriptors2 = descriptors2.astype(np.float32)
    else:
        descriptors1 = descriptors1.astype(np.uint8)
        descriptors2 = descriptors2.astype(np.uint8)
    
    # FLANNベースのマッチングを使用
    search_params = dict(checks=50)  # より多くのチェックを行う
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 良いマッチのみを選択（Ratio Test）
    good_matches = []
    for match in matches:
        if len(match) == 2:  # 2つのマッチが見つかった場合のみ処理
            m, n = match
            if m.distance < ratio * n.distance:
                good_matches.append(m)

    # 上位マッチを選択
    max_matches = 100
    top_n_matches = sorted(good_matches, key=lambda x: x.distance)[:max_matches]

    # マッチした特徴点の座標をリストに格納
    target_coords = [keypoints1[match.queryIdx].pt for match in top_n_matches]
    db_coords = [keypoints2[match.trainIdx].pt for match in top_n_matches]
    
    # マッチした特徴点を画像に描画
    result_image = cv2.drawMatches(target_image, keypoints1, db_image, keypoints2, top_n_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchesThickness=3)

    return len(good_matches), result_image, target_coords, db_coords


def show_results(title, target_image_path, sorted_paths, sorted_scores):
    fig, axes = plt.subplots(1, len(sorted_paths) + 1, figsize=(15, 5))

    # 対象画像を表示
    target_image = Image.open(target_image_path).convert("RGB")
    axes[0].imshow(target_image)
    axes[0].set_title("Target Image")
    axes[0].axis("off")

    # 類似画像を表示
    for i, (image_path, score) in enumerate(zip(sorted_paths, sorted_scores)):
        similar_image = Image.open(image_path).convert("RGB")
        axes[i + 1].imshow(similar_image)
        axes[i + 1].set_title(f"{title}\nScore: {score:.4f}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()


def ResNettoAKAZE(target_image_path, database_paths, database_features, resnet):
    print("Extracting features from the target image...")
    target_features = extract_features(target_image_path, resnet).astype('float32').reshape(1, -1)

    # coarse-to-fineな手法
    print("Searching for similar images...")
    similarity_scores = cosine_similarity(target_features, database_features)[0]
    top_k_indices = np.argsort(similarity_scores)[::-1][:5]  # 類似度の高い順に上位5件を取得

    match_scores = []
    combined_scores = []
    result_images = []
    target_list = []
    db_list = []
    target_image_cv = cv2.imread(target_image_path)

    for idx in top_k_indices:
        db_image_cv = cv2.imread(database_paths[idx])
        match_count, result_image, target, db = feature_matching(target_image_cv, db_image_cv)
        match_score = match_count / 100
        match_scores.append(match_score)
        result_images.append(result_image)
        target_list.append(target)
        db_list.append(db)
        combined_score = similarity_scores[idx] * 0.5 + match_score * 0.5
        combined_scores.append(combined_score)

    # 各スコアでのソート
    similarity_sorted_idx = np.argsort(np.array(similarity_scores[top_k_indices]))[::-1]
    match_sorted_idx = np.argsort(np.array(match_scores))[::-1]
    combined_sorted_idx = np.argsort(np.array(combined_scores))[::-1]

    # 各ランキングに基づく画像とスコアのリスト
    similarity_sorted_paths = [database_paths[top_k_indices[i]] for i in similarity_sorted_idx]
    similarity_sorted_scores = [similarity_scores[top_k_indices[i]] for i in similarity_sorted_idx]

    match_sorted_paths = [database_paths[top_k_indices[i]] for i in match_sorted_idx]
    match_sorted_scores = [match_scores[i] for i in match_sorted_idx]

    combined_sorted_paths = [database_paths[top_k_indices[i]] for i in combined_sorted_idx]
    combined_sorted_scores = [combined_scores[i] for i in combined_sorted_idx]

    # 各ランキングごとに画像を表示
    show_results("Similarity Score", target_image_path, similarity_sorted_paths, similarity_sorted_scores)
    show_results("Matching Score", target_image_path, match_sorted_paths, match_sorted_scores)
    show_results("Combined Score", target_image_path, combined_sorted_paths, combined_sorted_scores)

    # 最も混合スコアの高い画像を返す
    best_idx = combined_sorted_idx[0]
    best_result_image = result_images[best_idx]
    cv2.imwrite("topscore.png", best_result_image)

    best_image = database_paths[top_k_indices[best_idx]]
    best_target = np.array(target_list[best_idx])
    best_db = np.array(db_list[best_idx])
    
    return best_image, best_target, best_db
