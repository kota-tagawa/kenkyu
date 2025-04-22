import cv2
import numpy as np
import os
import warnings

class FeatureSelector:
    def __init__(self, image, output_point_path, output_line_path, detector, window_max_size, shift: int):
        # パスの設定
        self.output_point_path = output_point_path
        self.output_line_path = output_line_path

        self.detector = detector
        self.window_name = "Select Features (click to select, ESC to save)"
        self.shift = shift # 原点を画像中心とするかどうか

        self.img = image
        self.resized_img, self.scale = self.resize_to_fit_screen(window_max_size) # 画像サイズの調整
        self.display_img = self.resized_img.copy() # 選択した点の描画用画像
        self.selected_points = [] # 選択した点

        #
        # 検出器を使用する場合
        #
        if self.detector:
            self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

            # 特徴点と特徴線分の検出
            self.corners = self.detector.detect_corners(self.gray)
            if self.corners is None:
                self.corners = np.empty((0, 1, 2), dtype=np.float32) # エラー回避
            self.lines = self.detector.detect_lines(self.gray)
            if self.lines is None:
                self.lines = np.empty((0, 1, 4), dtype=np.float32) # エラー回避

            # スケール調整
            self.corners = (self.corners.astype(float) * self.scale).astype(int)
            self.lines = (self.lines.astype(float) * self.scale).astype(int)

            # 描画用の画像
            self.point_display_img = self.resized_img.copy()
            self.line_display_img = self.resized_img.copy()

            # 選択した特徴線分
            self.selected_lines = []


    #
    # 指定した解像度に合うように画像サイズを変更
    # (縮小のみで拡大はしない)
    #
    def resize_to_fit_screen(self, window_max_size):
        img_height, img_width = self.img.shape[:2]
        scale_width = window_max_size / img_width
        scale_height = window_max_size / img_height
        scale = min(scale_width, scale_height, 1.0)
        if scale > 1.0:
            warnings.warn("ウィンドウサイズには入力画像サイズ以下の値を指定してください。画像は拡大されません。")
            scale = 1.0
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        resized_img = cv2.resize(self.img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        return resized_img, scale

    #
    # 検出した特徴点からマウスの左クリックで特徴点を選択
    #
    def point_mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.corners) > 0:
            # 最も近い特徴点を探す
            distances = np.linalg.norm(self.corners.reshape(-1, 2) - np.array([x, y]), axis=1)
            nearest_idx = np.argmin(distances)
            nearest_point = tuple(self.corners[nearest_idx][0])

            # 重複チェック（同じ点を2度追加しない）
            if nearest_point not in self.selected_points:
                self.selected_points.append(nearest_point)
                cv2.circle(self.point_display_img, (int(nearest_point[0]), int(nearest_point[1])), 5, (0, 255, 0), -1)
    
    #
    # 検出した特徴線分からマウスの左クリックで特徴線分を選択
    #
    def line_mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.lines) > 0:
            click_point = np.array([x, y], dtype=np.float32)
            min_distance = float('inf')
            nearest_line = None

            # 各線分とクリック位置の距離を計算
            for line in self.lines:
                x1, y1, x2, y2 = line[0]
                pt1 = np.array([x1, y1], dtype=np.float32)
                pt2 = np.array([x2, y2], dtype=np.float32)

                # 線分と点の距離を計算
                line_vec = pt2 - pt1
                point_vec = click_point - pt1
                line_len_sq = np.dot(line_vec, line_vec)

                # 線分上の最近傍点を求める（射影ベクトル）
                t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
                projection = pt1 + t * line_vec

                distance = np.linalg.norm(click_point - projection)

                # 最小距離の線分を保持
                if distance < min_distance:
                    min_distance = distance
                    nearest_line = (x1, y1, x2, y2)

            # 重複チェック & 描画
            if nearest_line and nearest_line not in self.selected_lines:
                self.selected_lines.append(nearest_line)
                cv2.line(self.line_display_img,
                        (int(nearest_line[0]), int(nearest_line[1])),
                        (int(nearest_line[2]), int(nearest_line[3])),
                        (0, 255, 0), 2)

    #
    # 検出された特徴点の描画と選択を行うウィンドウの表示
    # (dで最後の選択点を削除)
    #
    def point_run(self):
        # 赤色で全特徴点を表示
        for pt in self.corners:
            x, y = pt.ravel()
            cv2.circle(self.point_display_img, (int(x), int(y)), 3, (0, 0, 255), 1)

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.point_mouse_callback)

        while True:
            cv2.imshow(self.window_name, self.point_display_img)
            key = cv2.waitKey(1)

            # dで最後の選択点を削除
            if key == ord('d'):
                if self.selected_points:
                    self.selected_points.pop()

                    # 再描画
                    self.point_display_img = self.resized_img.copy()
                    for pt in self.corners:
                        x, y = pt.ravel()
                        cv2.circle(self.point_display_img, (int(x), int(y)), 3, (0, 0, 255), 1)
                    for pt in self.selected_points:
                        cv2.circle(self.point_display_img, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
                    print("Last selected point removed.")
                else:
                    print("No points to remove.")

            if key == 27:  # ESCで終了
                break

        cv2.destroyAllWindows()
        self.save_selected_points()
    
    #
    # 検出された特徴線分の描画と選択を行うウィンドウの表示
    # (dで最後の選択線を削除)
    #
    def line_run(self):
        # 緑色で全特徴線分を表示
        for line in self.lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(self.line_display_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 緑の線

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.line_mouse_callback)

        while True:
            cv2.imshow(self.window_name, self.line_display_img)
            key = cv2.waitKey(1)

            # dで最後の選択線を削除
            if key == ord('d'):
                if self.selected_lines:
                    self.selected_lines.pop()

                    # 再描画
                    self.line_display_img = self.resized_img.copy()
                    for line in self.lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(self.line_display_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    for line in self.selected_lines:
                        x1, y1, x2, y2 = line
                        cv2.line(self.line_display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    print("Last selected line removed.")
                else:
                    print("No lines to remove.")

            if key == 27:  # ESCで終了
                break

        cv2.destroyAllWindows()
        self.save_selected_lines()

    #
    # クリックした座標を取得
    #
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_points.append((x, y))
            cv2.circle(self.display_img, (x, y), 5, (0, 255, 0), -1)

    #
    # 点を選択するウィンドウを表示
    # (dで最後の点を削除)
    #
    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            cv2.imshow(self.window_name, self.display_img)
            key = cv2.waitKey(1)

            # dで最後の選択点を削除
            if key == ord('d'):
                if self.selected_points:
                    self.selected_points.pop()

                    # 再描画
                    self.display_img = self.resized_img.copy()
                    for pt in self.selected_points:
                        cv2.circle(self.display_img, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
                    print("Last selected point removed.")
                else:
                    print("No points to remove.")

            if key == 27:  # ESCで終了
                break

        cv2.destroyAllWindows()
        self.save_selected_points()

    #
    # 選択した特徴点の保存
    #
    def save_selected_points(self):
        os.makedirs(os.path.dirname(self.output_point_path), exist_ok=True)

        # 元のスケールに変換
        selected_points = np.array(self.selected_points, dtype=float)
        selected_points /= self.scale

        # 画像中心を原点としたデータを保存
        if self.shift:
            height, width = self.img.shape[:2]
            cx, cy = width // 2, height // 2
            shifted_points = [(int(x - cx), int(y - cy)) for (x, y) in selected_points]
            np.savetxt(self.output_point_path, np.array(shifted_points), fmt='%d')
        else:
            np.savetxt(self.output_point_path, selected_points, fmt='%d')
        print(f"{len(self.selected_points)} point(s) saved to: {self.output_point_path}")

    #
    # 選択した特徴線分の保存
    #
    def save_selected_lines(self):
        os.makedirs(os.path.dirname(self.output_line_path), exist_ok=True)

        # 元のスケールに変換
        selected_lines = np.array(self.selected_lines, dtype=float)
        selected_lines /= self.scale

        # 画像中心を原点としたデータを保存
        if self.shift:
            height, width = self.img.shape[:2]
            cx, cy = width // 2, height // 2
            shifted_lines = [(int(x1 - cx), int(y1 - cy), int(x2 - cx), int(y2 - cy)) for (x1, y1, x2, y2) in selected_lines]
            np.savetxt(self.output_line_path, np.array(shifted_lines), fmt='%d')
        else:
            np.savetxt(self.output_line_path, selected_lines, fmt='%d')
        print(f"{len(self.selected_lines)} line(s) saved (as 2 points each) to: {self.output_line_path}")

