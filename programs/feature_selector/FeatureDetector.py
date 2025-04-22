import cv2
import numpy as np
# from pylsd.lsd import lsd

class FeatureDetector:
    
    def __init__(self, use_harris=True,
                 max_corners=100, quality_level=0.01, min_distance=10,
                 canny_thresh1=50, canny_thresh2=200,
                 hough_thresh=80, min_line_length=30, max_line_gap=10):

        # 特徴点検出のパラメータ
        self.use_harris = use_harris
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance

        # エッジ検出 & ハフ変換のパラメータ
        self.canny_thresh1 = canny_thresh1
        self.canny_thresh2 = canny_thresh2
        self.hough_thresh = hough_thresh
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
    
    @classmethod
    def from_args(cls, args):
        return cls(
            max_corners=args.max_corners,
            quality_level=args.quality_level,
            min_distance=args.min_distance,
            canny_thresh1=args.canny_thresh1,
            canny_thresh2=args.canny_thresh2,
            hough_thresh=args.hough_thresh,
            min_line_length=args.min_line_length,
            max_line_gap=args.max_line_gap
        )

    def detect_corners(self, gray_img):
        """特徴点の検出"""
        corners = cv2.goodFeaturesToTrack(
            gray_img,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            useHarrisDetector=self.use_harris,
            k=0.04 if self.use_harris else 0.0
        )
        return corners

    def detect_lines(self, gray_img):
        """特徴線分の検出"""
        edges = cv2.Canny(gray_img, self.canny_thresh1, self.canny_thresh2, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_thresh,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        return lines
    
    # def detect_lines_lsd(self, gray_img):
    #     lines = []
    #     linesL = lsd(gray_img)
    #     for line in linesL:
    #         x1, y1, x2, y2 = map(int,line[:4])
    #         lines.append([x1, y1, x2, y2])

    #     return np.array(lines).reshape(-1, 1, 4)

    def draw_features(self, img, corners, lines):
        """特徴点と線分を描画"""
        if corners is not None:
            for pt in corners:
                x, y = pt.ravel()
                cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)  # 赤い点

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 緑の線

        return img

    def process_image(self, image_path, output_path):
        """1枚の画像を処理して保存"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners = self.detect_corners(gray)
        lines = self.detect_lines(gray)
        # lines = self.detect_lines_lsd(gray)

        result_img = self.draw_features(img.copy(), corners, lines)

        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result_img)
