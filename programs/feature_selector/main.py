import cv2
import argparse
import FeatureDetector as fd
import FeatureSelector as fs

def main(args):
    img = cv2.imread(args.i)
    if img is None:
        raise FileNotFoundError("画像ファイルが見つかりません")

    # ウィンドウサイズの設定
    if args.wsize:
        window_max_size = args.wsize
    else:
        height, width = img.shape[:2]
        window_max_size = max(height, width)

    # 検出器を使用する場合
    if args.use_detector:
        if args.op or args.ol:
            detector = fd.FeatureDetector.from_args(args)
            selector = fs.FeatureSelector(img, args.op, args.ol, detector, window_max_size, args.shift)
            if args.op:
                selector.point_run()
            if args.ol:
                selector.line_run()
        else:
            parser.error("When a detector is used, at least one of --op or --ol must be specified.")

    # 検出器を使用しない場合
    else:
        if args.op:
            selector = fs.FeatureSelector(img, args.op, None, None, window_max_size, args.shift)
            selector.run()
        else:
            parser.error("When a detector is not used, --op must be specified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Select feature points and lines to save to a file.')

    parser.add_argument('-i', required=True, help='Input image path')
    parser.add_argument('--wsize', required=False, type=int, help='Window max size')
    parser.add_argument('--shift', action='store_true', required=False, help='Using the center of the image as the origin')
    parser.add_argument('--op', required=False, help='Output point file path')
    parser.add_argument('--ol', required=False, help='Output line file path')
    parser.add_argument('--use_detector', action='store_true', required=False, help='Detect features using a detector')
    # コーナー検出（goodFeaturesToTrack）
    parser.add_argument("--max_corners", type=int, required=False, default=100, help="Maximum number of corners to return (if there are more, strongest ones are returned)")
    parser.add_argument("--quality_level", type=float, required=False, default=0.01, help="Minimum accepted quality of image corners (0 to 1)")
    parser.add_argument("--min_distance", type=int, required=False, default=10, help="Minimum possible Euclidean distance between returned corners")
    # Cannyエッジ検出
    parser.add_argument("--canny_thresh1", type=int, required=False, default=50, help="First threshold for the Canny edge detector (lower threshold)")
    parser.add_argument("--canny_thresh2", type=int, required=False, default=200, help="Second threshold for the Canny edge detector (upper threshold)")
    # ハフ変換
    parser.add_argument("--hough_thresh", type=int, required=False, default=80, help="Accumulator threshold for the Hough line transform (higher means fewer lines)")
    parser.add_argument("--min_line_length", type=int, required=False, default=30, help="Minimum line length for HoughLinesP; shorter lines are rejected")
    parser.add_argument("--max_line_gap", type=int, required=False, default=10, help="Maximum allowed gap between line segments to treat them as a single line")
    
    args = parser.parse_args()

    main(args)