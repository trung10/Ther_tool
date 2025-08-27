// main.cpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// ======= Tham số mặc định =======
static const int    DEF_ROWS = 12;        // hàng (vertical) của chessboard nội suy bởi findChessboardCorners
static const int    DEF_COLS = 16;        // cột (horizontal)
static const float  DEF_SQUARE_PX = 50.f; // kích thước 1 ô ở ảnh đích (px)
static const float  DEF_R_MIN = 1000.f;   // brute-force R từ ...
static const float  DEF_R_MAX = 10000.f;  // ... đến ...
static const float  DEF_R_STEP = 500.f;   // bước

// Tạo điểm 3D của chessboard trên mặt trụ bán kính R, ô có kích thước = 1.0 (đơn vị tuỳ ý)
static vector<Point3f> board_cylinder_points(int rows, int cols, float square_size_1, float R) {
    vector<Point3f> pts;
    pts.reserve(rows * cols);
    float cx = (cols - 1) / 2.0f; // tâm theo trục cột
    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < cols; i++) {
            float angle = (i - cx) * square_size_1 / R;
            float X = R * std::sin(angle);
            float Z = R * (1 - std::cos(angle));
            float Y = j * square_size_1; // trục dọc không cong
            pts.emplace_back(X, Y, Z);
        }
    }
    return pts;
}

// Tính lỗi chiếu trung bình (px)
static double mean_reproj_error(const vector<Point2f>& imgPts,
                                const vector<Point3f>& objPts,
                                const Mat& K, const Mat& D,
                                Mat& rvec, Mat& tvec)
{
    vector<Point2f> proj;
    projectPoints(objPts, rvec, tvec, K, D, proj);
    double err = 0.0;
    for (size_t k = 0; k < proj.size(); ++k) {
        err += norm(proj[k] - imgPts[k]);
    }
    return err / (double)proj.size();
}

static void print_usage(const char* argv0) {
    cout <<
    "Usage:\n"
    "  " << argv0 << " <input_image> [output_image] [rows cols square_px R_min R_max R_step]\n\n"
    "Defaults:\n"
    "  rows=" << DEF_ROWS << ", cols=" << DEF_COLS << ", square_px=" << DEF_SQUARE_PX << ", R in [" << DEF_R_MIN << "," << DEF_R_MAX << "] step " << DEF_R_STEP << "\n"
    "Example:\n"
    "  " << argv0 << " chess_led.jpg unwrapped.png 12 16 50 1000 10000 500\n";
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    string inPath  = argv[1];
    string outPath = (argc >= 3 ? argv[2] : "unwrapped.png");

    int rows = (argc >= 4 ? atoi(argv[3]) : DEF_ROWS);
    int cols = (argc >= 5 ? atoi(argv[4]) : DEF_COLS);
    float squarePX = (argc >= 6 ? (float)atof(argv[5]) : DEF_SQUARE_PX);
    float Rmin = (argc >= 7 ? (float)atof(argv[6]) : DEF_R_MIN);
    float Rmax = (argc >= 8 ? (float)atof(argv[7]) : DEF_R_MAX);
    float Rstep= (argc >= 9 ? (float)atof(argv[8]) : DEF_R_STEP);

    Mat img = imread(inPath);
    if (img.empty()) {
        cerr << "ERROR: Cannot read image: " << inPath << "\n";
        return 2;
    }

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // --- Tìm góc bàn cờ ---
    vector<Point2f> corners;
    int flags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK;
    bool found = findChessboardCorners(gray, Size(cols, rows), corners, flags);

    if (!found) {
        cerr << "ERROR: Chessboard not found. Check rows/cols or image quality.\n";
        return 3;
    }

    cornerSubPix(gray, corners, Size(11,11), Size(-1,-1),
        TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 40, 0.01));

    // --- Camera matrix tạm thời (xấp xỉ) ---
    int w = img.cols, h = img.rows;
    Mat K = (Mat_<double>(3,3) << w, 0, w/2.0, 0, w, h/2.0, 0, 0, 1);
    Mat D = Mat::zeros(5,1, CV_64F);

    // --- Brute-force R để giảm reprojection error ---
    double bestErr = 1e18;
    float bestR = Rmin;
    Mat bestRvec, bestTvec;

    cout << "Brute-forcing R in [" << Rmin << ", " << Rmax << "] step " << Rstep << " ...\n";
    for (float R = Rmin; R <= Rmax + 1e-6f; R += Rstep) {
        vector<Point3f> obj3d = board_cylinder_points(rows, cols, 1.0f, R);

        Mat rvec, tvec;
        bool ok = solvePnP(obj3d, corners, K, D, rvec, tvec, false, SOLVEPNP_ITERATIVE);
        if (!ok) continue;

        double err = mean_reproj_error(corners, obj3d, K, D, rvec, tvec);

        // In tiến độ (tuỳ thích)
        // cout << "R=" << R << " -> mean error = " << err << " px\n";

        if (err < bestErr) {
            bestErr = err;
            bestR = R;
            bestRvec = rvec.clone();
            bestTvec = tvec.clone();
        }
    }

    if (bestErr >= 1e17) {
        cerr << "ERROR: PnP failed on all R values.\n";
        return 4;
    }

    cout << fixed;
    cout << "Best R = " << bestR << " (mean reprojection error ~ " << bestErr << " px)\n";

    // (Tuỳ chọn) Tối ưu tinh thêm quanh bestR với bước nhỏ hơn
    {
        float fineStep = std::max(10.0f, Rstep * 0.1f);
        float Rf_min = std::max(bestR - Rstep, Rmin);
        float Rf_max = std::min(bestR + Rstep, Rmax);

        for (float R = Rf_min; R <= Rf_max + 1e-6f; R += fineStep) {
            vector<Point3f> obj3d = board_cylinder_points(rows, cols, 1.0f, R);
            Mat rvec, tvec;
            bool ok = solvePnP(obj3d, corners, K, D, rvec, tvec, false, SOLVEPNP_ITERATIVE);
            if (!ok) continue;
            double err = mean_reproj_error(corners, obj3d, K, D, rvec, tvec);
            if (err < bestErr) {
                bestErr = err;
                bestR = R;
                bestRvec = rvec.clone();
                bestTvec = tvec.clone();
            }
        }
        cout << "Refined R = " << bestR << " (mean error ~ " << bestErr << " px)\n";
    }

    // --- Unwrap nhanh bằng Homography 2D (xấp xỉ phẳng) ---
    // Lưới phẳng đích: mỗi ô squarePX pixel
    vector<Point2f> dstGrid;
    dstGrid.reserve(rows*cols);
    for (int j = 0; j < rows; ++j)
        for (int i = 0; i < cols; ++i)
            dstGrid.emplace_back(i * squarePX, j * squarePX);

    Mat Hh = findHomography(corners, dstGrid, RANSAC);
    if (Hh.empty()) {
        cerr << "ERROR: findHomography failed.\n";
        return 5;
    }

    Mat unwrapped;
    warpPerspective(img, unwrapped, Hh, Size((int)std::round(cols * squarePX),
                                            (int)std::round(rows * squarePX)));

    // Lưu kết quả
    if (!imwrite(outPath, unwrapped)) {
        cerr << "ERROR: cannot write output image to " << outPath << "\n";
        return 6;
    }

    // (Tuỳ chọn) lưu ảnh debug corners
    Mat dbg = img.clone();
    drawChessboardCorners(dbg, Size(cols, rows), corners, true);
    imwrite("corners_debug.png", dbg);

    cout << "Saved: " << outPath << "\n";
    cout << "Saved debug corners: corners_debug.png\n";

    // Hiển thị (nếu có GUI)
    try {
        imshow("Input", img);
        imshow("Unwrapped", unwrapped);
        waitKey(0);
    } catch (...) {
        // bỏ qua nếu không có GUI
    }

    return 0;
}

