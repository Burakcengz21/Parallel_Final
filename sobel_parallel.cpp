#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>

using namespace cv;
using namespace std;

Mat loadImage(const string &imagePath) {
    Mat image = imread(imagePath, IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Error: Image cannot be loaded!" << endl;
        exit(1);
    }
    return image;
}

void sobelFilterParallel(const Mat &src, Mat &dst) {
    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int Gy[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };

    dst = Mat::zeros(src.size(), src.type());

    #pragma omp parallel for collapse(2)
    for (int y = 1; y < src.rows - 1; ++y) {
        for (int x = 1; x < src.cols - 1; ++x) {
            int sumX = 0, sumY = 0;
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    sumX += Gx[i + 1][j + 1] * src.at<uchar>(y + i, x + j);
                    sumY += Gy[i + 1][j + 1] * src.at<uchar>(y + i, x + j);
                }
            }
            int magnitude = sqrt(sumX * sumX + sumY * sumY);
            dst.at<uchar>(y, x) = saturate_cast<uchar>(magnitude);
        }
    }
}

int main() {
    string imagePath = "path_to_your_image.jpg";
    Mat image = loadImage(imagePath);

    Mat edgesParallel;

    double start = omp_get_wtime();
    sobelFilterParallel(image, edgesParallel);
    double end = omp_get_wtime();
    cout << "Parallel execution time: " << (end - start) << " seconds." << endl;

    imshow("Original Image", image);
    imshow("Edges (Parallel)", edgesParallel);

    waitKey(0);
    return 0;
}
