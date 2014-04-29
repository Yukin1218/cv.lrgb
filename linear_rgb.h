#ifndef LINEAR_RGB_H
#define LINEAR_RGB_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

typedef struct _lrgb_vr
{
	double w_r;
	double w_g;
	double w_b;
	double vr;
	int num_bins;
	Mat L;
} Lrgb_vr;

Lrgb_vr calc_lrgb_vr(Mat &image, Mat &fg_mask, Mat &bg_mask, int num_bins, 
	double *weights, int num_weights);
Mat calc_lrgb(Mat &image, double *weights, int num_weights, double scale);
double weighted_var(Mat &input, Mat &weights);
bool compare_lrgb_vr(Lrgb_vr a, Lrgb_vr b);
Mat calc_mask_1d_hist_range(const Mat &img, const Mat &mask, int channel, int bins, float range, bool no_normalize);
Mat calc_backproject_lrgb_vr(Mat &image, Lrgb_vr g);

#endif