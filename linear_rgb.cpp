#include "linear_rgb.h"

Lrgb_vr calc_lrgb_vr(Mat &image, Mat &fg_mask, Mat &bg_mask, int num_bins, 
		double *weights, int num_weights)
{
	Mat lrgb;
	Mat lrgb_hist_fg;
	Mat lrgb_hist_bg;
	int n_obj = countNonZero(fg_mask);
	int n_bg = countNonZero(bg_mask);
	double delta = 0.001;
	lrgb = calc_lrgb(image, weights, num_weights, num_bins - 1);
	/////////////////////////////////////
	//namedWindow("linear RGB.");
	//imshow("linear RGB.", lrgb / num_bins);
	//waitKey(100);
	//////////////////////////////////////
	// calculate histograms
	lrgb_hist_fg = calc_mask_1d_hist_range(lrgb, fg_mask, 0, num_bins, num_bins, true);
	lrgb_hist_bg = calc_mask_1d_hist_range(lrgb, bg_mask, 0, num_bins, num_bins, true);
	// do the math
	// p, q, L, v
	Mat p, q, p_delta, q_delta, poq, L;
	lrgb_hist_fg.convertTo(p, CV_32FC1);
	p /= n_obj;
	lrgb_hist_bg.convertTo(q, CV_32FC1);
	q /= n_bg;
	max(p, delta, p_delta);
	max(q, delta, q_delta);
	divide(p_delta, q_delta, poq);
	log(poq, L);
	p_delta.release();
	q_delta.release();
	poq.release();
	////////////////////////////////////
	//cout<<L.type()<<" "<<CV_64FC1<<endl;
	//namedWindow("L");
	//imshow("L", L);
	//waitKey(0);
	////////////////////////////////////
	////////////////////////////////////
	//Mat bp;
	//int bp_channels[] = {0};
	//float bp_channel_ranges[] = {0, num_bins};
	//const float* bp_ranges[] = {bp_channel_ranges};
	//calcBackProject(&lrgb, 1, bp_channels, L, bp, bp_ranges, 1, true);
	//namedWindow("Back projection");
	//imshow("Back projection", bp);
	//waitKey(0);
	//bp.release();
	////////////////////////////////////
	double vr = 0;
	Mat mean_pq;
	mean_pq = (p + q) / 2;
	vr = weighted_var(L, mean_pq) / (weighted_var(L, p) + weighted_var(L, q));
	//cout<<"VR: "<<vr<<endl;
	mean_pq.release();
	p.release();
	q.release();
	Lrgb_vr g;
	g.w_r = weights[2];
	g.w_g = weights[1];
	g.w_b = weights[0];
	g.vr = vr;
	g.num_bins = num_bins;
	L.copyTo(g.L);
	L.release();
	lrgb.release();
	lrgb_hist_bg.release();
	lrgb_hist_fg.release();
	lrgb.release();
	return g;
}

Mat calc_lrgb(Mat &image, double *weights, int num_weights, double scale)
{
	Mat lrgb = Mat::zeros(image.size(), CV_32FC1);
	for(int row = 0; row < image.rows; ++row) {
		Vec3b *image_row = image.ptr<Vec3b>(row);
		float *lrgb_row = lrgb.ptr<float>(row);
		for(int col = 0; col < image.cols; ++col) {
			for(int i = 0; i < num_weights; ++i) {
				lrgb_row[col] += weights[i] * image_row[col][i];
			}
		}
	}
	// may normalize
	normalize(lrgb, lrgb, scale, 0, NORM_MINMAX);
	return lrgb;
}

/*
 * Calculate weighted variance of input.
 * Assume both weights and input are CV_32FC1.
 * Equation according to Eq. 5 in "On-Line Selection of Discriminative Tracking Features".
 */
double weighted_var(Mat &input, Mat &weights)
{
	double v = 0;
	double left = 0;
	double right = 0;
	double tmp = 0;
	for(int row = 0; row < input.rows; ++row) {
		float *input_row = input.ptr<float>(row);
		float *weights_row = weights.ptr<float>(row);
		for(int col = 0; col < input.cols; ++col) {
			tmp = weights_row[col] * input_row[col];
			left += tmp * input_row[col];
			right += tmp;
		}
	}
	v = left - right * right;
	return v;
}

Mat calc_mask_1d_hist_range(const Mat &img, const Mat &mask, int channel, int bins, float range, bool no_normalize)
{
	Mat hist;
	int channels[] = {channel};
	int histSize[] = {bins};
	float channel_ranges[] = {0, range};
	const float* ranges[] = {channel_ranges};
	calcHist(&img, 1, channels, mask, hist, 1, histSize, ranges, true, false);

	// normalization
	//float num_pixels = 1;
	//if(mask.empty()) {
	//	num_pixels = (float) img.rows * img.cols;
	//} else {
	//	num_pixels = countNonZero(mask);
	//}
	if(!no_normalize) {
		normalize(hist, hist, 1, 0, NORM_L1);
	} else {
		// pass
	}
	return hist;
}

/*
 * Large vr goes first.
 */
bool compare_lrgb_vr(Lrgb_vr a, Lrgb_vr b)
{
	return (a.vr > b.vr);
}

Mat calc_backproject_lrgb_vr(Mat &image, Lrgb_vr g)
{
	Mat lrgb;
	double *ws = new double[3];
	ws[0] = g.w_b;
	ws[1] = g.w_g;
	ws[2] = g.w_r;
	lrgb = calc_lrgb(image, ws, 3, g.num_bins - 1);
	Mat bp;
	int bp_channels[] = {0};
	float bp_channel_ranges[] = {0, g.num_bins};
	const float* bp_ranges[] = {bp_channel_ranges};
	calcBackProject(&lrgb, 1, bp_channels, g.L, bp, bp_ranges, 1, true);
	normalize(bp, bp, 1, 0, NORM_MINMAX);
	delete [] ws;
	lrgb.release();
	return bp;
}
