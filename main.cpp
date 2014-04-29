/*
 * Show the best 5 spaces for given image and foreground mask.
 */
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "linear_rgb.h"

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
	// read in weights
	int num_weights = 3;
	int num_weights_cands = 49;
	double *lrgb_weights = new double[num_weights * num_weights_cands];
	fstream l49;
	l49.open("l49.txt", fstream::in);
	int cl49 = 0;
	for(int i = 0; i < num_weights_cands; ++i) {
		l49>>*(lrgb_weights + cl49)>>*(lrgb_weights + cl49 + 1)>>*(lrgb_weights + cl49 + 2);
		cl49 += 3;
	}
	//for(int i = 0; i < num_weights * num_weights_cands; ++i) {
	//	cout<<*(lrgb_weights + i)<<" ";
	//}
	//cout<<endl;
	l49.close();

	// configurations
	int num_bins = 64;

	Mat image = imread(argv[1]);
	Mat fg_mask = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
	threshold(fg_mask, fg_mask, 10, 255, THRESH_BINARY);
	Mat bg_mask;
	bitwise_not(fg_mask, bg_mask);
	// for each weights configuration, calc lrgb, hist and var.
	vector<Lrgb_vr> ls;
	int n_obj = countNonZero(fg_mask);
	int n_bg = countNonZero(bg_mask);
	double delta = 0.001;
	for(int index = 0; index < num_weights_cands; ++index) {
		Lrgb_vr g = calc_lrgb_vr(image, fg_mask, bg_mask, num_bins, 
			lrgb_weights + index * num_weights, num_weights);
		ls.push_back(g);
	}
	// sort and show
	Mat lrgb;
	sort(ls.begin(), ls.end(), compare_lrgb_vr);
	for(int i = 0; i < 5; ++i) {
		Mat bp = calc_backproject_lrgb_vr(image, ls[i]);
		cout<<"VR: "<<ls[i].vr<<endl;
		namedWindow("Back projection");
		imshow("Back projection", bp);
		waitKey(0);
		bp.release();
	}
	lrgb.release();
	bg_mask.release();
	fg_mask.release();
	image.release();

	delete lrgb_weights;
	return 0;
}