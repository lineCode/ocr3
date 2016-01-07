#include"segment.h"
#include"sheet2Chars.h"
#include<iostream>

#define SHOWTEMPIMAGE //显示临时结果开关
//#undef _DEBUG
int segment(const char* filepath, vector<vector<cv::Mat>>& vvM)
{
	int res = 0;
	cv::Mat body;
	vector<cv::Mat> vmitems;
	vector<cv::Mat> vmROIitems;
	vector<cv::Mat> vmlines;
	res = cutBody(filepath, body);
	if (res != 0)
	{
		cerr << "cutBody error" << endl;
		return res;
	}
#ifdef _DEBUG
	cv::imshow("body", body);
	//cv::imwrite("./body.png", body);
	cv::waitKey();
#endif//_DEBUG

	res = analysisLayout(body, vmitems);
	if (res != 0)
	{
		cerr << "analysisLayout error" << endl;
		return res;
	}
#ifdef _DEBUG
	for (size_t it = 0; it < vmitems.size(); it++)
	{
		char name[64] = { 0 };
		sprintf(name, "vmitems_%d.png", it);
		cv::imshow(name, vmitems[it]);
		//cv::imwrite(name, vmitems[it]);
		cv::waitKey();
	}
#endif//_DEBUG
	res = getROIItems(vmitems, vmROIitems);
	if (res != 0)
	{
		cerr << "getROIItems error" << endl;
		return res;
	}
#ifdef _DEBUG
	for (size_t it = 0; it < vmROIitems.size(); it++)
	{
		cv::imshow("vmROIitems", vmROIitems[it]);
		cv::waitKey();
	}
#endif//_DEBUG
	//res = cutLines(vmROIitems, vmlines);
	//if (res != 0)
	//{
	//	cerr << "vmROIitems error" << endl;
	//	return res;
	//}
	res = segmentChars(vmROIitems, vvM);
	if (res != 0)
	{
		cerr << "segmentChars error" << endl;
		return res;
	}

#ifdef _DEBUG
	
	for (size_t it = 0; it < vvM.size(); it++)
	{
		char name[64] = { 0 };
		for(size_t it2=0; it2<vvM[it].size();it2++)
		{
			sprintf(name, "../Output/h2/vvM%04d%04d.png", it,it2);
			//cv::imshow(name, vvM[it][it2]);
			imwrite(name, vvM[it][it2]);
			//cv::waitKey();
		}
	}
#endif//_DEBUG

	return 0;
}