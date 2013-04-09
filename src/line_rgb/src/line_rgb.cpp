/*#******************************************************************************
 ** IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 **
 ** By downloading, copying, installing or using the software you agree to this license.
 ** If you do not agree to this license, do not download, install,
 ** copy or use the software.
 **
 **
 ** LineRGB: this method allows OpenCV users to train and use a detector able to recognize in time-real, low-textured or textureless objects in videos or images. Presented model originate from Stefan Hinterstoisser's original Line2D method and from Willow Garage's implementation.
 **
 ** Use: create object templates from color and mask images and use them to train a detector. The trained detector will be able to find the objects in color images or videos.
 **
 **
 **  Creation - enhancement process 2012-2013
 **      Authors: Manuele Tamburrano (manuele.tamburrano@gmail.com), University La Sapienza di Roma, Rome, Italy
 **      	      Stefano Fabri (s.fabri@email.it), Rome, Italy
 **
 ** This algorithm has been developed by Manuele Tamburrano since his thesis with Maria de Marsico at University La Sapienza di Roma and the research he pursued with Stefano Fabri and Carlo Maria Medaglia at CATTID Lab.
 * 
 ** Refer to the following research paper for more information:
 ** ......
 ** 
 **                          License Agreement
 **               For Open Source Computer Vision Library
 **
 ** Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 ** Copyright (C) 2008-2011, Willow Garage Inc., all rights reserved.
 **
 **               For Human Visual System tools (hvstools)
 ** Copyright (C) 2007-2011, LISTIC Lab, Annecy le Vieux and GIPSA Lab, Grenoble, France, all rights reserved.
 **
 ** Third party copyrights are property of their respective owners.
 **
 ** Redistribution and use in source and binary forms, with or without modification,
 ** are permitted provided that the following conditions are met:
 **
 ** * Redistributions of source code must retain the above copyright notice,
 **    this list of conditions and the following disclaimer.
 **
 ** * Redistributions in binary form must reproduce the above copyright notice,
 **    this list of conditions and the following disclaimer in the documentation
 **    and/or other materials provided with the distribution.
 **
 ** * The name of the copyright holders may not be used to endorse or promote products
 **    derived from this software without specific prior written permission.
 **
 ** This software is provided by the copyright holders and contributors "as is" and
 ** any express or implied warranties, including, but not limited to, the implied
 ** warranties of merchantability and fitness for a particular purpose are disclaimed.
 ** In no event shall the Intel Corporation or contributors be liable for any direct,
 ** indirect, incidental, special, exemplary, or consequential damages
 ** (including, but not limited to, procurement of substitute goods or services;
 ** loss of use, data, or profits; or business interruption) however caused
 ** and on any theory of liability, whether in contract, strict liability,
 ** or tort (including negligence or otherwise) arising in any way out of
 ** the use of this software, even if advised of the possibility of such damage.
 *******************************************************************************/

/*
 * line_rgb.cpp
 *
 *  Created on: 2012
 *     Authors: Manuele Tamburrano (manuele.tamburrano@gmail.com), University La Sapienza di Roma, Rome, Italy
 *   	      Stefano Fabri (s.fabri@email.it), Rome, Italy
 */

#include "objdetect_line_rgb.hpp"
#include <limits>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/core_c.h"
#include "opencv2/core/internal.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iomanip>
#include <cmath>

//#include "opencv2/objdetect/objdetect_tegra.hpp"

namespace cv {
namespace line_rgb {

class Timer {
public:
    Timer() :
            start_(0), time_(0) {
    }

    void start() {
        start_ = cv::getTickCount();
    }

    void stop() {
        CV_Assert(start_ != 0);
        int64 end = cv::getTickCount();
        time_ += end - start_;
        start_ = 0;
    }

    double time() {
        double ret = time_ / cv::getTickFrequency();
        time_ = 0;
        return ret;
    }

private:
    int64 start_, time_;
};

// struct Feature

/**
 * \brief Get the label [0,8) of the single bit set in quantized.
 */
static inline int getLabel(int quantized) {
    switch (quantized) {
    case 1:
        return 0;
    case 2:
        return 1;
    case 4:
        return 2;
    case 8:
        return 3;
    case 16:
        return 4;
    case 32:
        return 5;
    case 64:
        return 6;
    case 128:
        return 7;
    default:
        CV_Assert(false);
        return -1; //avoid warning
    }
}

void Feature::read(const FileNode& fn) {
    FileNodeIterator fni = fn.begin();
    fni >> x >> y >> label >> rgb_label >> on_border;
}

void Feature::write(FileStorage& fs) const {
    fs << "[:" << x << y << label << rgb_label << on_border << "]";
}

// struct Template
/**
 * \brief Crop a set of overlapping templates from different modalities.
 *
 * \param[in,out] templates Set of templates representing the same object view.
 * \param[in] mask_template Mask of the object at higher pyramid level
 *
 * \return The bounding box of all the templates in original image coordinates.
 */
Rect cropTemplates(std::vector<Template>& templates, const Mat& mask_template) {

    int min_x = std::numeric_limits<int>::max();
    int min_y = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::min();
    int max_y = std::numeric_limits<int>::min();

    // First pass: find min/max feature x,y over all pyramid levels and modalities
    for (int i = 0; i < (int) templates.size(); ++i) {
        Template& templ = templates[i];

        for (int j = 0; j < (int) templ.features.size(); ++j) {
            int x = templ.features[j].x << templ.pyramid_level;
            int y = templ.features[j].y << templ.pyramid_level;
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
        }
    }

    /// @todo Why require even min_x, min_y?
    if (min_x % 2 == 1)
        --min_x;
    if (min_y % 2 == 1)
        --min_y;

    // Second pass: set width/height and shift all feature positions
    for (int i = 0; i < (int) templates.size(); ++i) {
        Template& templ = templates[i];
        templ.width = (max_x - min_x) >> templ.pyramid_level;
        templ.height = (max_y - min_y) >> templ.pyramid_level;

        Mat cropped_mask = mask_template(
                Rect(min_x, min_y, max_x - min_x, max_y - min_y));

        int offset_x = min_x >> templ.pyramid_level;
        int offset_y = min_y >> templ.pyramid_level;

        for (int j = 0; j < (int) templ.features.size(); ++j) {
            templ.features[j].x -= offset_x;
            templ.features[j].y -= offset_y;
        }

        for (int j = 0; j < (int) templ.color_features.size(); ++j) {
            templ.color_features[j].x -= offset_x;
            templ.color_features[j].y -= offset_y;
        }

        //extract the contour of the object from mask at higher pyramid level for the signature
        if (templ.pyramid_level == 0) {
            Mat tmp_mask;
            cropped_mask.copyTo(tmp_mask);

            vector < vector<Point> > contours;
            Mat gray;

            findContours(tmp_mask, contours, CV_RETR_EXTERNAL,
                    CV_CHAIN_APPROX_TC89_L1);
            templ.contour = contours[0];
        }

    }

    return Rect(min_x, min_y, max_x - min_x, max_y - min_y);
}

void Template::read(const FileNode& fn) {
    width = fn["width"];
    height = fn["height"];
    pyramid_level = fn["pyramid_level"];
    total_candidates = fn["total_candidates"];
    fn["contour"] >> contour;
    FileNode features_fn = fn["features"];
    features.resize(features_fn.size());
    FileNodeIterator it = features_fn.begin(), it_end = features_fn.end();
    for (int i = 0; it != it_end; ++it, ++i) {
        features[i].read(*it);
    }
    FileNode features_signature_fn = fn["color_features"];
    color_features.resize(features_signature_fn.size());

    FileNodeIterator its = features_signature_fn.begin(), its_end =
            features_signature_fn.end();
    for (int i = 0; its != its_end; ++its, ++i) {
        color_features[i].read(*its);
    }
}

void Template::write(FileStorage& fs) const {
    fs << "width" << width;
    fs << "height" << height;
    fs << "pyramid_level" << pyramid_level;
    fs << "total_candidates" << total_candidates;
    fs << "contour" << contour;
    fs << "features" << "[";
    for (int i = 0; i < (int) features.size(); ++i) {
        features[i].write(fs);
    }
    fs << "]"; // features
    fs << "color_features" << "[";
    for (int i = 0; i < (int) color_features.size(); ++i) {
        color_features[i].write(fs);
    }
    fs << "]"; // color_features
}

/****************************************************************************************\
*                             Modality interfaces                                        *
 \****************************************************************************************/

void QuantizedPyramid::selectScatteredFeatures(
        const std::vector<Candidate>& candidates,
        std::vector<Feature>& features, size_t num_features, float distance) {
    features.clear();
    float distance_sq = CV_SQR(distance);
    int i = 0;
    while (features.size() < num_features) {
        Candidate c = candidates[i];

        // Add if sufficient distance away from any previously chosen feature
        bool keep = true;
        for (int j = 0; (j < (int) features.size()) && keep; ++j) {
            Feature f = features[j];
            keep = CV_SQR(c.f.x - f.x) + CV_SQR(c.f.y - f.y) >= distance_sq;
        }
        if (keep)
            features.push_back(c.f);

        if (++i == (int) candidates.size()) {
            // Start back at beginning, and relax required distance
            i = 0;
            distance -= 1.0f;
            distance_sq = CV_SQR(distance);
        }
    }
}

Ptr<Modality> Modality::create(const std::string& modality_type) {
    if (modality_type == "ColorGradient")
        return new ColorGradient();
    else if (modality_type == "DepthNormal")
        return new DepthNormal();
    else
        return NULL;
}

Ptr<Modality> Modality::create(const FileNode& fn) {
    std::string type = fn["type"];
    Ptr < Modality > modality = create(type);
    modality->read(fn);
    return modality;
}

void colormap(const Mat& quantized, Mat& dst) {
    std::vector < Vec3b > lut(8);
    lut[0] = Vec3b(0, 0, 255);
    lut[1] = Vec3b(0, 170, 255);
    lut[2] = Vec3b(0, 255, 170);
    lut[3] = Vec3b(0, 255, 0);
    lut[4] = Vec3b(170, 255, 0);
    lut[5] = Vec3b(255, 170, 0);
    lut[6] = Vec3b(255, 0, 0);
    lut[7] = Vec3b(255, 0, 170);

    dst = Mat::zeros(quantized.size(), CV_8UC3);
    for (int r = 0; r < dst.rows; ++r) {
        const uchar* quant_r = quantized.ptr(r);
        Vec3b* dst_r = dst.ptr < Vec3b > (r);
        for (int c = 0; c < dst.cols; ++c) {
            uchar q = quant_r[c];
            if (q)
                dst_r[c] = lut[getLabel(q)];
        }
    }
}

/****************************************************************************************\
*                             Color gradient modality                                    *
 \****************************************************************************************/

// Forward declaration
void hysteresisGradient(Mat& magnitude, Mat& angle, Mat& ap_tmp,
        float w_threshold, float s_threshold, bool compute_magnitude_strong,
        Mat& magnitude_strong);

int *bar(int *array) {
    int *start = array;
    while (*array) {
        *array++ += 1;
    }
    return start;
}

void writeMat(Mat& m) {
    string fileNameConf = "./matrice_stampata.txt";
    FileStorage fsConf(fileNameConf, FileStorage::WRITE);
    fsConf << "m" << m;

    fsConf.release();
}

/** 
 * \brief returns if the values of 3 channels are similar to black or to white 
 * 255+255+255 = 765
 * <382 = black
 * >=382 = white
 */
uchar blackOrWhite(const unsigned char* & r_g_b) {
    ushort sum = r_g_b[0] + r_g_b[1] + r_g_b[2];
    if (sum < 382)
        return 128; //black
    else
        return 64; //white
}

// Color quantization table:			R  G  B
// index column = index_best		  \|1 |2 |4 |
// index row = 0 + channel>threshold  R|1 |8 |16|
//									  G|8 |2 |32|
//                                    B|16|32|4 |
static const unsigned char color_quantization_table[4][3] = { { 1, 2, 4 }, { 1,
        8, 16 }, { 8, 2, 32 }, { 16, 32, 4 } };

/** 
 * \brief returns the quantization based on the diffs among values of rgb channels
 * 00000001 = R = 1
 * 00000010 = G = 2
 * 00000100 = B = 4
 * 00001000 = RG = 8
 * 00010000 = RB = 16
 * 00100000 = GB = 32
 * 01000000 = RGB_WHITE = 64
 * 10000000 = RGB_BLACK = 128
 * 
 * \param[in]  r_g_b       Array of the 3 values of RGB channels
 * \param[in]  index_best  Index of the channel with highest color value
 * \param[in]  threshold   If the difference of the color values is below this threshold, they are considered similar
 * 
 * \return the color quantization computed.
 * **/
unsigned char get_RGB_quantization(const unsigned char* r_g_b,
        unsigned short index_best, int threshold) {
    unsigned char response = -1;
    int threshold_added = r_g_b[index_best] - threshold;

    if (threshold_added <= 0)
        threshold_added = 0;

    unsigned short secondary_channel_1 = (index_best + 1) % 3;
    unsigned short secondary_channel_2 = (index_best + 2) % 3;
    unsigned char column = index_best;
    unsigned char row = 0;

    //if all the channels are above the threshold, the response is black or white
    if (r_g_b[secondary_channel_1] >= threshold_added
            && r_g_b[secondary_channel_2] >= threshold_added)
        response = blackOrWhite(r_g_b);
    else //else set the right row offset and the relative response
    {
        if (r_g_b[secondary_channel_1] >= threshold_added)
            row = secondary_channel_1 + 1;
        if (r_g_b[secondary_channel_2] >= threshold_added)
            row = secondary_channel_2 + 1;

        response = color_quantization_table[row][column];
    }

    return response;
}

/**
 * \brief Compute quantized orientation image from color image.
 *
 * Implements section 2.2 "Computing the Gradient Orientations."
 *
 * \param[in]  src       				The source 8-bit, 3-channel image.
 * \param[out] magnitude 				Destination floating-point array of squared magnitudes.
 * \param[out] angle     				Destination 8-bit array of orientations. Each bit
 *                       					represents one bin of the orientation space.
 * \param[out] rgb       				Destination 8-bit array of colors. Each bit
 *                       					represents one bin of the color space.
 * \param threshold 					Magnitude threshold. Keep only gradients whose norms are
 *             					        	larger than this.
 * \param[in]  compute_magnitude_strong Boolean specifies if the magnitude_strong array must be filled
 * \param[out] magnitude_strong         Destination floating-point array of higher squared magnitudes.
 * 
 */
void quantizedOrientations(const Mat& src, Mat& magnitude, Mat& angle, Mat& rgb,
        float w_threshold, float s_threshold, float threshold_rgb,
        bool compute_magnitude_strong, Mat& magnitude_strong) {

    magnitude.create(src.size(), CV_32F);
    rgb.create(src.size(), CV_8UC1);

    //if we are quantizing the orientations and colors to the higher pyramid level, we initialize the mat to store higher magnitudes in the source for further analysis (signature)
    if (compute_magnitude_strong == true) {
        magnitude_strong.create(src.size(), CV_32F);
        magnitude_strong.setTo(0);
    }
    // Allocate temporary buffers
    Size size = src.size();

    Mat sobel_3dx; // per-channel horizontal derivative
    Mat sobel_3dy; // per-channel vertical derivative
    Mat sobel_dx(size, CV_32F);      // maximum horizontal derivative
    Mat sobel_dy(size, CV_32F);      // maximum vertical derivative
    Mat sobel_ag;  // final gradient orientation (unquantized)
    Mat smoothed;

    // Compute horizontal and vertical image derivatives on all color channels separately
    static const int KERNEL_SIZE = 3;

    // For some reason cvSmooth/cv::GaussianBlur, cvSobel/cv::Sobel have different defaults for border handling...
    GaussianBlur(src, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0,
            BORDER_REPLICATE);
    Sobel(smoothed, sobel_3dx, CV_16S, 1, 0, KERNEL_SIZE/*CV_SCHARR*/, 1.0, 0.0,
            BORDER_REPLICATE);
    Sobel(smoothed, sobel_3dy, CV_16S, 0, 1, KERNEL_SIZE/*CV_SCHARR*/, 1.0, 0.0,
            BORDER_REPLICATE);

    short * ptrx = (short *) sobel_3dx.data;
    short * ptry = (short *) sobel_3dy.data;
    float * ptr0x = (float *) sobel_dx.data;
    float * ptr0y = (float *) sobel_dy.data;
    float * ptrmg = (float *) magnitude.data;

    uchar * ptrrgb = (uchar *) rgb.data;
    uchar * ptrsrc = (uchar *) src.data;

    const int length0 = sobel_3dy.cols * 3;
    const int length1 = static_cast<const int>(sobel_3dx.step1());
    const int length2 = static_cast<const int>(sobel_3dy.step1());
    const int length3 = static_cast<const int>(sobel_dx.step1());
    const int length4 = static_cast<const int>(sobel_dy.step1());
    const int length5 = static_cast<const int>(magnitude.step1());
    const int length6 = static_cast<const int>(rgb.step1());
    const int length7 = static_cast<const int>(src.step1());

    int max = 0;
    int min = 0;

    for (int r = 0; r < sobel_3dy.rows; ++r) {
        int ind = 0;

        for (int i = 0; i < length0; i += 3) {

            // Use the gradient orientation of the channel whose magnitude is largest
            int mag1 = CV_SQR(ptrx[i]) + CV_SQR(ptry[i]);
            int mag2 = CV_SQR(ptrx[i + 1]) + CV_SQR(ptry[i + 1]);
            int mag3 = CV_SQR(ptrx[i + 2]) + CV_SQR(ptry[i + 2]);

            if (mag1 >= mag2 && mag1 >= mag3) {
                ptr0x[ind] = ptrx[i];
                ptr0y[ind] = ptry[i];
                ptrmg[ind] = (float) mag1;
            } else if (mag2 >= mag1 && mag2 >= mag3) {
                ptr0x[ind] = ptrx[i + 1];
                ptr0y[ind] = ptry[i + 1];
                ptrmg[ind] = (float) mag2;

            } else {
                ptr0x[ind] = ptrx[i + 2];
                ptr0y[ind] = ptry[i + 2];
                ptrmg[ind] = (float) mag3;

            }

            unsigned char red = ptrsrc[i + 2];
            unsigned char green = ptrsrc[i + 1];
            unsigned char blue = ptrsrc[i];

            const unsigned char r_g_b[3] = { red, green, blue };
            if (red >= green && red >= blue)
                ptrrgb[ind] = get_RGB_quantization(r_g_b, 0, threshold_rgb);
            else if (green >= red && green >= blue)
                ptrrgb[ind] = get_RGB_quantization(r_g_b, 1, threshold_rgb);
            else
                ptrrgb[ind] = get_RGB_quantization(r_g_b, 2, threshold_rgb);

            ++ind;
        }
        ptrx += length1;
        ptry += length2;
        ptr0x += length3;
        ptr0y += length4;
        ptrmg += length5;
        ptrrgb += length6;
        ptrsrc += length7;
    }

    // Calculate the final gradient orientations
    phase(sobel_dx, sobel_dy, sobel_ag, true);
    hysteresisGradient(magnitude, angle, sobel_ag, CV_SQR(w_threshold),
            CV_SQR(s_threshold), compute_magnitude_strong, magnitude_strong);

}

void hysteresisGradient(Mat& magnitude, Mat& quantized_angle, Mat& angle,
        float w_threshold, float s_threshold, bool compute_magnitude_strong,
        Mat& magnitude_strong) {

    // Quantize 360 degree range of orientations into 16 buckets
    // Note that [0, 11.25), [348.75, 360) both get mapped in the end to label 0,
    // for stability of horizontal and vertical features.
    Mat_<unsigned char> quantized_unfiltered;
    angle.convertTo(quantized_unfiltered, CV_8U, 16.0 / 360.0);

    // Zero out top and bottom rows
    /// @todo is this necessary, or even correct?
    memset(quantized_unfiltered.ptr(), 0, quantized_unfiltered.cols);
    memset(quantized_unfiltered.ptr(quantized_unfiltered.rows - 1), 0,
            quantized_unfiltered.cols);
    // Zero out first and last columns
    for (int r = 0; r < quantized_unfiltered.rows; ++r) {
        quantized_unfiltered(r, 0) = 0;
        quantized_unfiltered(r, quantized_unfiltered.cols - 1) = 0;
    }

    // Mask 16 buckets into 8 quantized orientations
    for (int r = 1; r < angle.rows - 1; ++r) {
        uchar* quant_r = quantized_unfiltered.ptr < uchar > (r);
        for (int c = 1; c < angle.cols - 1; ++c) {
            quant_r[c] &= 7;
        }
    }

    // Filter the raw quantized image. Only accept pixels where the magnitude is above some
    // threshold, and there is local agreement on the quantization.
    quantized_angle = Mat::zeros(angle.size(), CV_8U);
    for (int r = 1; r < angle.rows - 1; ++r) {
        float* mag_r = magnitude.ptr<float>(r);

        for (int c = 1; c < angle.cols - 1; ++c) {

            if (mag_r[c] > w_threshold) {
                // Compute histogram of quantized bins in 3x3 patch around pixel
                int histogram[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

                uchar* patch3x3_row = &quantized_unfiltered(r - 1, c - 1);
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                patch3x3_row += quantized_unfiltered.step1();
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                patch3x3_row += quantized_unfiltered.step1();
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                // Find bin with the most votes from the patch
                int max_votes = 0;
                int index = -1;
                for (int i = 0; i < 8; ++i) {
                    if (max_votes < histogram[i]) {
                        index = i;
                        max_votes = histogram[i];
                    }
                }
                //store all pixels at higher pyramid level with magnitude > threshold
                if (compute_magnitude_strong == true
                        && mag_r[c] > s_threshold) {
                    magnitude_strong.at<float>(r, c) = mag_r[c];
                }

                // Only accept the quantization if majority of pixels in the patch agree
                static const int NEIGHBOR_THRESHOLD = 5;
                if (max_votes >= NEIGHBOR_THRESHOLD) {
                    quantized_angle.at < uchar > (r, c) = 1 << index;

                }
            }
        }
    }

}

class ColorGradientPyramid: public QuantizedPyramid {
public:
    ColorGradientPyramid(const Mat& src, const Mat& mask, float weak_threshold,
            size_t num_features, float strong_threshold, float threshold_rgb);

    virtual void quantize(Mat& dst) const;

    virtual void quantizeRGB(Mat& dst) const;

    virtual bool extractTemplate(Template& templ) const;

    virtual void pyrDown();

    Mat magnitude_strong;
    Mat magnitude;
protected:
    /// Recalculate angle and magnitude images
    void update(bool compute_magnitude_strong);

    Mat src;
    Mat mask;
    Mat angle;
    Mat rgb;

    int pyramid_level;

    float weak_threshold;
    size_t num_features;
    float strong_threshold;
    float threshold_rgb;
};

ColorGradientPyramid::ColorGradientPyramid(const Mat& src, const Mat& mask,
        float weak_threshold, size_t num_features, float strong_threshold,
        float threshold_rgb) :
        src(src), mask(mask), pyramid_level(0), weak_threshold(weak_threshold), num_features(
                num_features), strong_threshold(strong_threshold), threshold_rgb(
                threshold_rgb) {
    update(true);
}

void ColorGradientPyramid::update(bool compute_magnitude_strong) {
    quantizedOrientations(src, magnitude, angle, rgb, weak_threshold,
            strong_threshold, threshold_rgb, compute_magnitude_strong,
            magnitude_strong);
}

void ColorGradientPyramid::pyrDown() {
    // Some parameters need to be adjusted
    num_features /= 2; /// @todo Why not 4?
    ++pyramid_level;

    // Downsample the current inputs
    Size size(src.cols / 2, src.rows / 2);
    Mat next_src;
    cv::pyrDown(src, next_src, size);
    src = next_src;
    if (!mask.empty()) {
        Mat next_mask;
        resize(mask, next_mask, size, 0.0, 0.0, CV_INTER_NN);
        mask = next_mask;
    }

    update(false);

}

void ColorGradientPyramid::quantize(Mat& dst) const {
    dst = Mat::zeros(angle.size(), CV_8U);
    angle.copyTo(dst, mask);
}

void ColorGradientPyramid::quantizeRGB(Mat& dst) const {
    dst = Mat::zeros(rgb.size(), CV_8UC1);
    rgb.copyTo(dst, mask);
}

//count neighbor's rgb values and check how many oh them are inside the mask
int fillRgbNeighbours(int c_begin, int c_end, int r_begin, int r_end,
        const Mat& rgb, const Mat& magnitude, const Mat& mask,
        int* rgb_values) {
    int countInMask = 0;

    for (int i = r_begin; i <= r_end; i++) {
        const float* magnitude_r = magnitude.ptr<float>(i);
        const uchar* rgb_r = rgb.ptr < uchar > (i);
        const uchar* mask_r = mask.ptr < uchar > (i);

        for (int j = c_begin; j <= c_end; j++) {
            if (mask_r[j])
                countInMask++;

            rgb_values[(int) log2((double) rgb_r[j])]++;
        }
    }

    return countInMask;
}

void calcCR(int c, int r, const Mat& magnitude, uchar p_angle,
        int offsetPlusMinus, int& c_begin, int& c_end, int& r_begin, int& r_end,
        int kS) {
    //first 8 x or y values are positives (greater than central_mass), last 8 are negatives
    int offsetRInit[16] = {/*1+*/-(kS / 2),/*2+*/-(kS / 2),/*4+*/0,/*8+*/0,/*16+*/
    0,/*32+*/0,/*64+*/0,/*128+*/-(kS / 2),/*1-*/-(kS / 2),/*2-*/
    -(kS / 2),/*4-*/-kS + 1,/*8-*/-kS + 1,/*16-*/-kS + 1,/*32-*/-kS + 1,/*64-*/
    -kS + 1,/*128-*/-(kS / 2) };
    int offsetCInit[16] = {/*1+*/-kS + 1,/*2+*/-kS + 1,/*4+*/0,/*8+*/-(kS / 2),/*16+*/
    -(kS / 2),/*32+*/-(kS / 2),/*64+*/-kS + 1,/*128+*/-kS + 1,/*1-*/0,/*2-*/
    0,/*4-*/-kS + 1,/*8-*/-(kS / 2),/*16-*/-(kS / 2),/*32-*/-(kS / 2),/*64-*/
    0,/*128-*/0 };
    int offsetREnd[16] = {/*1+*/kS / 2,/*2+*/kS / 2,/*4+*/kS - 1,/*8+*/kS - 1,/*16+*/
    kS - 1,/*32+*/kS - 1,/*64+*/kS - 1,/*128+*/kS / 2,/*1-*/kS / 2,/*2-*/
    kS / 2,/*4-*/0,/*8-*/0,/*16-*/0,/*32-*/0,/*64-*/0,/*128-*/kS / 2 };
    int offsetCEnd[16] = {/*1+*/0,/*2+*/0,/*4+*/kS - 1,/*8+*/kS / 2,/*16+*/kS
            / 2,/*32+*/kS / 2,/*64+*/0,/*128+*/0,/*1-*/kS - 1,/*2-*/kS - 1,/*4-*/
    0,/*8-*/kS / 2,/*16-*/kS / 2,/*32-*/kS / 2,/*64-*/kS - 1,/*128-*/kS - 1 };

    int quantAngle = getLabel(p_angle);

    c_begin = c + offsetCInit[quantAngle + offsetPlusMinus];
    if (c_begin < 0)
        c_begin = 0;

    c_end = c + offsetCEnd[quantAngle + offsetPlusMinus];
    if (c_end > magnitude.cols)
        c_end = magnitude.cols;

    r_begin = r + offsetRInit[quantAngle + offsetPlusMinus];
    if (r_begin < 0)
        r_begin = 0;

    r_end = r + offsetREnd[quantAngle + offsetPlusMinus];
    if (r_end > magnitude.rows)
        r_end = magnitude.rows;

}

/**
 * \brief Find the most probable color quantization for a pixel on the object border, based on his neighbors
 *
 * Implements section 2.2 "Computing the Gradient Orientations."
 *
 * \param[in] c       		Index of the column of the point
 * \param[in] r 			Index of the row of the point	Destination 
 * \param[in] mag_center    Magnitude of the point 				Destination 8-bit array of orientations. Each bit
 * \param[in] p_angle     	Quantized orientation of the point			Destination 8-bit array of orientations. Each bit
 * \param[in] rgb     		8-bit array of colors.		Destination 8-bit array of orientations. Each bit
 * \param[in] magnitude     Floating-point array of squared magnitudes.				Destination 8-bit array of orientations. Each bit
 * \param[in] mask			Mask of the object
 * \param[in] central_mass  Point of central mass in the mask
 * 
 * \return the color quantization based on the most voted color by his neighbors, excluding the points on the background
 */
uchar votesRgbMag(int c, int r, float mag_center, uchar p_angle, const Mat& rgb,
        const Mat& magnitude, const Mat& mask, Point central_mass) {

    int rgb_values[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

    int kernelSize = 5;

    //check if we need to prove the positive angle or the negative one
    int offsetPlusMinus = 0;
    if (p_angle == 1 || p_angle == 2 || p_angle == 128) {
        if (c >= (int) central_mass.x)
            offsetPlusMinus = 0;
        else
            offsetPlusMinus = 8;
    } else if (p_angle == 8 || p_angle == 16 || p_angle == 32 || p_angle == 64
            || p_angle == 4) {
        if (r <= (int) central_mass.y)
            offsetPlusMinus = 0;
        else
            offsetPlusMinus = 8;
    } else {
        std::cout << "Error: Angle quantization not expected" << std::endl;
        CV_Assert(false);
    }

    int c_begin, c_end, r_begin, r_end;
    calcCR(c, r, magnitude, p_angle, offsetPlusMinus, c_begin, c_end, r_begin,
            r_end, kernelSize);

    int countInMask = fillRgbNeighbours(c_begin, c_end, r_begin, r_end, rgb,
            magnitude, mask, rgb_values);

    //check if enough neighbors are inside the mask
    int neighThresh = ((kernelSize * kernelSize) / 2) + 1;
    if (countInMask < neighThresh) {

        if (offsetPlusMinus == 0)
            offsetPlusMinus = 8;
        else
            offsetPlusMinus = 0;

        calcCR(c, r, magnitude, p_angle, offsetPlusMinus, c_begin, c_end,
                r_begin, r_end, kernelSize);

        for (int g = 0; g < 8; g++)
            rgb_values[g] = 0;

        countInMask = fillRgbNeighbours(c_begin, c_end, r_begin, r_end, rgb,
                magnitude, mask, rgb_values);

        //if we still don't have enough neighbors, return the start value
        if (countInMask < neighThresh) {
            const uchar* rgb_r = rgb.ptr < uchar > (r);
            return rgb_r[c];
        }
    }

    int max_votes = 0;
    int index = -1;
    for (int i = 0; i < 8; i++) {
        if (rgb_values[i] > max_votes) {
            max_votes = rgb_values[i];
            index = i;
        }
    }

    uchar result;
    if (index != -1)
        result = (uchar) pow(2, index);
    else {
        const uchar* rgb_r = rgb.ptr < uchar > (r);
        result = rgb_r[c];
    }

    return result;

}

bool ColorGradientPyramid::extractTemplate(Template& templ) const {
    // Want features on the border to distinguish from background
    Mat local_mask;
    Mat border_mask;

    if (!mask.empty()) {
        mask.copyTo(local_mask);
        erode(mask, border_mask, Mat(), Point(-1, -1), 3, BORDER_REPLICATE);
        subtract(mask, border_mask, border_mask);
    }

    // Create sorted list of all pixels with magnitude greater than a threshold
    std::vector < Candidate > candidates;
    std::vector < Candidate > candidates_color_features;
    bool no_mask = local_mask.empty();
    float threshold_sq = CV_SQR(strong_threshold);

    int quantCount = 0;

    //find the central mass point
    Moments moms = moments(mask, true);
    Point central_mass(moms.m10 / moms.m00, moms.m01 / moms.m00);
    float mask_area = moms.m00;
    float num_color_features = mask_area / 40;
    float color_distance = sqrtf(static_cast<float>(mask_area))
            / sqrtf(static_cast<float>(num_color_features)) + 1.5f;

    int on_borderCount = 0;

    for (int r = 0; r < magnitude.rows; ++r) {
        const uchar* angle_r = angle.ptr < uchar > (r);
        const uchar* rgb_r = rgb.ptr < uchar > (r);

        const float* magnitude_r = magnitude.ptr<float>(r);
        const uchar* mask_r = no_mask ? NULL : local_mask.ptr < uchar > (r);
        const uchar* border_mask_r =
                no_mask ? NULL : border_mask.ptr < uchar > (r);

        for (int c = 0; c < magnitude.cols; ++c) {

            if (no_mask || mask_r[c]) {
                uchar quantized = angle_r[c];
                uchar quantized_rgb = rgb_r[c];

                if (magnitude_r[c] > threshold_sq) {
                    quantCount++;
                }

                if (pyramid_level == 1 && !border_mask_r[c]) {
                    candidates_color_features.push_back(
                            Candidate(c, r, -1, getLabel(quantized_rgb), false,
                                    -1));
                }

                if (quantized > 0) {

                    float score = magnitude_r[c];

                    if (score > threshold_sq) {

                        if (border_mask_r[c]) {
                            //Voting color neighbors
                            uchar voted = votesRgbMag(c, r, score, quantized,
                                    rgb, magnitude, mask, central_mass);
                            candidates.push_back(
                                    Candidate(c, r, getLabel(quantized),
                                            getLabel(voted), true, score));
                            on_borderCount++;
                        } else
                            candidates.push_back(
                                    Candidate(c, r, getLabel(quantized),
                                            getLabel(quantized_rgb), false,
                                            score));

                    }
                }
            }
        }

    }

    templ.total_candidates = quantCount;

    // We require a certain number of features
    if (candidates.size() < num_features)
        return false;
    // NOTE: Stable sort to agree with old code, which used std::list::sort()
    std::stable_sort(candidates.begin(), candidates.end());

    if (pyramid_level == 1 && on_borderCount <= (candidates.size() / 2)) {
        candidates_color_features.clear();
    } else if (pyramid_level == 1) {
        selectScatteredFeatures(candidates_color_features, templ.color_features,
                num_color_features, color_distance);
    }

    // Use heuristic based on surplus of candidates in narrow outline for initial distance threshold
    float distance = static_cast<float>(candidates.size() / num_features + 1);
    selectScatteredFeatures(candidates, templ.features, num_features, distance);

    // Size determined externally, needs to match templates for other modalities
    templ.width = -1;
    templ.height = -1;
    templ.pyramid_level = pyramid_level;

    return true;
}

ColorGradient::ColorGradient() :
        weak_threshold(50.0f), num_features(126), strong_threshold(70.0f), threshold_rgb(
                65) {
}

ColorGradient::ColorGradient(float weak_threshold, size_t num_features,
        float strong_threshold, float threshold_rgb) :
        weak_threshold(weak_threshold), num_features(num_features), strong_threshold(
                strong_threshold), threshold_rgb(threshold_rgb) {
}

static const char CG_NAME[] = "ColorGradient";

std::string ColorGradient::name() const {
    return CG_NAME;
}

Ptr<QuantizedPyramid> ColorGradient::processImpl(const Mat& src,
        const Mat& mask) const {
    return new ColorGradientPyramid(src, mask, weak_threshold, num_features,
            strong_threshold, threshold_rgb);
}

void ColorGradient::read(const FileNode& fn) {
    std::string type = fn["type"];
    CV_Assert(type == CG_NAME);

    weak_threshold = fn["weak_threshold"];
    num_features = int(fn["num_features"]);
    strong_threshold = fn["strong_threshold"];
}

void ColorGradient::write(FileStorage& fs) const {
    fs << "type" << CG_NAME;
    fs << "weak_threshold" << weak_threshold;
    fs << "num_features" << int(num_features);
    fs << "strong_threshold" << strong_threshold;
}

/****************************************************************************************\
*                               Depth normal modality                                    *
 \****************************************************************************************/

// Contains GRANULARITY and NORMAL_LUT
#include "normal_lut.i"

static void accumBilateral(long delta, long i, long j, long * A, long * b,
        int threshold) {
    long f = std::abs(delta) < threshold ? 1 : 0;

    const long fi = f * i;
    const long fj = f * j;

    A[0] += fi * i;
    A[1] += fi * j;
    A[3] += fj * j;
    b[0] += fi * delta;
    b[1] += fj * delta;
}

/**
 * \brief Compute quantized normal image from depth image.
 *
 * Implements section 2.6 "Extension to Dense Depth Sensors."
 *
 * \param[in]  src  The source 16-bit depth image (in mm).
 * \param[out] dst  The destination 8-bit image. Each bit represents one bin of
 *                  the view cone.
 * \param distance_threshold   Ignore pixels beyond this distance.
 * \param difference_threshold When computing normals, ignore contributions of pixels whose
 *                             depth difference with the central pixel is above this threshold.
 *
 * \todo Should also need camera model, or at least focal lengths? Replace distance_threshold with mask?
 */
void quantizedNormals(const Mat& src, Mat& dst, int distance_threshold,
        int difference_threshold) {
    dst = Mat::zeros(src.size(), CV_8U);

    IplImage src_ipl = src;
    IplImage* ap_depth_data = &src_ipl;
    IplImage dst_ipl = dst;
    IplImage* dst_ipl_ptr = &dst_ipl;
    IplImage** m_dep = &dst_ipl_ptr;

    unsigned short * lp_depth = (unsigned short *) ap_depth_data->imageData;
    unsigned char * lp_normals = (unsigned char *) m_dep[0]->imageData;

    const int l_W = ap_depth_data->width;
    const int l_H = ap_depth_data->height;

    const int l_r = 5; // used to be 7
    const int l_offset0 = -l_r - l_r * l_W;
    const int l_offset1 = 0 - l_r * l_W;
    const int l_offset2 = +l_r - l_r * l_W;
    const int l_offset3 = -l_r;
    const int l_offset4 = +l_r;
    const int l_offset5 = -l_r + l_r * l_W;
    const int l_offset6 = 0 + l_r * l_W;
    const int l_offset7 = +l_r + l_r * l_W;

    const int l_offsetx = GRANULARITY / 2;
    const int l_offsety = GRANULARITY / 2;

    for (int l_y = l_r; l_y < l_H - l_r - 1; ++l_y) {
        unsigned short * lp_line = lp_depth + (l_y * l_W + l_r);
        unsigned char * lp_norm = lp_normals + (l_y * l_W + l_r);

        for (int l_x = l_r; l_x < l_W - l_r - 1; ++l_x) {
            long l_d = lp_line[0];

            if (l_d < distance_threshold) {
                // accum
                long l_A[4];
                l_A[0] = l_A[1] = l_A[2] = l_A[3] = 0;
                long l_b[2];
                l_b[0] = l_b[1] = 0;
                accumBilateral(lp_line[l_offset0] - l_d, -l_r, -l_r, l_A, l_b,
                        difference_threshold);
                accumBilateral(lp_line[l_offset1] - l_d, 0, -l_r, l_A, l_b,
                        difference_threshold);
                accumBilateral(lp_line[l_offset2] - l_d, +l_r, -l_r, l_A, l_b,
                        difference_threshold);
                accumBilateral(lp_line[l_offset3] - l_d, -l_r, 0, l_A, l_b,
                        difference_threshold);
                accumBilateral(lp_line[l_offset4] - l_d, +l_r, 0, l_A, l_b,
                        difference_threshold);
                accumBilateral(lp_line[l_offset5] - l_d, -l_r, +l_r, l_A, l_b,
                        difference_threshold);
                accumBilateral(lp_line[l_offset6] - l_d, 0, +l_r, l_A, l_b,
                        difference_threshold);
                accumBilateral(lp_line[l_offset7] - l_d, +l_r, +l_r, l_A, l_b,
                        difference_threshold);

                // solve
                long l_det = l_A[0] * l_A[3] - l_A[1] * l_A[1];
                long l_ddx = l_A[3] * l_b[0] - l_A[1] * l_b[1];
                long l_ddy = -l_A[1] * l_b[0] + l_A[0] * l_b[1];

                /// @todo Magic number 1150 is focal length? This is something like
                /// f in SXGA mode, but in VGA is more like 530.
                float l_nx = static_cast<float>(1150 * l_ddx);
                float l_ny = static_cast<float>(1150 * l_ddy);
                float l_nz = static_cast<float>(-l_det * l_d);

                float l_sqrt = sqrtf(l_nx * l_nx + l_ny * l_ny + l_nz * l_nz);

                if (l_sqrt > 0) {
                    float l_norminv = 1.0f / (l_sqrt);

                    l_nx *= l_norminv;
                    l_ny *= l_norminv;
                    l_nz *= l_norminv;

                    //*lp_norm = fabs(l_nz)*255;

                    int l_val1 = static_cast<int>(l_nx * l_offsetx + l_offsetx);
                    int l_val2 = static_cast<int>(l_ny * l_offsety + l_offsety);
                    int l_val3 = static_cast<int>(l_nz * GRANULARITY
                            + GRANULARITY);

                    *lp_norm = NORMAL_LUT[l_val3][l_val2][l_val1];
                } else {
                    *lp_norm = 0; // Discard shadows from depth sensor
                }
            } else {
                *lp_norm = 0; //out of depth
            }
            ++lp_line;
            ++lp_norm;
        }
    }
    cvSmooth(m_dep[0], m_dep[0], CV_MEDIAN, 5, 5);
}

class DepthNormalPyramid: public QuantizedPyramid {
public:
    DepthNormalPyramid(const Mat& src, const Mat& mask, int distance_threshold,
            int difference_threshold, size_t num_features,
            int extract_threshold);

    virtual void quantize(Mat& dst) const;

    virtual void quantizeRGB(Mat& dst) const;

    virtual bool extractTemplate(Template& templ) const;

    virtual void pyrDown();

protected:
    Mat mask;

    int pyramid_level;
    Mat normal;

    size_t num_features;
    int extract_threshold;
};

DepthNormalPyramid::DepthNormalPyramid(const Mat& src, const Mat& mask,
        int distance_threshold, int difference_threshold, size_t num_features,
        int extract_threshold) :
        mask(mask), pyramid_level(0), num_features(num_features), extract_threshold(
                extract_threshold) {
    quantizedNormals(src, normal, distance_threshold, difference_threshold);
}

void DepthNormalPyramid::pyrDown() {
    // Some parameters need to be adjusted
    num_features /= 2; /// @todo Why not 4?
    extract_threshold /= 2;
    ++pyramid_level;

    // In this case, NN-downsample the quantized image
    Mat next_normal;
    Size size(normal.cols / 2, normal.rows / 2);
    resize(normal, next_normal, size, 0.0, 0.0, CV_INTER_NN);
    normal = next_normal;
    if (!mask.empty()) {
        Mat next_mask;
        resize(mask, next_mask, size, 0.0, 0.0, CV_INTER_NN);
        mask = next_mask;
    }
}

void DepthNormalPyramid::quantize(Mat& dst) const {
    dst = Mat::zeros(normal.size(), CV_8U);
    normal.copyTo(dst, mask);
}

void DepthNormalPyramid::quantizeRGB(Mat& dst) const {
    dst = Mat::zeros(normal.size(), CV_8U);
    normal.copyTo(dst, mask);
}

bool DepthNormalPyramid::extractTemplate(Template& templ) const {
    // Features right on the object border are unreliable
    Mat local_mask;
    if (!mask.empty()) {
        erode(mask, local_mask, Mat(), Point(-1, -1), 2, BORDER_REPLICATE);
    }

    // Compute distance transform for each individual quantized orientation
    Mat temp = Mat::zeros(normal.size(), CV_8U);
    Mat distances[8];
    for (int i = 0; i < 8; ++i) {
        temp.setTo(1 << i, local_mask);
        bitwise_and(temp, normal, temp);
        // temp is now non-zero at pixels in the mask with quantized orientation i
        distanceTransform(temp, distances[i], CV_DIST_C, 3);
    }

    // Count how many features taken for each label
    int label_counts[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

    // Create sorted list of candidate features
    std::vector < Candidate > candidates;
    bool no_mask = local_mask.empty();
    for (int r = 0; r < normal.rows; ++r) {
        const uchar* normal_r = normal.ptr < uchar > (r);
        const uchar* mask_r = no_mask ? NULL : local_mask.ptr < uchar > (r);

        for (int c = 0; c < normal.cols; ++c) {
            if (no_mask || mask_r[c]) {
                uchar quantized = normal_r[c];

                if (quantized != 0 && quantized != 255) // background and shadow
                        {
                    int label = getLabel(quantized);

                    // Accept if distance to a pixel belonging to a different label is greater than
                    // some threshold. IOW, ideal feature is in the center of a large homogeneous
                    // region.
                    float score = distances[label].at<float>(r, c);
                    if (score >= extract_threshold) {
                        candidates.push_back(Candidate(c, r, label, score));
                        ++label_counts[label];
                    }
                }
            }
        }
    }
    // We require a certain number of features
    if (candidates.size() < num_features)
        return false;

    // Prefer large distances, but also want to collect features over all 8 labels.
    // So penalize labels with lots of candidates.
    for (size_t i = 0; i < candidates.size(); ++i) {
        Candidate& c = candidates[i];
        c.score /= (float) label_counts[c.f.label];
    }
    std::stable_sort(candidates.begin(), candidates.end());

    // Use heuristic based on object area for initial distance threshold
    int area = static_cast<int>(
            no_mask ? normal.total() : countNonZero(local_mask));
    float distance = sqrtf(static_cast<float>(area))
            / sqrtf(static_cast<float>(num_features)) + 1.5f;
    selectScatteredFeatures(candidates, templ.features, num_features, distance);

    // Size determined externally, needs to match templates for other modalities
    templ.width = -1;
    templ.height = -1;
    templ.pyramid_level = pyramid_level;

    return true;
}

DepthNormal::DepthNormal() :
        distance_threshold(2000), difference_threshold(50), num_features(63), extract_threshold(
                2) {
}

DepthNormal::DepthNormal(int distance_threshold, int difference_threshold,
        size_t num_features, int extract_threshold) :
        distance_threshold(distance_threshold), difference_threshold(
                difference_threshold), num_features(num_features), extract_threshold(
                extract_threshold) {
}

static const char DN_NAME[] = "DepthNormal";

std::string DepthNormal::name() const {
    return DN_NAME;
}

Ptr<QuantizedPyramid> DepthNormal::processImpl(const Mat& src,
        const Mat& mask) const {
    return new DepthNormalPyramid(src, mask, distance_threshold,
            difference_threshold, num_features, extract_threshold);
}

void DepthNormal::read(const FileNode& fn) {
    std::string type = fn["type"];
    CV_Assert(type == DN_NAME);

    distance_threshold = fn["distance_threshold"];
    difference_threshold = fn["difference_threshold"];
    num_features = int(fn["num_features"]);
    extract_threshold = fn["extract_threshold"];
}

void DepthNormal::write(FileStorage& fs) const {
    fs << "type" << DN_NAME;
    fs << "distance_threshold" << distance_threshold;
    fs << "difference_threshold" << difference_threshold;
    fs << "num_features" << int(num_features);
    fs << "extract_threshold" << extract_threshold;
}

/****************************************************************************************\
*                                 Response maps                                          *
 \****************************************************************************************/

void orUnaligned8u(const uchar * src, const int src_stride, uchar * dst,
        const int dst_stride, const int width, const int height) {
#if CV_SSE2
    volatile bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
    volatile bool haveSSE3 = checkHardwareSupport(CV_CPU_SSE3);
//haveSSE2 = false;
//haveSSE3 = false;
    bool src_aligned = reinterpret_cast<unsigned long long>(src) % 16 == 0;
#endif

    for (int r = 0; r < height; ++r) {
        int c = 0;

#if CV_SSE2
        // Use aligned loads if possible
        if (haveSSE2 && src_aligned)
        {
            for (; c < width - 15; c += 16)
            {
                const __m128i* src_ptr = reinterpret_cast<const __m128i*>(src + c);
                __m128i* dst_ptr = reinterpret_cast<__m128i*>(dst + c);
                *dst_ptr = _mm_or_si128(*dst_ptr, *src_ptr);
            }
        }
#if CV_SSE3
        // Use LDDQU for fast unaligned load
        else if (haveSSE3)
        {
            for (; c < width - 15; c += 16)
            {
                __m128i val = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(src + c));
                __m128i* dst_ptr = reinterpret_cast<__m128i*>(dst + c);
                *dst_ptr = _mm_or_si128(*dst_ptr, val);
            }
        }
#endif
        // Fall back to MOVDQU
        else if (haveSSE2)
        {
            for (; c < width - 15; c += 16)
            {
                __m128i val = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + c));
                __m128i* dst_ptr = reinterpret_cast<__m128i*>(dst + c);
                *dst_ptr = _mm_or_si128(*dst_ptr, val);
            }
        }
#endif

        for (; c < width; ++c)
            dst[c] |= src[c];

        // Advance to next row
        src += src_stride;
        dst += dst_stride;
    }

}

/**
 * \brief Spread binary labels in a quantized image.
 *
 * Implements section 2.3 "Spreading the Orientations."
 *
 * \param[in]  src The source 8-bit quantized image.
 * \param[out] dst Destination 8-bit spread image.
 * \param      T   Sampling step. Spread labels T/2 pixels in each direction.
 */
void spread(const Mat& src, Mat& dst, int T) {
    // Allocate and zero-initialize spread (OR'ed) image
    dst = Mat::zeros(src.size(), CV_8U);

    // Fill in spread gradient image (section 2.3)
    for (int r = 0; r < T; ++r) {
        int height = src.rows - r;
        for (int c = 0; c < T; ++c) {
            orUnaligned8u(&src.at<unsigned char>(r, c),
                    static_cast<const int>(src.step1()), dst.ptr(),
                    static_cast<const int>(dst.step1()), src.cols - c, height);
        }
    }
}

// Auto-generated by create_similarity_lut.py
CV_DECL_ALIGNED(16) static const unsigned char SIMILARITY_LUT[256] = {0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 4, 4, 3, 3, 4, 4, 2, 3, 4, 4, 3, 3, 4, 4, 0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 0, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4, 0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 3, 4, 4, 3, 3, 4, 4, 2, 3, 4, 4, 3, 3, 4, 4, 0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 2, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 0, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3, 0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};

static const unsigned char SIMILARITY_RGB_LUT[8][256] = { { 0, 16, 0, 16, 0, 16,
        0, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2,
        16, 2, 16, 2, 16, 0, 16, 0, 16, 0, 16, 0, 16, 2, 16, 2, 16, 2, 16, 2,
        16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 1, 16, 1,
        16, 1, 16, 1, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2,
        16, 2, 16, 2, 16, 2, 16, 2, 16, 1, 16, 1, 16, 1, 16, 1, 16, 2, 16, 2,
        16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2,
        16, 1, 16, 1, 16, 1, 16, 1, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2,
        16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 1, 16, 1, 16, 1, 16, 1,
        16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2,
        16, 2, 16, 2, 16, 1, 16, 1, 16, 1, 16, 1, 16, 2, 16, 2, 16, 2, 16, 2,
        16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 1, 16, 1,
        16, 1, 16, 1, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 16, 2,
        16, 2, 16, 2, 16, 2, 16, 2, 16 }, { 0, 0, 16, 16, 0, 0, 16, 16, 2, 2,
        16, 16, 2, 2, 16, 16, 0, 0, 16, 16, 0, 0, 16, 16, 2, 2, 16, 16, 2, 2,
        16, 16, 2, 2, 16, 16, 2, 2, 16, 16, 2, 2, 16, 16, 2, 2, 16, 16, 2, 2,
        16, 16, 2, 2, 16, 16, 2, 2, 16, 16, 2, 2, 16, 16, 1, 1, 16, 16, 1, 1,
        16, 16, 2, 2, 16, 16, 2, 2, 16, 16, 1, 1, 16, 16, 1, 1, 16, 16, 2, 2,
        16, 16, 2, 2, 16, 16, 2, 2, 16, 16, 2, 2, 16, 16, 2, 2, 16, 16, 2, 2,
        16, 16, 2, 2, 16, 16, 2, 2, 16, 16, 2, 2, 16, 16, 2, 2, 16, 16, 1, 1,
        16, 16, 1, 1, 16, 16, 2, 2, 16, 16, 2, 2, 16, 16, 1, 1, 16, 16, 1, 1,
        16, 16, 2, 2, 16, 16, 2, 2, 16, 16, 2, 2, 16, 16, 2, 2, 16, 16, 2, 2,
        16, 16, 2, 2, 16, 16, 2, 2, 16, 16, 2, 2, 16, 16, 2, 2, 16, 16, 2, 2,
        16, 16, 1, 1, 16, 16, 1, 1, 16, 16, 2, 2, 16, 16, 2, 2, 16, 16, 1, 1,
        16, 16, 1, 1, 16, 16, 2, 2, 16, 16, 2, 2, 16, 16, 2, 2, 16, 16, 2, 2,
        16, 16, 2, 2, 16, 16, 2, 2, 16, 16, 2, 2, 16, 16, 2, 2, 16, 16, 2, 2,
        16, 16, 2, 2, 16, 16 }, { 0, 0, 0, 0, 16, 16, 16, 16, 0, 0, 0, 0, 16,
        16, 16, 16, 2, 2, 2, 2, 16, 16, 16, 16, 2, 2, 2, 2, 16, 16, 16, 16, 2,
        2, 2, 2, 16, 16, 16, 16, 2, 2, 2, 2, 16, 16, 16, 16, 2, 2, 2, 2, 16, 16,
        16, 16, 2, 2, 2, 2, 16, 16, 16, 16, 1, 1, 1, 1, 16, 16, 16, 16, 1, 1, 1,
        1, 16, 16, 16, 16, 2, 2, 2, 2, 16, 16, 16, 16, 2, 2, 2, 2, 16, 16, 16,
        16, 2, 2, 2, 2, 16, 16, 16, 16, 2, 2, 2, 2, 16, 16, 16, 16, 2, 2, 2, 2,
        16, 16, 16, 16, 2, 2, 2, 2, 16, 16, 16, 16, 1, 1, 1, 1, 16, 16, 16, 16,
        1, 1, 1, 1, 16, 16, 16, 16, 2, 2, 2, 2, 16, 16, 16, 16, 2, 2, 2, 2, 16,
        16, 16, 16, 2, 2, 2, 2, 16, 16, 16, 16, 2, 2, 2, 2, 16, 16, 16, 16, 2,
        2, 2, 2, 16, 16, 16, 16, 2, 2, 2, 2, 16, 16, 16, 16, 1, 1, 1, 1, 16, 16,
        16, 16, 1, 1, 1, 1, 16, 16, 16, 16, 2, 2, 2, 2, 16, 16, 16, 16, 2, 2, 2,
        2, 16, 16, 16, 16, 2, 2, 2, 2, 16, 16, 16, 16, 2, 2, 2, 2, 16, 16, 16,
        16, 2, 2, 2, 2, 16, 16, 16, 16, 2, 2, 2, 2, 16, 16, 16, 16 }, { 0, 2, 2,
        2, 0, 2, 2, 2, 16, 16, 16, 16, 16, 16, 16, 16, 2, 2, 2, 2, 2, 2, 2, 2,
        16, 16, 16, 16, 16, 16, 16, 16, 2, 2, 2, 2, 2, 2, 2, 2, 16, 16, 16, 16,
        16, 16, 16, 16, 2, 2, 2, 2, 2, 2, 2, 2, 16, 16, 16, 16, 16, 16, 16, 16,
        2, 2, 2, 2, 2, 2, 2, 2, 16, 16, 16, 16, 16, 16, 16, 16, 2, 2, 2, 2, 2,
        2, 2, 2, 16, 16, 16, 16, 16, 16, 16, 16, 2, 2, 2, 2, 2, 2, 2, 2, 16, 16,
        16, 16, 16, 16, 16, 16, 2, 2, 2, 2, 2, 2, 2, 2, 16, 16, 16, 16, 16, 16,
        16, 16, 2, 2, 2, 2, 2, 2, 2, 2, 16, 16, 16, 16, 16, 16, 16, 16, 2, 2, 2,
        2, 2, 2, 2, 2, 16, 16, 16, 16, 16, 16, 16, 16, 2, 2, 2, 2, 2, 2, 2, 2,
        16, 16, 16, 16, 16, 16, 16, 16, 2, 2, 2, 2, 2, 2, 2, 2, 16, 16, 16, 16,
        16, 16, 16, 16, 2, 2, 2, 2, 2, 2, 2, 2, 16, 16, 16, 16, 16, 16, 16, 16,
        2, 2, 2, 2, 2, 2, 2, 2, 16, 16, 16, 16, 16, 16, 16, 16, 2, 2, 2, 2, 2,
        2, 2, 2, 16, 16, 16, 16, 16, 16, 16, 16, 2, 2, 2, 2, 2, 2, 2, 2, 16, 16,
        16, 16, 16, 16, 16, 16 }, { 0, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16 }, { 0, 0, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16 }, { 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16 }, { 0, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16 } };
/**
 * \brief Precompute response maps for a spread quantized image.
 *
 * Implements section 2.4 "Precomputing Response Maps."
 *
 * \param[in]  src           The source 8-bit spread quantized image.
 * \param[out] response_maps Vector of 8 response maps, one for each bit label.
 */
void computeResponseMaps(const Mat& src, std::vector<Mat>& response_maps) {

    CV_Assert((src.rows * src.cols) % 16 == 0);

    // Allocate response maps
    response_maps.resize(8);
    for (int i = 0; i < 8; ++i)
        response_maps[i].create(src.size(), CV_8U);

    Mat lsb4(src.size(), CV_8U);
    Mat msb4(src.size(), CV_8U);

    for (int r = 0; r < src.rows; ++r) {
        const uchar* src_r = src.ptr(r);
        uchar* lsb4_r = lsb4.ptr(r);
        uchar* msb4_r = msb4.ptr(r);

        for (int c = 0; c < src.cols; ++c) {
            // Least significant 4 bits of spread image pixel
            lsb4_r[c] = src_r[c] & 15;
            // Most significant 4 bits, right-shifted to be in [0, 16)
            msb4_r[c] = (src_r[c] & 240) >> 4;
        }
    }

    {
        // For each of the 8 quantized orientations...
        for (int ori = 0; ori < 8; ++ori) {
            uchar* map_data = response_maps[ori].ptr<uchar>();
            uchar* lsb4_data = lsb4.ptr<uchar>();
            uchar* msb4_data = msb4.ptr<uchar>();
            const uchar* lut_low = SIMILARITY_LUT + 32 * ori;
            const uchar* lut_hi = lut_low + 16;

            for (int i = 0; i < src.rows * src.cols; ++i) {
                map_data[i] = std::max(lut_low[lsb4_data[i]],
                        lut_hi[msb4_data[i]]);

            }

        }
    }
}

void computeResponseMapsRGB(const Mat& src, std::vector<Mat>& response_maps) {

    // Allocate response maps
    response_maps.resize(8);
    for (int i = 0; i < 8; ++i)
        response_maps[i].create(src.size(), CV_8UC1);

    int ind = 0;
    for (int r = 0; r < src.rows; ++r) {
        const uchar* src_r = src.ptr(r);

        for (int c = 0; c < src.cols; ++c) {

            for (int ori = 0; ori < 8; ori++) {
                response_maps[ori].at < uchar > (r, c) =
                        SIMILARITY_RGB_LUT[ori][(int) src_r[c]];
            }

        }

    }

}

/**
 * \brief Convert a response map to fast linearized ordering.
 *
 * Implements section 2.5 "Linearizing the Memory for Parallelization."
 *
 * \param[in]  response_map The 2D response map, an 8-bit image.
 * \param[out] linearized   The response map in linearized order. It has T*T rows,
 *                          each of which is a linear memory of length (W/T)*(H/T).
 * \param      T            Sampling step.
 */
void linearize(const Mat& response_map, Mat& linearized, int T) {

    CV_Assert(response_map.rows % T == 0);
    CV_Assert(response_map.cols % T == 0);

    // linearized has T^2 rows, where each row is a linear memory
    int mem_width = response_map.cols / T;
    int mem_height = response_map.rows / T;

    linearized.create(T * T, mem_width * mem_height, CV_8U);

    // Outer two for loops iterate over top-left T^2 starting pixels
    int index = 0;
    for (int r_start = 0; r_start < T; ++r_start) {
        for (int c_start = 0; c_start < T; ++c_start) {
            uchar* memory_uchar = linearized.ptr(index);
            ++index;

            // Inner two loops copy every T-th pixel into the linear memory
            for (int r = r_start; r < response_map.rows; r += T) {
                const uchar* response_data = response_map.ptr(r);

                for (int c = c_start; c < response_map.cols; c += T)
                    *memory_uchar++ = response_data[c];
            }
        }
    }

}

/****************************************************************************************\
*                               Linearized similarities                                  *
 \****************************************************************************************/

const unsigned char* accessLinearMemory(const std::vector<Mat>& linear_memories,
        const Feature& f, int T, int W) {
    // Retrieve the TxT grid of linear memories associated with the feature label
    const Mat& memory_grid = linear_memories[f.label];
    CV_DbgAssert(memory_grid.rows == T * T);
    CV_DbgAssert(f.x >= 0);
    CV_DbgAssert(f.y >= 0);
    // The LM we want is at (x%T, y%T) in the TxT grid (stored as the rows of memory_grid)
    int grid_x = f.x % T;
    int grid_y = f.y % T;
    int grid_index = grid_y * T + grid_x;
    CV_DbgAssert(grid_index >= 0);
    CV_DbgAssert(grid_index < memory_grid.rows);
    const unsigned char* memory = memory_grid.ptr(grid_index);
    // Within the LM, the feature is at (x/T, y/T). W is the "width" of the LM, the
    // input image width decimated by T.
    int lm_x = f.x / T;
    int lm_y = f.y / T;
    int lm_index = lm_y * W + lm_x;
    CV_DbgAssert(lm_index >= 0);
    CV_DbgAssert(lm_index < memory_grid.cols);
    return memory + lm_index;
}

const unsigned char* accessLinearMemoryRGB(
        const std::vector<Mat>& linear_memories, const Feature& f, int T,
        int W) {
    // Retrieve the TxT grid of linear memories associated with the feature label
    const Mat& memory_grid = linear_memories[f.rgb_label];

    CV_DbgAssert(memory_grid.rows == T * T);
    CV_DbgAssert(f.x >= 0);
    CV_DbgAssert(f.y >= 0);
    // The LM we want is at (x%T, y%T) in the TxT grid (stored as the rows of memory_grid)
    int grid_x = f.x % T;
    int grid_y = f.y % T;
    int grid_index = grid_y * T + grid_x;
    CV_DbgAssert(grid_index >= 0);
    CV_DbgAssert(grid_index < memory_grid.rows);
    const unsigned char* memory = memory_grid.ptr(grid_index);
    // Within the LM, the feature is at (x/T, y/T). W is the "width" of the LM, the
    // input image width decimated by T.
    int lm_x = f.x / T;
    int lm_y = f.y / T;
    int lm_index = lm_y * W + lm_x;
    CV_DbgAssert(lm_index >= 0);
    CV_DbgAssert(lm_index < memory_grid.cols);
    return memory + lm_index;
}

/**
 * \brief Compute similarity measure for a given template at each sampled image location.
 *
 * Uses linear memories to compute the similarity measure as described in Fig. 7.
 *
 * \param[in]  linear_memories Vector of 8 linear memories, one for each label.
 * \param[in]  templ           Template to match against.
 * \param[out] dst             Destination 8-bit similarity image of size (W/T, H/T).
 * \param      size            Size (W, H) of the original input image.
 * \param      T               Sampling step.
 */
void similarity(const std::vector<Mat>& linear_memories, const Template& templ,
        Mat& dst, Size size, int T) {

    // 63 features or less is a special case because the max similarity per-feature is 4.
    // 255/4 = 63, so up to that many we can add up similarities in 8 bits without worrying
    // about overflow. Therefore here we use _mm_add_epi8 as the workhorse, whereas a more
    // general function would use _mm_add_epi16.
    CV_Assert(templ.features.size() <= 63);
    /// @todo Handle more than 255/MAX_RESPONSE features!!

    // Decimate input image size by factor of T
    int W = size.width / T;
    int H = size.height / T;

    // Feature dimensions, decimated by factor T and rounded up
    int wf = (templ.width - 1) / T + 1;
    int hf = (templ.height - 1) / T + 1;

    // Span is the range over which we can shift the template around the input image
    int span_x = W - wf;
    int span_y = H - hf;

    // Compute number of contiguous (in memory) pixels to check when sliding feature over
    // image. This allows template to wrap around left/right border incorrectly, so any
    // wrapped template matches must be filtered out!
    int template_positions = span_y * W + span_x + 1; // why add 1?
    //int template_positions = (span_y - 1) * W + span_x; // More correct?

    /// @todo In old code, dst is buffer of size m_U. Could make it something like
    /// (span_x)x(span_y) instead?
    dst = Mat::zeros(H, W, CV_8U);
    uchar* dst_ptr = dst.ptr<uchar>();

#if CV_SSE2
    volatile bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
#if CV_SSE3
    volatile bool haveSSE3 = checkHardwareSupport(CV_CPU_SSE3);
#endif
#endif

//haveSSE2 = false;
//haveSSE3 = false;

    // Compute the similarity measure for this template by accumulating the contribution of
    // each feature
    for (int i = 0; i < (int) templ.features.size(); ++i) {
        // Add the linear memory at the appropriate offset computed from the location of
        // the feature in the template
        Feature f = templ.features[i];
        // Discard feature if out of bounds
        /// @todo Shouldn't actually see x or y < 0 here?
        if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height)
            continue;
        const uchar* lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        // Now we do an aligned/unaligned add of dst_ptr and lm_ptr with template_positions elements
        int j = 0;
        // Process responses 16 at a time if vectorization possible
#if CV_SSE2
#if CV_SSE3
        if (haveSSE3)
        {
            // LDDQU may be more efficient than MOVDQU for unaligned load of next 16 responses
            for (; j < template_positions - 15; j += 16)
            {
                __m128i responses = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(lm_ptr + j));
                __m128i* dst_ptr_sse = reinterpret_cast<__m128i*>(dst_ptr + j);
                *dst_ptr_sse = _mm_add_epi8(*dst_ptr_sse, responses);
            }
        }
        else
#endif
        if (haveSSE2)
        {
            // Fall back to MOVDQU
            for (; j < template_positions - 15; j += 16)
            {
                __m128i responses = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lm_ptr + j));
                __m128i* dst_ptr_sse = reinterpret_cast<__m128i*>(dst_ptr + j);
                *dst_ptr_sse = _mm_add_epi8(*dst_ptr_sse, responses);
            }
        }
#endif

        for (; j < template_positions; ++j)
            dst_ptr[j] += lm_ptr[j];
    }
}

/**
 * \brief Compute similarity measure for a given template at each sampled image location.
 *
 * Uses linear memories to compute the similarity measure as described in Fig. 7.
 *
 * \param[in]  linear_memories     		Vector of 8 linear memories, one for each label.
 * \param[in]  linear_memories_rgb 		Vector of 8 rgb linear memories, one for each rgb label.
 * \param[in]  templ           	   		Template to match against.
 * \param[out] dst                 		Destination 8-bit similarity image of size (W/T, H/T).
 * \param[out] dst                 		Destination 8-bit similarity rgb image of size (W/T, H/T).
 * \param      size                		Size (W, H) of the original input image.
 * \param      T                   		Sampling step.
 * \param      color_features_enabled   Boolean to estabilish if we also need to process the color features.
 */
void similarityRGB(const std::vector<Mat>& linear_memories,
        const std::vector<Mat>& linear_memories_rgb, const Template& templ,
        Mat& dst, Mat& dst_rgb, Size size, int T, bool color_features_enabled) {

    //the 63 features limit has been removed in LINERGB
    //CV_Assert(templ.features.size() <= 63);

    // Decimate input image size by factor of T
    int W = size.width / T;
    int H = size.height / T;

    // Feature dimensions, decimated by factor T and rounded up
    int wf = (templ.width - 1) / T + 1;
    int hf = (templ.height - 1) / T + 1;

    // Span is the range over which we can shift the template around the input image
    int span_x = W - wf;
    int span_y = H - hf;

    // Compute number of contiguous (in memory) pixels to check when sliding feature over
    // image. This allows template to wrap around left/right border incorrectly, so any
    // wrapped template matches must be filtered out!
    int template_positions = span_y * W + span_x + 1; // why add 1?
    //int template_positions = (span_y - 1) * W + span_x; // More correct?

    /// @todo In old code, dst is buffer of size m_U. Could make it something like
    /// (span_x)x(span_y) instead?

    dst = Mat::zeros(H, W, CV_16U);
    dst_rgb = Mat::zeros(H, W, CV_16U);

    //ptr to dst matrix if we are using more than 63 features
    ushort* dst_ptr_ushort = dst.ptr<ushort>();
    //ptr to dst_rgb matrix if we are using more than 63 features
    ushort* dst_ptr_rgb_ushort = dst_rgb.ptr<ushort>();

//#if CV_SSE2
    volatile bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
    volatile bool haveSSE3 = checkHardwareSupport(CV_CPU_SSE3);
//#endif

    // Compute the similarity measure for this template by accumulating the contribution of
    // each feature

    int total_features_size;
    std::vector < cv::line_rgb::Feature > total_features;

    total_features_size = (int) templ.features.size();
    total_features.insert(total_features.end(), templ.features.begin(),
            templ.features.end());

    //we need this offset to know if we are evaluating the usual features or the signature features, in this case we only need to process the RGB part
    int offset_end_usual_features = total_features_size;

    if (color_features_enabled == true) {
        total_features_size += (int) templ.color_features.size();
        total_features.insert(total_features.end(),
                templ.color_features.begin(), templ.color_features.end());
    }

    for (int i = 0; i < total_features_size; ++i) {
        bool only_color_features = false;
        if (i >= offset_end_usual_features)
            only_color_features = true;

        // Add the linear memory at the appropriate offset computed from the location of
        // the feature in the template
        Feature f = total_features[i];
        // Discard feature if out of bounds

        /// @todo Shouldn't actually see x or y < 0 here?
        if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height)
            continue;

        const uchar* lm_ptr;
        if (only_color_features == false)
            lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        const uchar* lm_ptr_rgb = accessLinearMemoryRGB(linear_memories_rgb, f,
                T, W);

        // Now we do an aligned/unaligned add of dst_ptr and lm_ptr with template_positions elements
        int j = 0;

        // Process responses 8 at a time if vectorization possible
#if CV_SSE2
#if CV_SSE3
        if (haveSSE3)
        {
            // LDDQU may be more efficient than MOVDQU for unaligned load of next 16 responses
            for (; j < template_positions - 15; j += 16)
            {

                if(only_color_features == false)
                {
                    __m128i v_aligned = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(lm_ptr + j));
                    __m128i v_alignedLo = _mm_unpacklo_epi8(v_aligned, _mm_setzero_si128());
                    __m128i v_alignedHi = _mm_unpackhi_epi8(v_aligned, _mm_setzero_si128());

                    __m128i* dst_ptr_sse = reinterpret_cast<__m128i*>(dst_ptr_ushort + j);

                    dst_ptr_sse[0] = _mm_add_epi16(dst_ptr_sse[0], v_alignedLo);
                    dst_ptr_sse[1] = _mm_add_epi16(dst_ptr_sse[1], v_alignedHi);
                }

                __m128i v_aligned_rgb = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(lm_ptr_rgb + j));
                __m128i v_alignedLo_rgb = _mm_unpacklo_epi8(v_aligned_rgb, _mm_setzero_si128());
                __m128i v_alignedHi_rgb = _mm_unpackhi_epi8(v_aligned_rgb, _mm_setzero_si128());

                __m128i* dst_ptr_rgb_sse = reinterpret_cast<__m128i*>(dst_ptr_rgb_ushort + j);

                dst_ptr_rgb_sse[0] = _mm_add_epi16(dst_ptr_rgb_sse[0], v_alignedLo_rgb);
                dst_ptr_rgb_sse[1] = _mm_add_epi16(dst_ptr_rgb_sse[1], v_alignedHi_rgb);

            }
        }
        else
#endif
        if (haveSSE2)
        {
            // Fall back to MOVDQU
            for (; j < template_positions - 15; j += 16)
            {
                if(only_color_features == false)
                {
                    __m128i v_aligned = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lm_ptr + j));
                    __m128i v_alignedLo = _mm_unpacklo_epi8(v_aligned, _mm_setzero_si128());
                    __m128i v_alignedHi = _mm_unpackhi_epi8(v_aligned, _mm_setzero_si128());

                    __m128i* dst_ptr_sse = reinterpret_cast<__m128i*>(dst_ptr_ushort + j);

                    dst_ptr_sse[0] = _mm_add_epi16(dst_ptr_sse[0], v_alignedLo);
                    dst_ptr_sse[1] = _mm_add_epi16(dst_ptr_sse[1], v_alignedHi);
                }

                __m128i v_aligned_rgb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lm_ptr_rgb + j));
                __m128i v_alignedLo_rgb = _mm_unpacklo_epi8(v_aligned_rgb, _mm_setzero_si128());
                __m128i v_alignedHi_rgb = _mm_unpackhi_epi8(v_aligned_rgb, _mm_setzero_si128());

                __m128i* dst_ptr_rgb_sse = reinterpret_cast<__m128i*>(dst_ptr_rgb_ushort + j);

                dst_ptr_rgb_sse[0] = _mm_add_epi16(dst_ptr_rgb_sse[0], v_alignedLo_rgb);
                dst_ptr_rgb_sse[1] = _mm_add_epi16(dst_ptr_rgb_sse[1], v_alignedHi_rgb);

            }
        }
#endif

        for (; j < template_positions; ++j) {
            if (only_color_features == false)
                dst_ptr_ushort[j] += lm_ptr[j];
            dst_ptr_rgb_ushort[j] += lm_ptr_rgb[j];
        }

    }

}

/**
 * \brief Compute similarity measure for a given template in a local region.
 *
 * \param[in]  linear_memories Vector of 8 linear memories, one for each label.
 * \param[in]  templ           Template to match against.
 * \param[out] dst             Destination 8-bit similarity image, 16x16.
 * \param      size            Size (W, H) of the original input image.
 * \param      T               Sampling step.
 * \param      center          Center of the local region.
 */
void similarityLocal(const std::vector<Mat>& linear_memories,
        const Template& templ, Mat& dst, Size size, int T, Point center) {
    // Similar to whole-image similarity() above. This version takes a position 'center'
    // and computes the energy in the 16x16 patch centered on it.
    CV_Assert(templ.features.size() <= 63);

    // Compute the similarity map in a 16x16 patch around center
    int W = size.width / T;
    dst = Mat::zeros(16, 16, CV_8U);

    // Offset each feature point by the requested center. Further adjust to (-8,-8) from the
    // center to get the top-left corner of the 16x16 patch.
    // NOTE: We make the offsets multiples of T to agree with results of the original code.
    int offset_x = (center.x / T - 8) * T;
    int offset_y = (center.y / T - 8) * T;

#if CV_SSE2
    volatile bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
#if CV_SSE3
    volatile bool haveSSE3 = checkHardwareSupport(CV_CPU_SSE3);
#endif
    __m128i* dst_ptr_sse = dst.ptr<__m128i>();
#endif
//haveSSE2 = false;
//haveSSE3 = false;
    for (int i = 0; i < (int) templ.features.size(); ++i) {
        Feature f = templ.features[i];
        f.x += offset_x;
        f.y += offset_y;
        // Discard feature if out of bounds, possibly due to applying the offset
        if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
            continue;

        const uchar* lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        // Process whole row at a time if vectorization possible
#if CV_SSE2
#if CV_SSE3
        if (haveSSE3)
        {
            // LDDQU may be more efficient than MOVDQU for unaligned load of 16 responses from current row
            for (int row = 0; row < 16; ++row)
            {
                __m128i aligned = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(lm_ptr));
                dst_ptr_sse[row] = _mm_add_epi8(dst_ptr_sse[row], aligned);
                lm_ptr += W; // Step to next row
            }
        }
        else
#endif
        if (haveSSE2)
        {
            // Fall back to MOVDQU
            for (int row = 0; row < 16; ++row)
            {
                __m128i aligned = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lm_ptr));
                dst_ptr_sse[row] = _mm_add_epi8(dst_ptr_sse[row], aligned);
                lm_ptr += W; // Step to next row
            }
        }
        else
#endif
        {
            uchar* dst_ptr = dst.ptr<uchar>();
            for (int row = 0; row < 16; ++row) {
                for (int col = 0; col < 16; ++col)
                    dst_ptr[col] += lm_ptr[col];
                dst_ptr += 16;
                lm_ptr += W;
            }
        }
    }
}

/**
 * \brief Compute similarity measure for a given template in a local region.
 *
 * \param[in]  linear_memories     		Vector of 8 linear memories, one for each label.
 * \param[in]  linear_memories_rgb 		Vector of 8 rgb linear memories, one for each rgb label.
 * \param[in]  templ           	   		Template to match against.
 * \param[out] dst                 		Destination 8-bit similarity image, 16x16
 * \param[out] dst                 		Destination 8-bit similarity rgb image, 16x16
 * \param      size                		Size (W, H) of the original input image.
 * \param      T                   		Sampling step.
 * \param      color_features_enabled   Boolean to estabilish if we also need to process the color features.
 */
void similarityLocalRGB(const std::vector<Mat>& linear_memories,
        const std::vector<Mat>& linear_memories_rgb, const Template& templ,
        Mat& dst, Mat& dst_rgb, Size size, int T, Point center,
        const bool color_features_enabled) {

    // Similar to whole-image similarity() above. This version takes a position 'center'
    // and computes the energy in the 16x16 patch centered on it.

    //CV_Assert(templ.features.size() <= 63);

    // Compute the similarity map in a 16x16 patch around center
    int W = size.width / T;

    dst = Mat::zeros(16, 16, CV_16U);
    dst_rgb = Mat::zeros(16, 16, CV_16U);

    // Offset each feature point by the requested center. Further adjust to (-8,-8) from the
    // center to get the top-left corner of the 16x16 patch.
    // NOTE: We make the offsets multiples of T to agree with results of the original code.
    int offset_x = (center.x / T - 8) * T;
    int offset_y = (center.y / T - 8) * T;

//#if CV_SSE2
    volatile bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
    volatile bool haveSSE3 = checkHardwareSupport(CV_CPU_SSE3);
    //if(haveSSE2 || haveSSE3)
    __m128i* dst_ptr_sse = dst.ptr<__m128i>();
    __m128i* dst_ptr_rgb_sse = dst_rgb.ptr<__m128i>();
    haveSSE2 = false;
//haveSSE3 = false;
//#endif

    int total_features_size;
    std::vector < cv::line_rgb::Feature > total_features;

    total_features_size = (int) templ.features.size();
    total_features.insert(total_features.end(), templ.features.begin(),
            templ.features.end());
    //we need this offset to know if we are evaluating the usual features or the signature features, in this case we only need to process the RGB part
    int offset_end_usual_features = total_features_size;

    if (color_features_enabled == true) {
        total_features_size += (int) templ.color_features.size();
        total_features.insert(total_features.end(),
                templ.color_features.begin(), templ.color_features.end());
    }

    for (int i = 0; i < total_features_size; ++i) {
        bool only_color_features = false;
        if (i >= offset_end_usual_features)
            only_color_features = true;

        // Add the linear memory at the appropriate offset computed from the location of
        // the feature in the template
        Feature f = total_features[i];

        f.x += offset_x;
        f.y += offset_y;
        // Discard feature if out of bounds, possibly due to applying the offset
        if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
            continue;

        const uchar* lm_ptr;
        if (only_color_features == false)
            lm_ptr = accessLinearMemory(linear_memories, f, T, W);
        const uchar* lm_ptr_rgb = accessLinearMemoryRGB(linear_memories_rgb, f,
                T, W);

        // Process whole row at a time if vectorization possible

#if CV_SSE2
#if CV_SSE3
        if (haveSSE3)
        {
            // LDDQU may be more efficient than MOVDQU for unaligned load of 16 responses from current row
            for (int row = 0; row < 32; row+=2)
            {
                if(only_color_features == false)
                {
                    __m128i v_aligned = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(lm_ptr));
                    __m128i v_alignedLo = _mm_unpacklo_epi8(v_aligned, _mm_setzero_si128());
                    __m128i v_alignedHi = _mm_unpackhi_epi8(v_aligned, _mm_setzero_si128());

                    dst_ptr_sse[row] = _mm_add_epi16(dst_ptr_sse[row], v_alignedLo);
                    dst_ptr_sse[row + 1] = _mm_add_epi16(dst_ptr_sse[row + 1], v_alignedHi);

                    lm_ptr += W; // Step to next row
                }

                __m128i v_aligned_rgb = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(lm_ptr_rgb));
                __m128i v_alignedLo_rgb = _mm_unpacklo_epi8(v_aligned_rgb, _mm_setzero_si128());
                __m128i v_alignedHi_rgb = _mm_unpackhi_epi8(v_aligned_rgb, _mm_setzero_si128());

                dst_ptr_rgb_sse[row] = _mm_add_epi16(dst_ptr_rgb_sse[row], v_alignedLo_rgb);
                dst_ptr_rgb_sse[row + 1] = _mm_add_epi16(dst_ptr_rgb_sse[row + 1], v_alignedHi_rgb);

                lm_ptr_rgb += W; // Step to next row
            }

        }
        else
#endif
        if (haveSSE2)
        {

            // Fall back to MOVDQU
            for (int row = 0; row < 32; row+=2)
            {
                if(only_color_features == false)
                {
                    __m128i v_aligned = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lm_ptr));
                    __m128i v_alignedLo = _mm_unpacklo_epi8(v_aligned, _mm_setzero_si128());
                    __m128i v_alignedHi = _mm_unpackhi_epi8(v_aligned, _mm_setzero_si128());

                    dst_ptr_sse[row] = _mm_add_epi16(dst_ptr_sse[row], v_alignedLo);
                    dst_ptr_sse[row + 1] = _mm_add_epi16(dst_ptr_sse[row + 1], v_alignedHi);

                    lm_ptr += W; // Step to next row
                }

                __m128i v_aligned_rgb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lm_ptr_rgb));
                __m128i v_alignedLo_rgb = _mm_unpacklo_epi8(v_aligned_rgb, _mm_setzero_si128());
                __m128i v_alignedHi_rgb = _mm_unpackhi_epi8(v_aligned_rgb, _mm_setzero_si128());

                dst_ptr_rgb_sse[row] = _mm_add_epi16(dst_ptr_rgb_sse[row], v_alignedLo_rgb);
                dst_ptr_rgb_sse[row + 1] = _mm_add_epi16(dst_ptr_rgb_sse[row + 1], v_alignedHi_rgb);

                lm_ptr_rgb += W; // Step to next row
            }

        }
        else
#endif

        {

            if (only_color_features == false) {
                ushort* dst_ptr = dst.ptr<ushort>();
                for (int row = 0; row < 16; ++row) {
                    for (int col = 0; col < 16; ++col)
                        dst_ptr[col] += lm_ptr[col];

                    dst_ptr += 16;
                    lm_ptr += W;
                }
            }

            ushort* dst_ptr_rgb = dst_rgb.ptr<ushort>();
            for (int row = 0; row < 16; ++row) {
                for (int col = 0; col < 16; ++col)
                    dst_ptr_rgb[col] += lm_ptr_rgb[col];

                dst_ptr_rgb += 16;
                lm_ptr_rgb += W;
            }

        }

    }

}

void addUnaligned8u16u(const uchar * src1, const uchar * src2, ushort * res,
        int length) {
    const uchar * end = src1 + length;

    while (src1 != end) {
        *res = *src1 + *src2;

        ++src1;
        ++src2;
        ++res;
    }
}

/**
 * \brief Accumulate one or more 8-bit similarity images.
 *
 * \param[in]  similarities Source 8-bit similarity images.
 * \param[out] dst          Destination 16-bit similarity image.
 */
void addSimilarities(const std::vector<Mat>& similarities, Mat& dst) {

    if (similarities.size() == 1) {
        similarities[0].convertTo(dst, CV_16U);
    } else {
        // NOTE: add() seems to be rather slow in the 8U + 8U -> 16U case
        dst.create(similarities[0].size(), CV_16U);
        addUnaligned8u16u(similarities[0].ptr(), similarities[1].ptr(),
                dst.ptr<ushort>(), static_cast<int>(dst.total()));

        /// @todo Optimize 16u + 8u -> 16u when more than 2 modalities
        for (size_t i = 2; i < similarities.size(); ++i)
            add(dst, similarities[i], dst, noArray(), CV_16U);
    }
}

/****************************************************************************************\
*                               High-level Detector API                                  *
 \****************************************************************************************/

Detector::Detector() {
}

Detector::Detector(const std::vector<Ptr<Modality> >& modalities,
        const std::vector<int>& T_pyramid, const bool color_features_enabled) :
        modalities(modalities), pyramid_levels(
                static_cast<int>(T_pyramid.size())), T_at_level(T_pyramid), color_features_enabled(
                color_features_enabled) {
}

// Used to filter out weak matches
struct MatchPredicate {
    MatchPredicate(float threshold) :
            threshold(threshold) {
    }
    bool operator()(const Match& m) {
        return m.similarity < threshold;
    }
    float threshold;
};

// Used to filter out weak RGB matches 
struct MatchPredicateRGB {
    MatchPredicateRGB(float threshold) :
            threshold(threshold) {
    }
    bool operator()(const Match& m) {
        return m.similarity_rgb < threshold;
    }
    float threshold;
};

// Used to filter out weak combined score matches 
struct MatchPredicateCombined {
    MatchPredicateCombined(float threshold) :
            threshold(threshold) {
    }
    bool operator()(const Match& m) {
        return m.sim_combined < threshold;
    }
    float threshold;
};

struct RejectedPredicate {
    RejectedPredicate() {
    }
    bool operator()(const Match& m) {
        return m.rejected == 1;
    }
};

void Detector::match(const std::vector<Mat>& sources, float threshold,
        std::vector<Match>& matches, const std::vector<std::string>& class_ids,
        OutputArrayOfArrays quantized_images,
        const std::vector<Mat>& masks) const {
    matches.clear();
    if (quantized_images.needed())
        quantized_images.create(1,
                static_cast<int>(pyramid_levels * modalities.size()), CV_8U);

    assert(sources.size() == modalities.size());
    // Initialize each modality with our sources
    std::vector < Ptr<QuantizedPyramid> > quantizers;
    for (int i = 0; i < (int) modalities.size(); ++i) {
        Mat mask, source;
        source = sources[i];
        if (!masks.empty()) {
            assert(masks.size() == modalities.size());
            mask = masks[i];
        }
        assert(mask.empty() || mask.size() == source.size());
        quantizers.push_back(modalities[i]->process(source, mask));
    }
    // pyramid level -> modality -> quantization
    LinearMemoryPyramid lm_pyramid(pyramid_levels,
            std::vector < LinearMemories
                    > (modalities.size(), LinearMemories(8)));
    LinearMemoryPyramid lm_pyramid_rgb(pyramid_levels,
            std::vector < LinearMemories
                    > (modalities.size(), LinearMemories(8)));

    // For each pyramid level, precompute linear memories for each modality
    std::vector < Size > sizes;
    for (int l = 0; l < pyramid_levels; ++l) {
        int T = T_at_level[l];
        std::vector < LinearMemories > &lm_level = lm_pyramid[l];
        std::vector < LinearMemories > &lm_level_rgb = lm_pyramid_rgb[l];

        if (l > 0) {
            for (int i = 0; i < (int) quantizers.size(); ++i)
                quantizers[i]->pyrDown();
        }

        Mat quantized, spread_quantized, quantized_rgb, spread_quantized_rgb;
        std::vector < Mat > response_maps;
        std::vector < Mat > response_maps_rgb;
        for (int i = 0; i < (int) quantizers.size(); ++i) {
            quantizers[i]->quantize(quantized);
            quantizers[i]->quantizeRGB(quantized_rgb);
            spread(quantized, spread_quantized, T);
            spread(quantized_rgb, spread_quantized_rgb, T);

            computeResponseMaps(spread_quantized, response_maps);
            computeResponseMapsRGB(spread_quantized_rgb, response_maps_rgb);

            LinearMemories& memories = lm_level[i];
            LinearMemories& memories_rgb = lm_level_rgb[i];

            for (int j = 0; j < 8; ++j) {
                linearize(response_maps[j], memories[j], T);
                linearize(response_maps_rgb[j], memories_rgb[j], T);
            }

            if (quantized_images.needed()) //use copyTo here to side step reference semantics.
                quantized.copyTo(
                        quantized_images.getMatRef(
                                static_cast<int>(l * quantizers.size() + i)));
        }

        sizes.push_back(quantized.size());
    }

    if (class_ids.empty()) {
        // Match all templates
        TemplatesMap::const_iterator it = class_templates.begin(), itend =
                class_templates.end();
        for (; it != itend; ++it)
            matchClass(lm_pyramid, lm_pyramid_rgb, sizes, threshold, matches,
                    it->first, it->second);
    } else {
        // Match only templates for the requested class IDs
        for (int i = 0; i < (int) class_ids.size(); ++i) {
            TemplatesMap::const_iterator it = class_templates.find(
                    class_ids[i]);
            if (it != class_templates.end())
                matchClass(lm_pyramid, lm_pyramid_rgb, sizes, threshold,
                        matches, it->first, it->second);
        }
    }

    QuantizedPyramid * tmpQp = quantizers[0];
    ColorGradientPyramid * cgp = dynamic_cast<ColorGradientPyramid *>(tmpQp);

    // Sort matches by similarity, and prune any duplicates introduced by pyramid refinement
    std::sort(matches.begin(), matches.end());
    std::vector<Match>::iterator new_end = std::unique(matches.begin(),
            matches.end());
    matches.erase(new_end, matches.end());

    if (matches.size() > 0) {

        for (int m = 0; m < modalities.size(); ++m) {
            int not_rejected = 0;

            for (std::vector<Match>::iterator it_rejected = matches.begin();
                    it_rejected != matches.end();) {
                const std::vector<cv::line_rgb::Template>& templates =
                        getTemplates((*it_rejected).class_id,
                                (*it_rejected).template_id);

                cv::Point offset = cv::Point((*it_rejected).x,
                        (*it_rejected).y);
                cv::Point offset_end = cv::Point(
                        (*it_rejected).x + templates[0].width,
                        (*it_rejected).y + templates[0].height);

                Mat magnitude_matched = cgp->magnitude_strong(
                        Rect((*it_rejected).x, (*it_rejected).y,
                                templates[0].width, templates[0].height));

                Scalar colorM = Scalar(255, 0, 0);
                vector < vector<Point> > contours;
                contours.push_back(templates[0].contour);
                drawContours(magnitude_matched, contours, 0, colorM);

                int count_magnitude = 0;
                for (int r = 0; r < magnitude_matched.rows; ++r) {
                    float* source_r = magnitude_matched.ptr<float>(r);
                    for (int c = 0; c < magnitude_matched.cols; ++c) {
                        if (source_r[c] > 0) {
                            double res = pointPolygonTest(contours[0],
                                    Point(c, r), true);
                            if (res >= 0)
                                count_magnitude++;
                        }
                    }
                }

                int diff_features = count_magnitude
                        - templates[0].total_candidates;
                int allowed_more_mag = 1;
                int thresh_allow = (templates[0].total_candidates / 100)
                        * allowed_more_mag;
                int impact = 0;
                if (diff_features > 0)
                    impact = diff_features / thresh_allow;

                float similarity_adjusted = (*it_rejected).sim_combined
                        - (float) impact;

                if (similarity_adjusted < threshold) {
                    (*it_rejected).rejected = 1;
                    it_rejected++;
                } else {
                    it_rejected++;
                }

            }

            //erase the rejected matches
            std::vector<Match>::iterator new_begin = matches.begin();
            std::vector<Match>::iterator new_end_rejected = std::remove_if(
                    new_begin, matches.end(), RejectedPredicate());
            matches.erase(new_end_rejected, matches.end());

        }
    }

}

void Detector::matchClass(const LinearMemoryPyramid& lm_pyramid,
        const LinearMemoryPyramid& lm_pyramid_rgb,
        const std::vector<Size>& sizes, float threshold,
        std::vector<Match>& matches, const std::string& class_id,
        const std::vector<TemplatePyramid>& template_pyramids) const {

    Timer extract_timer;
    extract_timer.start();
    // For each template...
    for (size_t template_id = 0; template_id < template_pyramids.size();
            ++template_id) {
        const TemplatePyramid& tp = template_pyramids[template_id];

        // First match over the whole image at the lowest pyramid level
        /// @todo Factor this out into separate function
        const std::vector<LinearMemories>& lowest_lm = lm_pyramid.back();
        const std::vector<LinearMemories>& lowest_lm_rgb =
                lm_pyramid_rgb.back();

        // Compute similarity maps for each modality at lowest pyramid level
        std::vector < Mat > similarities(modalities.size());
        std::vector < Mat > similarities_rgb(1);
        int lowest_start = static_cast<int>(tp.size() - modalities.size());
        int lowest_T = T_at_level.back();
        int num_features = 0;
        int num_color_features = 0;
        for (int i = 0; i < (int) modalities.size(); ++i) {

            const Template& templ = tp[lowest_start + i];
            num_features += static_cast<int>(templ.features.size());
            num_color_features += static_cast<int>(templ.color_features.size());
            if (i == 0) //color modality, add rgb
                similarityRGB(lowest_lm[i], lowest_lm_rgb[i], templ,
                        similarities[i], similarities_rgb[i], sizes.back(),
                        lowest_T, color_features_enabled);
            if (i == 1) //depth modality, without rgb
                similarity(lowest_lm[i], templ, similarities[i], sizes.back(),
                        lowest_T);

        }

        // Combine into overall similarity
        /// @todo Support weighting the modalities
        Mat total_similarity;
        addSimilarities(similarities, total_similarity);
        Mat total_similarity_rgb;
        addSimilarities(similarities_rgb, total_similarity_rgb);

        // Convert user-friendly percentage to raw similarity threshold. The percentage
        // threshold scales from half the max response (what you would expect from applying
        // the template to a completely random image) to the max response.
        // NOTE: This assumes max per-feature response is 4, so we scale between [2*nf, 4*nf].
        int raw_threshold = static_cast<int>(2 * num_features
                + (threshold / 100.f) * (2 * num_features) + 0.5f);
        int raw_threshold_rgb = static_cast<int>(8 * num_features
                + (threshold / 100.f) * (8 * num_features) + 0.5f);

        // Find initial matches
        std::vector < Match > candidates;
        for (int r = 0; r < total_similarity.rows; ++r) {
            ushort* row = total_similarity.ptr < ushort > (r);
            ushort* row_rgb = total_similarity_rgb.ptr < ushort > (r);

            for (int c = 0; c < total_similarity.cols; ++c) {
                int raw_score = row[c];
                int raw_score_rgb = row_rgb[c];    //*modalities.size();

                int raw_score_combined = (raw_score + raw_score_rgb) / 2;

                if (raw_score > raw_threshold) {
                    int offset = lowest_T / 2 + (lowest_T % 2 - 1);
                    int x = c * lowest_T + offset;
                    int y = r * lowest_T + offset;
                    float score = (raw_score_combined * 100.f)
                            / (4 * num_features) + 0.5f;
                    float score_rgb;

                    if (color_features_enabled == false)
                        score_rgb = (raw_score_rgb * 100.f)
                                / (16 * num_features) + 0.5f;
                    else
                        score_rgb = (raw_score_rgb * 100.f)
                                / (16 * (num_features + num_color_features))
                                + 0.5f;

                    candidates.push_back(
                            Match(x, y, 0, score, score_rgb, class_id,
                                    static_cast<int>(template_id), 0));
                }
            }
        }

        // Locally refine each match by marching up the pyramid
        for (int l = pyramid_levels - 2; l >= 0; --l) {

            const std::vector<LinearMemories>& lms = lm_pyramid[l];
            const std::vector<LinearMemories>& lms_rgb = lm_pyramid_rgb[l];

            int T = T_at_level[l];
            int start = static_cast<int>(l * modalities.size());
            Size size = sizes[l];
            int border = 8 * T;
            int offset = T / 2 + (T % 2 - 1);
            int max_x = size.width - tp[start].width - border;
            int max_y = size.height - tp[start].height - border;

            //local redefinition
            std::vector < Mat > similarities(modalities.size());
            std::vector < Mat > similarities_rgb(1);
            Mat total_similarity;
            Mat total_similarity_rgb;

            for (int m = 0; m < (int) candidates.size(); ++m) {
                Match& match = candidates[m];
                int x = match.x * 2 + 1; /// @todo Support other pyramid distance
                int y = match.y * 2 + 1;

                // Require 8 (reduced) row/cols to the up/left
                x = std::max(x, border);
                y = std::max(y, border);

                // Require 8 (reduced) row/cols to the down/left, plus the template size
                x = std::min(x, max_x);
                y = std::min(y, max_y);

                // Compute local similarity maps for each modality
                int num_features = 0;
                int num_color_features = 0;
                for (int i = 0; i < (int) modalities.size(); ++i) {
                    const Template& templ = tp[start + i];
                    num_features += static_cast<int>(templ.features.size());
                    num_color_features +=
                            static_cast<int>(templ.color_features.size());

                    if (i == 0) //color modality, add rgb
                        similarityLocalRGB(lms[i], lms_rgb[i], templ,
                                similarities[i], similarities_rgb[i], size, T,
                                Point(x, y), color_features_enabled);
                    if (i == 1) //color modality, without rgb
                        similarityLocal(lms[i], templ, similarities[i], size, T,
                                Point(x, y));

                }

                addSimilarities(similarities, total_similarity);
                addSimilarities(similarities_rgb, total_similarity_rgb);

                // Find best local adjustment
                int best_score = 0;
                int best_score_rgb = 0;
                int best_score_combined = 0;
                int best_r = -1, best_c = -1;
                for (int r = 0; r < total_similarity.rows; ++r) {
                    ushort* row = total_similarity.ptr < ushort > (r);
                    ushort* row_rgb = total_similarity_rgb.ptr < ushort > (r);
                    for (int c = 0; c < total_similarity.cols; ++c) {
                        int score = row[c];
                        int score_rgb = row_rgb[c] * modalities.size();

                        score = (score * 100.f) / (4 * num_features);

                        if (color_features_enabled == false)
                            score_rgb = (score_rgb * 100.f)
                                    / (16 * num_features) + 0.5f;
                        else
                            score_rgb = (score_rgb * 100.f)
                                    / (16 * (num_features + num_color_features))
                                    + 0.5f;

                        int score_combined = (score + score_rgb) / 2;

                        if (score_combined > best_score_combined) {
                            best_score_combined = score_combined;
                            best_score = score;
                            best_score_rgb = score_rgb;
                            best_r = r;
                            best_c = c;
                        }

                    }
                }
                // Update current match
                match.x = (x / T - 8 + best_c) * T + offset;
                match.y = (y / T - 8 + best_r) * T + offset;

                match.similarity = best_score;
                match.similarity_rgb = best_score_rgb;
                match.sim_combined = best_score_combined;
                match.width = tp[start].width;
                match.height = tp[start].height;
            }

            // Filter out any matches that drop below the similarity threshold
            std::vector<Match>::iterator new_end;
            std::vector<Match>::iterator new_end_rgb;

            new_end = std::remove_if(candidates.begin(), candidates.end(),
                    MatchPredicate(threshold));
            candidates.erase(new_end, candidates.end());

            new_end_rgb = std::remove_if(candidates.begin(), candidates.end(),
                    MatchPredicateRGB(threshold));
            candidates.erase(new_end_rgb, candidates.end());

        }

        matches.insert(matches.end(), candidates.begin(), candidates.end());
    }
    extract_timer.stop();

}

int Detector::addTemplate(const std::vector<Mat>& sources,
        const std::string& class_id, const Mat& object_mask,
        Rect* bounding_box) {
    int num_modalities = static_cast<int>(modalities.size());
    std::vector < TemplatePyramid > &template_pyramids =
            class_templates[class_id];
    int template_id = static_cast<int>(template_pyramids.size());

    TemplatePyramid tp;
    tp.resize(num_modalities * pyramid_levels);
    // For each modality...
    for (int i = 0; i < num_modalities; ++i) {
        // Extract a template at each pyramid level
        Ptr < QuantizedPyramid > qp = modalities[i]->process(sources[i],
                object_mask);

        for (int l = 0; l < pyramid_levels; ++l) {
            /// @todo Could do mask subsampling here instead of in pyrDown()
            if (l > 0) {
                qp->pyrDown();

            }
            bool success = qp->extractTemplate(tp[l * num_modalities + i]);

            if (!success)
                return -1;
        }
    }

    Rect bb = cropTemplates(tp, object_mask);
    if (bounding_box)
        *bounding_box = bb;

    /// @todo Can probably avoid a copy of tp here with swap
    template_pyramids.push_back(tp);
    return template_id;
}

int Detector::addSyntheticTemplate(const std::vector<Template>& templates,
        const std::string& class_id) {
    std::vector < TemplatePyramid > &template_pyramids =
            class_templates[class_id];
    int template_id = static_cast<int>(template_pyramids.size());
    template_pyramids.push_back(templates);
    return template_id;
}

const std::vector<Template>& Detector::getTemplates(const std::string& class_id,
        int template_id) const {
    TemplatesMap::const_iterator i = class_templates.find(class_id);
    CV_Assert(i != class_templates.end());
    CV_Assert(i->second.size() > size_t(template_id));
    return i->second[template_id];
}

int Detector::numTemplates() const {
    int ret = 0;
    TemplatesMap::const_iterator i = class_templates.begin(), iend =
            class_templates.end();
    for (; i != iend; ++i)
        ret += static_cast<int>(i->second.size());
    return ret;
}

int Detector::numTemplates(const std::string& class_id) const {
    TemplatesMap::const_iterator i = class_templates.find(class_id);
    if (i == class_templates.end())
        return 0;
    return static_cast<int>(i->second.size());
}

std::vector<std::string> Detector::classIds() const {
    std::vector < std::string > ids;
    TemplatesMap::const_iterator i = class_templates.begin(), iend =
            class_templates.end();
    for (; i != iend; ++i) {
        ids.push_back(i->first);
    }

    return ids;
}

void Detector::read(const FileNode& fn) {
    class_templates.clear();
    pyramid_levels = fn["pyramid_levels"];
    fn["T"] >> T_at_level;

    modalities.clear();
    FileNode modalities_fn = fn["modalities"];
    FileNodeIterator it = modalities_fn.begin(), it_end = modalities_fn.end();
    for (; it != it_end; ++it) {
        modalities.push_back(Modality::create(*it));
    }
}

void Detector::write(FileStorage& fs) const {
    fs << "pyramid_levels" << pyramid_levels;
    fs << "T" << T_at_level;

    fs << "modalities" << "[";
    for (int i = 0; i < (int) modalities.size(); ++i) {
        fs << "{";
        modalities[i]->write(fs);
        fs << "}";
    }
    fs << "]"; // modalities
}

std::string Detector::readClass(const FileNode& fn,
        const std::string &class_id_override) {
    // Verify compatible with Detector settings
    FileNode mod_fn = fn["modalities"];
    CV_Assert(mod_fn.size() == modalities.size());
    FileNodeIterator mod_it = mod_fn.begin(), mod_it_end = mod_fn.end();
    int i = 0;
    for (; mod_it != mod_it_end; ++mod_it, ++i)
        CV_Assert(modalities[i]->name() == (std::string)(*mod_it));
    CV_Assert((int) fn["pyramid_levels"] == pyramid_levels);

    // Detector should not already have this class
    std::string class_id;
    if (class_id_override.empty()) {
        std::string class_id_tmp = fn["class_id"];
        CV_Assert(class_templates.find(class_id_tmp) == class_templates.end());
        class_id = class_id_tmp;
    } else {
        class_id = class_id_override;
    }

    TemplatesMap::value_type v(class_id, std::vector<TemplatePyramid>());
    std::vector < TemplatePyramid > &tps = v.second;
    int expected_id = 0;

    FileNode tps_fn = fn["template_pyramids"];
    tps.resize(tps_fn.size());
    FileNodeIterator tps_it = tps_fn.begin(), tps_it_end = tps_fn.end();
    for (; tps_it != tps_it_end; ++tps_it, ++expected_id) {
        int template_id = (*tps_it)["template_id"];
        CV_Assert(template_id == expected_id);
        FileNode templates_fn = (*tps_it)["templates"];
        tps[template_id].resize(templates_fn.size());

        FileNodeIterator templ_it = templates_fn.begin(), templ_it_end =
                templates_fn.end();
        int i = 0;
        for (; templ_it != templ_it_end; ++templ_it) {
            tps[template_id][i++].read(*templ_it);
        }
    }

    class_templates.insert(v);
    return class_id;
}

void Detector::writeClass(const std::string& class_id, FileStorage& fs) const {
    TemplatesMap::const_iterator it = class_templates.find(class_id);
    CV_Assert(it != class_templates.end());
    const std::vector<TemplatePyramid>& tps = it->second;

    fs << "class_id" << it->first;
    fs << "modalities" << "[:";
    for (size_t i = 0; i < modalities.size(); ++i)
        fs << modalities[i]->name();
    fs << "]"; // modalities
    fs << "pyramid_levels" << pyramid_levels;
    fs << "template_pyramids" << "[";
    for (size_t i = 0; i < tps.size(); ++i) {
        const TemplatePyramid& tp = tps[i];
        fs << "{";
        fs << "template_id" << int(i); //TODO is this cast correct? won't be good if rolls over...
        fs << "templates" << "[";
        for (size_t j = 0; j < tp.size(); ++j) {
            fs << "{";
            tp[j].write(fs);
            fs << "}"; // current template
        }
        fs << "]"; // templates
        fs << "}"; // current pyramid
    }
    fs << "]"; // pyramids
}

void Detector::readClasses(const std::vector<std::string>& class_ids,
        const std::string& format) {
    for (size_t i = 0; i < class_ids.size(); ++i) {
        const std::string& class_id = class_ids[i];
        std::string filename = cv::format(format.c_str(), class_id.c_str());
        FileStorage fs(filename, FileStorage::READ);
        readClass(fs.root());
    }
}

void Detector::writeClasses(const std::string& format) const {
    TemplatesMap::const_iterator it = class_templates.begin(), it_end =
            class_templates.end();
    for (; it != it_end; ++it) {
        const std::string& class_id = it->first;
        std::string filename = cv::format(format.c_str(), class_id.c_str());
        FileStorage fs(filename, FileStorage::WRITE);
        writeClass(class_id, fs);
    }
}

static const int T_DEFAULTS[] = { 5, 8 };

Ptr<Detector> getDefaultLINERGB(const bool color_features_enabled) {
    std::vector < Ptr<Modality> > modalities;
    modalities.push_back(new ColorGradient);
    return new Detector(modalities,
            std::vector<int>(T_DEFAULTS, T_DEFAULTS + 2),
            color_features_enabled);
}

Ptr<Detector> getDefaultLINEMODRGB(const bool color_features_enabled) {
    std::cout << std::endl << std::endl
            << "Still not implemented --- returned a defaultLINERGB detector"
            << std::endl << std::endl << std::endl;
    std::vector < Ptr<Modality> > modalities;
    modalities.push_back(new ColorGradient);
    //modalities.push_back(new DepthNormal);
    return new Detector(modalities,
            std::vector<int>(T_DEFAULTS, T_DEFAULTS + 2),
            color_features_enabled);
}

} // namespace line_rgb
} // namespace cv
