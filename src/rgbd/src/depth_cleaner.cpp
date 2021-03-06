/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/rgbd/rgbd.hpp>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace
{
  class DepthCleanerImpl
  {
  public:
    DepthCleanerImpl(int window_size, int depth, cv::DepthCleaner::DEPTH_CLEANER_METHOD method)
        :
          depth_(depth),
          window_size_(window_size),
          method_(method)
    {
    }

    virtual
    ~DepthCleanerImpl()
    {
    }

    virtual void
    cache()=0;

    bool
    validate(int depth, int window_size, int method) const
    {
      return (window_size == window_size_) && (depth == depth_) && (method == method_);
    }
  protected:
    int depth_;
    int window_size_;
    cv::DepthCleaner::DEPTH_CLEANER_METHOD method_;
  };
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace
{
  /** Given a depth image, compute the normals as detailed in the LINEMOD paper
   * ``Gradient Response Maps for Real-Time Detection of Texture-Less Objects``
   * by S. Hinterstoisser, C. Cagniart, S. Ilic, P. Sturm, N. Navab, P. Fua, and V. Lepetit
   */
  template<typename T>
  class NIL: public DepthCleanerImpl
  {
  public:
    typedef cv::Vec<T, 3> Vec3T;
    typedef cv::Matx<T, 3, 3> Mat33T;

    NIL(int window_size, int depth, cv::DepthCleaner::DEPTH_CLEANER_METHOD method)
        :
          DepthCleanerImpl(window_size, depth, method)
    {
    }

    /** Compute cached data
     */
    virtual void
    cache()
    {
    }

    /** Compute the normals
     * @param r
     * @return
     */
    cv::Mat
    compute(const cv::Mat& depth_in) const
    {
      switch (depth_in.depth())
      {
        case CV_16U:
        {
          const cv::Mat_<unsigned short> &depth(depth_in);
          cv::Mat depth_out = computeImpl<unsigned short, float>(depth, 0.001);
          cv::Mat depth_out_typed;
          depth_out.convertTo(depth_out_typed, CV_16U);
          return depth_out_typed;
          break;
        }
        case CV_32F:
        {
          const cv::Mat_<float> &depth(depth_in);
          return computeImpl<float, float>(depth, 1);
          break;
        }
        case CV_64F:
        {
          const cv::Mat_<double> &depth(depth_in);
          return computeImpl<double, double>(depth, 1);
          break;
        }
      }
      return cv::Mat();
    }

  private:
    /** Compute the normals
     * @param r
     * @return
     */
    template<typename DepthDepth, typename ContainerDepth>
    cv::Mat
    computeImpl(const cv::Mat_<DepthDepth> &depth_in, ContainerDepth scale) const
    {
      const ContainerDepth theta_mean = 30. * CV_PI / 180;
      int rows = depth_in.rows;
      int cols = depth_in.cols;

      // Precompute some data
      const ContainerDepth sigma_L = 0.8 + 0.035 * theta_mean / (CV_PI / 2 - theta_mean);
      cv::Mat_<ContainerDepth> sigma_z(rows, cols);
      for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
          sigma_z(y, x) = 0.0012 + 0.0019 * (depth_in(y, x) * scale - 0.4) * (depth_in(y, x) * scale - 0.4);

      ContainerDepth difference_threshold = 10;
      cv::Mat_<ContainerDepth> Dw_sum = cv::Mat_<ContainerDepth>::zeros(rows, cols), w_sum =
          cv::Mat_<ContainerDepth>::zeros(rows, cols);
      for (int y = 0; y < rows - 1; ++y)
      {
        // Every pixel has had the contribution of previous pixels (in a row-major way)
        for (int x = 1; x < cols - 1; ++x)
        {
          for (int j = 0; j <= 1; ++j)
            for (int i = -1; i <= 1; ++i)
            {
              if ((j == 0) && (i == -1))
                continue;
              ContainerDepth delta_u = sqrt(
                  ContainerDepth(j) * ContainerDepth(j) + ContainerDepth(i) * ContainerDepth(i));
              ContainerDepth delta_z;
              if (depth_in(y, x) > depth_in(y + j, x + i))
                delta_z = depth_in(y, x) - depth_in(y + j, x + i);
              else
                delta_z = depth_in(y + j, x + i) - depth_in(y, x);
              if (delta_z < difference_threshold)
              {
                delta_z *= scale;
                ContainerDepth w = exp(
                    -delta_u * delta_u / 2 / sigma_L / sigma_L - delta_z * delta_z / 2 / sigma_z(y, x) / sigma_z(y, x));
                w_sum(y, x) += w;
                Dw_sum(y, x) += depth_in(y + j, x + i) * w;
                if ((j != 0) || (i != 0))
                {
                  w = exp(
                      -delta_u * delta_u / 2 / sigma_L / sigma_L - delta_z * delta_z / 2 / sigma_z(y + j, x + i)
                                                                   / sigma_z(y + j, x + i));
                  w_sum(y + j, x + i) += w;
                  Dw_sum(y + j, x + i) += depth_in(y, x) * w;
                }
              }
            }
        }
      }
      cv::Mat_<ContainerDepth> depth_out = Dw_sum / w_sum;
      std::cout << depth_in(200, 100) << " " << depth_out(200, 100) << " - " << std::endl;

      return depth_out;
    }
  };
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cv
{
  /** Default constructor of the Algorithm class that computes normals
   */
  DepthCleaner::DepthCleaner(int depth, int window_size, int method)
      :
        depth_(depth),
        window_size_(window_size),
        method_(method),
        depth_cleaner_impl_(0)
  {
    CV_Assert(depth == CV_16U || depth == CV_32F || depth == CV_64F);
  }

  /** Destructor
   */
  DepthCleaner::~DepthCleaner()
  {
    if (depth_cleaner_impl_ == 0)
      return;
    switch (method_)
    {
      case DEPTH_CLEANER_NIL:
      {
        switch (depth_)
        {
          case CV_16U:
            delete reinterpret_cast<const NIL<unsigned short> *>(depth_cleaner_impl_);
            break;
          case CV_32F:
            delete reinterpret_cast<const NIL<float> *>(depth_cleaner_impl_);
            break;
          case CV_64F:
            delete reinterpret_cast<const NIL<double> *>(depth_cleaner_impl_);
            break;
        }
        break;
      }
    }
  }

  void
  DepthCleaner::initialize_cleaner_impl() const
  {
    CV_Assert(depth_ == CV_16U || depth_ == CV_32F || depth_ == CV_64F);
    CV_Assert(window_size_ == 1 || window_size_ == 3 || window_size_ == 5 || window_size_ == 7);
    CV_Assert( method_ == DEPTH_CLEANER_NIL);
    switch (method_)
    {
      case (DEPTH_CLEANER_NIL):
      {
        switch (depth_)
        {
          case CV_16U:
            depth_cleaner_impl_ = new NIL<unsigned short>(window_size_, depth_, DEPTH_CLEANER_NIL);
            break;
          case CV_32F:
            depth_cleaner_impl_ = new NIL<float>(window_size_, depth_, DEPTH_CLEANER_NIL);
            break;
          case CV_64F:
            depth_cleaner_impl_ = new NIL<double>(window_size_, depth_, DEPTH_CLEANER_NIL);
            break;
        }
        break;
      }
    }

    reinterpret_cast<DepthCleanerImpl *>(depth_cleaner_impl_)->cache();
  }

  /** Initializes some data that is cached for later computation
   * If that function is not called, it will be called the first time normals are computed
   */
  void
  DepthCleaner::initialize() const
  {
    if (depth_cleaner_impl_ == 0)
      initialize_cleaner_impl();
    else if (!reinterpret_cast<DepthCleanerImpl *>(depth_cleaner_impl_)->validate(depth_, window_size_, method_))
      initialize_cleaner_impl();
  }

  /** Given a set of 3d points in a depth image, compute the normals at each point
   * using the SRI method described in
   * ``Fast and Accurate Computation of Surface Normals from Range Images``
   * by H. Badino, D. Huber, Y. Park and T. Kanade
   * @param depth depth a float depth image. Or it can be rows x cols x 3 is they are 3d points
   * @param window_size the window size on which to compute the derivatives
   * @return normals a rows x cols x 3 matrix
   */
  cv::Mat
  DepthCleaner::operator()(const cv::Mat &depth) const
  {
    CV_Assert(depth.dims == 2);
    CV_Assert(depth.channels() == 1);

    // Initialize the pimpl
    initialize();

    // Clean the depth
    cv::Mat depth_out;
    switch (method_)
    {
      case (DEPTH_CLEANER_NIL):
      {
        switch (depth_)
        {
          case CV_16U:
            depth_out = reinterpret_cast<const NIL<unsigned short> *>(depth_cleaner_impl_)->compute(depth);
            break;
          case CV_32F:
            depth_out = reinterpret_cast<const NIL<float> *>(depth_cleaner_impl_)->compute(depth);
            break;
          case CV_64F:
            depth_out = reinterpret_cast<const NIL<double> *>(depth_cleaner_impl_)->compute(depth);
            break;
        }
        break;
      }
    }

    return depth_out;
  }
}
