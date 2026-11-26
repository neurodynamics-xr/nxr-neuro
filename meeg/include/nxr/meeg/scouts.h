// nxr::meeg — A REGION'S SIGNAL AS THE FIRST PRINCIPAL COMPONENT OF ITS SOURCES (Brainstorm's scout function PCA with the sign flip, bst_scout_value of the release of 2018-10-12,
// written from its documented behaviour; Brainstorm is GPL, its code is run only to make the goldens). Pure: arrays in, arrays out; Eigen only.
//
// THE REFERENCE, on a region's source series F [n x T] over a span (the region's rows of a fixed-orientation kernel applied to the recording):
//   1. sign flip: the dominant orientation of the region's sources (the first right singular vector of their orientations [n x 3]); each series is multiplied by the sign of its
//      source's projection on it, the signs negated when fewer are positive than negative (the fewest series flipped): D = diag(flip);
//   2. Fc = D F with each row's mean over the span removed; u1 the first left singular vector of Fc (the leading eigenvector of the region's covariance D K_R C K_R^T D);
//   3. the component's sign from the data: the row `peak` where |u1| is largest; the sign of Fc(peak, t) at the sample where |Fc(peak, .)| is largest, times the sign of the
//      component u1^T Fc at the sample where it is largest in magnitude;
//   4. y(t) = sign x u1^T Fc(t) + mean(D F)(peak): the component, signed, plus the peak source's mean.
//
// AS A LINEAR MAP: the kernel being linear, F = K_R x. Step 2 needs only the channels' covariance C over the span (its mean removed), so u1 is computed from K_R C K_R^T, never
// from the region's series (`regionComponent`). Step 3 needs four extremes over the span, of two linear functionals of x (the component z = u1^T D K_R x and the peak source
// p = (D K_R)_peak x) and their means: a measurement over the span (`regionComponentSign`). Then y = s z - s mean(z) + mean(p): a row of channel weights and an offset.
//
// What it cannot resolve: the component is the direction of most variance within the region over THIS span, so it can change between spans and between people; it captures what
// its share (`explained`) says and no more; its sign is a convention set by the data's extremes.
#pragma once
#include <Eigen/Dense>
#include <cstdint>

namespace nxr::meeg {

/** Brainstorm's scout sign flip: +1 or -1 per source of a region from its orientations [n x 3] (0 for a source exactly orthogonal to the dominant orientation, as sign() gives);
 *  all +1 for a region of one source. std::invalid_argument for a non-finite orientation or the wrong number of columns. */
Eigen::VectorXd scoutSignFlip(const Eigen::MatrixXd& orientations);

struct RegionComponent {
  Eigen::RowVectorXd component;     // [M] u1^T D K_R: the region's first principal component as channel weights (its sign is set by regionComponentSign)
  Eigen::RowVectorXd peak;          // [M] (D K_R)_peak: the flipped kernel row of the source where |u1| is largest
  std::int64_t peakSource = 0;      // that source's index within the region (the first where several are equal)
  double explained = 0;             // s_1 / sum s: the share of the region's centred series the component carries (singular values: the square roots of the eigenvalues), Brainstorm's
};
/** The component of a region from its kernel rows K_R [n x M], the channels' covariance over the span C [M x M] (its mean removed; any positive scale) and the flip [n]. The
 *  leading eigenvector is of D K_R C K_R^T D, the region's own covariance, never formed for the whole cortex. std::invalid_argument on a shape mismatch, or a region whose
 *  series have no variance. */
RegionComponent regionComponent(const Eigen::MatrixXd& kernelRows, const Eigen::MatrixXd& covariance, const Eigen::VectorXd& flip);

/** The sign the reference gives the component from the span's measurements of the peak source's series (its maximum, minimum and mean) and of the component's: each factor is
 *  +1 when the series' largest excursion from its mean is upward, -1 when downward (+1 when the two are equal: the reference then takes the earlier sample, which a measurement
 *  does not keep). */
double regionComponentSign(double peakMax, double peakMin, double peakMean, double componentMax, double componentMin, double componentMean);

}  // namespace nxr::meeg
