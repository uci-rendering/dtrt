#pragma once
#ifndef PMF_H__
#define PMF_H__

#include "fwd.h"
#include <vector>

class DiscreteDistribution {
public:
    /// Allocate memory for a distribution with the given number of entries
    explicit inline DiscreteDistribution(size_t nEntries = 0) {
        reserve(nEntries);
        clear();
    }

    inline DiscreteDistribution(const DiscreteDistribution &distrb)
        : m_cdf(distrb.m_cdf), m_sum(distrb.m_sum), m_normalization(distrb.m_normalization)
        , m_normalized(distrb.m_normalized)
    {}

    /// Clear all entries
    inline void clear() {
        m_cdf.clear();
        m_cdf.push_back(0.0f);
        m_sum = m_normalization = 0.0f;
        m_normalized = false;
    }

    /// Reserve memory for a certain number of entries
    inline void reserve(size_t nEntries) {
        m_cdf.reserve(nEntries+1);
    }

    /// Append an entry with the specified discrete probability
    inline void append(Float pdfValue) {
        m_cdf.push_back(m_cdf[m_cdf.size()-1] + pdfValue);
    }

    /// Return the number of entries so far
    inline size_t size() const {
        return m_cdf.size()-1;
    }

    /// Access an entry by its index
    inline Float operator[](size_t entry) const {
        return m_cdf[entry+1] - m_cdf[entry];
    }

    /// Have the probability densities been normalized?
    inline bool isNormalized() const {
        return m_normalized;
    }

    /**
     * \brief Return the original (unnormalized) sum of all PDF entries
     *
     * This assumes that \ref normalize() has previously been called
     */
    inline Float getSum() const {
        return m_sum;
    }

    /**
     * \brief Return the normalization factor (i.e. the inverse of \ref getSum())
     *
     * This assumes that \ref normalize() has previously been called
     */
    inline Float getNormalization() const {
        return m_normalization;
    }

    /**
     * \brief Normalize the distribution
     *
     * Throws an exception when no entries were previously
     * added to the distribution.
     *
     * \return Sum of the (previously unnormalized) entries
     */
    inline Float normalize() {
        assert(m_cdf.size() > 1);
        m_sum = m_cdf[m_cdf.size()-1];
        if (m_sum > 0) {
            m_normalization = 1.0f / m_sum;
            for (size_t i=1; i<m_cdf.size(); ++i)
                m_cdf[i] *= m_normalization;
            m_cdf[m_cdf.size()-1] = 1.0f;
            m_normalized = true;
        } else {
            m_normalization = 0.0f;
        }
        return m_sum;
    }

    /**
     * \brief %Transform a uniformly distributed sample to the stored distribution
     *
     * \param[in] sampleValue
     *     An uniformly distributed sample on [0,1]
     * \return
     *     The discrete index associated with the sample
     */
    inline size_t sample(Float sampleValue) const {
        std::vector<Float>::const_iterator entry =
                std::lower_bound(m_cdf.begin(), m_cdf.end(), sampleValue);
        size_t index = std::min(m_cdf.size()-2,
            (size_t) std::max((ptrdiff_t) 0, entry - m_cdf.begin() - 1));

        /* Handle a rare corner-case where a entry has probability 0
           but is sampled nonetheless */
        while (operator[](index) == 0 && index < m_cdf.size()-1)
            ++index;

        return index;
    }

    /**
     * \brief %Transform a uniformly distributed sample to the stored distribution
     *
     * \param[in] sampleValue
     *     An uniformly distributed sample on [0,1]
     * \param[out] pdf
     *     Probability value of the sample
     * \return
     *     The discrete index associated with the sample
     */
    inline size_t sample(Float sampleValue, Float &pdf) const {
        size_t index = sample(sampleValue);
        pdf = operator[](index);
        return index;
    }

    /**
     * \brief %Transform a uniformly distributed sample to the stored distribution
     *
     * The original sample is value adjusted so that it can be "reused".
     *
     * \param[in, out] sampleValue
     *     An uniformly distributed sample on [0,1]
     * \return
     *     The discrete index associated with the sample
     */
    inline size_t sampleReuse(Float &sampleValue) const {
        size_t index = sample(sampleValue);
        sampleValue = (sampleValue - m_cdf[index])
            / (m_cdf[index + 1] - m_cdf[index]);
        return index;
    }

    /**
     * \brief %Transform a uniformly distributed sample.
     *
     * The original sample is value adjusted so that it can be "reused".
     *
     * \param[in,out]
     *     An uniformly distributed sample on [0,1]
     * \param[out] pdf
     *     Probability value of the sample
     * \return
     *     The discrete index associated with the sample
     */
    inline size_t sampleReuse(Float &sampleValue, Float &pdf) const {
        size_t index = sample(sampleValue, pdf);
        sampleValue = (sampleValue - m_cdf[index])
            / (m_cdf[index + 1] - m_cdf[index]);
        return index;
    }

    /**
     * \brief Turn the underlying distribution into a
     * human-readable string format
     */
    std::string toString() const {
        std::ostringstream oss;
        oss << "DiscreteDistribution[sum=" << m_sum << ", normalized="
            << (int) m_normalized << ", cdf={";
        for (size_t i=0; i<m_cdf.size(); ++i) {
            oss << m_cdf[i];
            if (i != m_cdf.size()-1)
                oss << ", ";
        }
        oss << "}]";
        return oss.str();
    }

private:
    std::vector<Float> m_cdf;
    Float m_sum, m_normalization;
    bool m_normalized;
};


#endif //PMF_H__
