/*
 * Mental Image Retrieval - C++ Machine Learning implementation
 * Copyright (C) 2017-2019  Andreas Ley <mail@andreas-ley.com>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef HISTOGRAMDATASOURCE_H
#define HISTOGRAMDATASOURCE_H

#include <eyeTracking/Dataset.h>

#include <convnet/training/TrainingDataSource.h>

#include <set>
#include <random>

struct HistogramData
{
    struct Image {
        std::string filename;
    };
    struct Subject {
        std::string name;
    };
    
    struct Histogram {
        unsigned rows, cols;
        struct HistogramCell {
            float timeSpend;
        };
        std::vector<HistogramCell> view;
        std::vector<HistogramCell> recall;
        unsigned image;
        unsigned subject;
        unsigned category;
		bool augmented;
    };
    std::vector<Histogram> histograms;
    std::vector<Image> images;
    std::vector<Subject> subjects;

    void sampleFromDataset(eyeTracking::Dataset &dataset, unsigned rows, unsigned cols, bool sampleView, bool sampleRecall, unsigned numRecallAugmentations = 0, bool mergeViewIntoRecall = false, bool normalize = false);
    
    void writeToFile(const char *filename) const;
    void readFromFile(const char *filename);
};


class HistogramDataSource : public convnet::TrainingDataSource
{
    public:
        void takeSubjects(HistogramData &data, const std::set<unsigned> &subjects);
        void takeImages(HistogramData &data, const std::set<unsigned> &images);
        void takeRandom(HistogramData &data, unsigned count);
        void takeAll(HistogramData &data);
        
        virtual void restart() override;
        virtual bool produceMinibatch(convnet::InputStagingBuffer &miniBatch) override;

        void uploadHistogram(convnet::InputStagingBuffer &miniBatch, const char *histogramDataPtr, unsigned stride, unsigned output, unsigned numInstances, unsigned instanceOffset);
        void uploadInt(convnet::InputStagingBuffer &miniBatch, const char *intDataPtr, unsigned stride, unsigned output, unsigned numInstances, unsigned instanceOffset);
        void uploadConst(convnet::InputStagingBuffer &miniBatch, float value, unsigned output, unsigned numInstances, unsigned instanceOffset);
        inline const std::vector<HistogramData::Histogram> &getHistograms() const { return m_histograms; }

        unsigned viewOutput = ~0u;
        unsigned recallOutput = ~0u;
        unsigned imageOutput = ~0u;
        unsigned subjectOutput = ~0u;
        unsigned categoryOutput = ~0u;
        unsigned outputZero = ~0u;
        unsigned outputOne = ~0u;

        unsigned opposingViewOutput = ~0u;
        unsigned opposingRecallOutput = ~0u;
        bool opposingSequenceFromDifferentImage = true;
        bool opposingSequenceFromSameSubject = false;

        bool m_augment = false;
    protected:
        unsigned m_nextScanPath = 0;
        std::mt19937 m_rng;
        std::vector<HistogramData::Histogram> m_histograms;        
};

#endif // HISTOGRAMDATASOURCE_H
