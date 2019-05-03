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

#include "HistogramDataSource.h"
#include <tools/RasterImage.h>
#include <tools/BinaryFStream.h>


//#define DRAW_TRACKS

#ifdef DRAW_TRACKS
#include <tools/SvgComposer.h>
#include <boost/format.hpp>
#include <fstream>
#endif

#include <iostream>

void HistogramData::sampleFromDataset(eyeTracking::Dataset &dataset, unsigned rows, unsigned cols, bool sampleView, bool sampleRecall, unsigned numRecallAugmentations, bool mergeViewIntoRecall, bool normalize)
{
    std::mt19937 rng;
    std::normal_distribution<float> angleChange(0.0f, 0.0045f*2);
    std::normal_distribution<float> scaleChange(1.0f, 0.0045f*5*2);
    std::normal_distribution<float> startingOffset(0.0f, 0.025f);
    std::normal_distribution<float> startingScale(1.0f, 0.025f);

    subjects.resize(dataset.getTestSubjects().size());
    for (unsigned subj = 0; subj < dataset.getTestSubjects().size(); subj++)
        subjects[subj].name = dataset.getTestSubjects()[subj].getName();

    images.resize(dataset.getImages().size());
    for (unsigned i = 0; i < dataset.getImages().size(); i++)
        images[i].filename = dataset.getImages()[i].filename;

    
    for (unsigned subj = 0; subj < dataset.getTestSubjects().size(); subj++) {
        const eyeTracking::TestSubject &subject = dataset.getTestSubjects()[subj];

        for (unsigned i = 0; i < dataset.getImages().size(); i++) {
            std::cout << "Subject " << subj << " image " << i << std::endl;
            const eyeTracking::Dataset::Image &image = dataset.getImages()[i];

            unsigned imgWidth, imgHeight;
            RasterImage::getImageFileDimensions((dataset.getImageBasePath() / image.filename).string().c_str(), imgWidth, imgHeight);
            
#ifdef DRAW_TRACKS
            RasterImage img;
            img.loadFromFile((dataset.getImageBasePath() / image.filename).string().c_str());

            std::fstream file((boost::format("%04d_%02d_recall.svg") % i % subj).str().c_str(), std::fstream::out);
            SvgComposer svgComposer(file, 1900, 1200);
            svgComposer
                << SvgComposer::Image(img, (1900 - img.getWidth())/2.0f, (1200 - img.getHeight())/2.0f);

            SvgComposer::MultiLine multiLine[numRecallAugmentations+1];
            unsigned currentSegment = 0;

            
#endif            

            auto sampleEyeMovement = [&](const eyeTracking::EyeMovement *src, std::vector<Histogram::HistogramCell> &dst, bool augment, bool drift) {
                if (src != nullptr) {
                    
                    dst.resize(rows*cols);
                    memset(dst.data(), 0, dst.size()*sizeof(Histogram::HistogramCell));
                    
                    if (normalize && augment) throw std::runtime_error("Normalization and augmentation not supported!");
                    
                    if (!augment) {
						if (!normalize) {
							for (unsigned s = 0; s < src->getRawSamples().size(); s++) {                        
								int c = std::min<int>(std::max<int>(
									src->getRawSamples()[s].location[0] / 1900.0f * cols, 0), cols-1);
								int r = std::min<int>(std::max<int>(
									src->getRawSamples()[s].location[1] / 1200.0f * rows, 0), rows-1);
								
								dst[c+r*cols].timeSpend += 0.001f;
							}        
						} else {
#if 1
							Eigen::Vector2d mean;
							mean.setZero();
							for (const auto &s : src->getRawSamples())
								mean += s.location.cast<double>();
							mean /= src->getRawSamples().size();
							
							double var = 0.0f;
							for (const auto &s : src->getRawSamples()) {
								Eigen::Vector2f D = s.location - mean.cast<float>();
								var += D.dot(D);
							}
							var /= src->getRawSamples().size();

							for (const auto &s : src->getRawSamples()) {
								Eigen::Vector2f D = s.location - mean.cast<float>();
								D *= cols / 2.0f / (std::sqrt(var) * 2.0f);
								int c = std::min<int>(std::max<int>(
									D[0] + cols / 2.0f, 0), cols-1);
								int r = std::min<int>(std::max<int>(
									D[1] + rows / 2.0f, 0), rows-1);
								
								dst[c+r*cols].timeSpend += 0.001f;
							}
#else
							std::vector<float> allX, allY;
							allX.reserve(src->getRawSamples().size());
							allY.reserve(src->getRawSamples().size());
							for (const auto &s : src->getRawSamples()) {
								allX.push_back(s.location[0]);
								allY.push_back(s.location[1]);
							}
							std::sort(allX.begin(), allX.end());
							std::sort(allY.begin(), allY.end());
							
							Eigen::Vector2f median(allX[allX.size()/2], allY[allY.size()/2]);
							
							std::vector<float> allR;
							allR.reserve(src->getRawSamples().size());
							for (const auto &s : src->getRawSamples()) {
								Eigen::Vector2f D = s.location - median;
								allR.push_back(D.dot(D));
							}
							std::sort(allR.begin(), allR.end());
							
							float scale = cols/2.0f / std::sqrt(allR[allR.size()/10]);

							for (const auto &s : src->getRawSamples()) {
								Eigen::Vector2f D = s.location - median;
								D *= scale;
								int c = std::min<int>(std::max<int>(
									D[0] + cols / 2.0f, 0), cols-1);
								int r = std::min<int>(std::max<int>(
									D[1] + rows / 2.0f, 0), rows-1);
								
								dst[c+r*cols].timeSpend += 0.001f;
							}
#endif
							
#if 0
							std::cout << "Hist: " << std::endl;
							for (unsigned r = 0; r < rows; r++) {
								for (unsigned c = 0; c < cols; c++) {
									int i = std::min<int>(9, dst[c+r*cols].timeSpend * 100);
									std::cout << i;
								}
								std::cout << std::endl;
							}
							std::cout << std::endl;
							std::cout << std::endl;
#endif
						}
                    } else {
                        Eigen::Matrix3f transform = Eigen::Matrix3f::Identity();
                        transform(0, 2) = startingOffset(rng) * (drift?1.0f:0.1f); // less for view!
                        transform(1, 2) = startingOffset(rng) * (drift?1.0f:0.1f);
                        transform(0, 0) = 
                        transform(1, 1) = (drift?startingScale(rng):1.0f);
                        Eigen::Vector2f lastTransformed;
                        for (unsigned s = 0; s < src->getRawSamples().size(); s++) {              
                            
                            Eigen::Vector2f l = transform.block<2, 2>(0, 0) * src->getRawSamples()[s].location + transform.block<2, 1>(0, 2);
                            
#ifdef DRAW_TRACKS
                            if (s > 0) {
                                if (currentSegment == 0) {
                                    multiLine[currentSegment].addSegment(
                                        src->getRawSamples()[s-1].location[0], src->getRawSamples()[s-1].location[1],
                                        src->getRawSamples()[s].location[0], src->getRawSamples()[s].location[1]
                                    );
                                }
                                multiLine[currentSegment+1].addSegment(
                                    lastTransformed[0], lastTransformed[1],
                                    l[0], l[1]
                                );
                            }
#endif 
                            
                            int c = std::min<int>(std::max<int>(
                                l[0] / 1900.0f * cols, 0), cols-1);
                            int r = std::min<int>(std::max<int>(
                                l[1] / 1200.0f * rows, 0), rows-1);
                            
                            dst[c+r*cols].timeSpend += 0.001f;
                            
                            if ((s % 50 == 0) && drift) {
                                Eigen::Matrix3f pivotBefore = Eigen::Matrix3f::Identity();
                                Eigen::Matrix3f pivotAfter = Eigen::Matrix3f::Identity();
                                
                                pivotBefore(0, 2) = src->getRawSamples()[s].location[0];
                                pivotBefore(1, 2) = src->getRawSamples()[s].location[1];
                                pivotAfter(0, 2) = -src->getRawSamples()[s].location[0];
                                pivotAfter(1, 2) = -src->getRawSamples()[s].location[1];
                                
                                Eigen::Matrix3f change = Eigen::Matrix3f::Identity();
                                float scale = scaleChange(rng);
                                float angle = angleChange(rng);
                                change(0, 0) = std::cos(angle) * scale;
                                change(1, 0) = std::sin(angle) * scale;
                                change(0, 1) = -std::sin(angle) * scale;
                                change(1, 1) = std::cos(angle) * scale;
                                
                                
                                transform *= pivotBefore * change * pivotAfter;
                            }
                            lastTransformed = l;
                        }        
                    }
                }
            };
            
            for (unsigned k = 0; k < std::max(1u, numRecallAugmentations); k++) {
#ifdef DRAW_TRACKS
                currentSegment = k;
#endif 
                Histogram histogram;
                histogram.rows = rows;
                histogram.cols = cols;
                histogram.category = image.category;
                histogram.image = i;
                histogram.subject = subj;
                if (sampleView && !mergeViewIntoRecall) {
                    sampleEyeMovement(subject.getViewingSequenceForImg(i), histogram.view, false, false);//k > 0, false);
                    if (histogram.view.empty()) continue;
                }
                if (sampleRecall) {
                    sampleEyeMovement(subject.getRecallSequenceForImg(i), histogram.recall, k > 0, true);
                    if (histogram.recall.empty()) continue;
                }
				histogram.augmented = k > 0;

                histograms.push_back(std::move(histogram));
                
                if (mergeViewIntoRecall && sampleView) {
                    Histogram histogram;
                    histogram.rows = rows;
                    histogram.cols = cols;
                    histogram.category = image.category;
                    histogram.image = i;
                    histogram.subject = subj;
                    if (sampleView) {
                        sampleEyeMovement(subject.getViewingSequenceForImg(i), histogram.recall, k > 0, true);//false);
                        if (histogram.recall.empty()) continue;
                    }
                    histogram.augmented = k > 0;

                    histograms.push_back(std::move(histogram));
                }
            }
#ifdef DRAW_TRACKS
            for (unsigned k = 0; k < numRecallAugmentations+1; k++) {
                if (k == 0)
                    multiLine[k].setStroke("red").setStrokeWidth(1.0f);
                else
                    multiLine[k].setStroke("blue").setStrokeWidth(1.0f);
                svgComposer
                    << multiLine[k];            
            }
#endif 
        }
    }
    
}

void HistogramData::writeToFile(const char *filename) const
{
    tools::BinaryFStream stream(filename, std::fstream::out);
    stream << (std::uint32_t) histograms.size();
    for (const auto &s : histograms) {
        stream 
            << s.rows
            << s.cols
            << s.view
            << s.recall
            << s.image
            << s.subject
            << s.category
			<< s.augmented;
    }
    stream << (std::uint32_t) images.size();
    for (const auto &i : images)
        stream << i.filename;
    stream << (std::uint32_t) subjects.size();
    for (const auto &s : subjects)
        stream << s.name;
}

void HistogramData::readFromFile(const char *filename)
{
    tools::BinaryFStream stream(filename, std::fstream::in);

    std::uint32_t count;
    stream >> count;
    histograms.resize(count);
    for (auto &s : histograms) {
        stream 
            >> s.rows
            >> s.cols
            >> s.view
            >> s.recall
            >> s.image
            >> s.subject
            >> s.category
			>> s.augmented;
    }
    stream >> count;
    images.resize(count);
    for (auto &i : images)
        stream >> i.filename;
    stream >> count;
    subjects.resize(count);
    for (auto &s : subjects)
        stream >> s.name;
}

    
void HistogramDataSource::takeSubjects(HistogramData &data, const std::set<unsigned> &subjects)
{
    for (unsigned i = 0; i < data.histograms.size();) {
        if (subjects.find(data.histograms[i].subject) != subjects.end()) {
            m_histograms.push_back(std::move(data.histograms[i]));
            data.histograms[i] = std::move(data.histograms.back());
            data.histograms.pop_back();
        } else i++;
    }    
    restart();
}


void HistogramDataSource::takeImages(HistogramData &data, const std::set<unsigned> &images)
{
    for (unsigned i = 0; i < data.histograms.size();) {
        if (images.find(data.histograms[i].image) != images.end()) {
            m_histograms.push_back(std::move(data.histograms[i]));
            data.histograms[i] = std::move(data.histograms.back());
            data.histograms.pop_back();
        } else i++;
    }    
    restart();
}

void HistogramDataSource::takeRandom(HistogramData &data, unsigned count)
{
    for (unsigned i = 0; i < count; i++) {
        std::uniform_int_distribution<unsigned> randomData(0, data.histograms.size()-1);
        unsigned idx = randomData(m_rng);

        m_histograms.push_back(std::move(data.histograms[idx]));
        data.histograms[idx] = std::move(data.histograms.back());
        data.histograms.pop_back();
    }    
    restart();
}

void HistogramDataSource::takeAll(HistogramData &data)
{
    m_histograms = std::move(data.histograms);
    restart();
}

void HistogramDataSource::restart()
{
    m_nextScanPath = 0;
    std::shuffle(m_histograms.begin(), m_histograms.end(), m_rng);
}

void HistogramDataSource::uploadHistogram(convnet::InputStagingBuffer &miniBatch, const char *histogramDataPtr, unsigned stride, unsigned output, unsigned numInstances, unsigned instanceOffset)
{
    if (output == ~0u) return;
            
    convnet::TensorData &data = dynamic_cast<convnet::TensorData&>(*miniBatch.outputs[output]);

    const std::vector<HistogramData::Histogram::HistogramCell> &hist = (const std::vector<HistogramData::Histogram::HistogramCell> &)(*histogramDataPtr);

    if (data.getValues().getSize().width * data.getValues().getSize().height != hist.size())
        throw std::runtime_error("Invalid scanpath width * height!");
    if (data.getValues().getSize().depth != 1)
        throw std::runtime_error("Invalid scanpath depth!");
    if (data.getValues().getSize().channels != 1)
        throw std::runtime_error("Invalid scanpath number of channels!");


    convnet::MappedTensor mappedData = data.getValues().lock(numInstances == data.getValues().getNumInstances());
    for (unsigned i = 0; i < numInstances; i++) {
        const std::vector<HistogramData::Histogram::HistogramCell> &histogramData = (const std::vector<HistogramData::Histogram::HistogramCell> &)*(histogramDataPtr + stride * i);

        for (unsigned r = 0; r < data.getValues().getSize().height; r++) {
            for (unsigned c = 0; c < data.getValues().getSize().width; c++) {
                mappedData.get<float>(c, r, 0, 0, instanceOffset+i) = histogramData[c+r*data.getValues().getSize().width].timeSpend;
            }
        }
    }
    data.getValues().unlock(mappedData);
}
    
void HistogramDataSource::uploadInt(convnet::InputStagingBuffer &miniBatch, const char *intDataPtr, unsigned stride, unsigned output, unsigned numInstances, unsigned instanceOffset)
{
    if (output == ~0u) return;
            
    convnet::TensorData &data = dynamic_cast<convnet::TensorData&>(*miniBatch.outputs[output]);

    if (data.getValues().getSize().width != 1)
        throw std::runtime_error("Invalid width!");
    if (data.getValues().getSize().height != 1)
        throw std::runtime_error("Invalid height!");
    if (data.getValues().getSize().depth != 1)
        throw std::runtime_error("Invalid depth!");
    if (data.getValues().getSize().channels != 1)
        throw std::runtime_error("Invalid number of channels!");

    convnet::MappedTensor mappedData = data.getValues().lock(numInstances == data.getValues().getNumInstances());
    for (unsigned i = 0; i < numInstances; i++) {
        const int &intValue = (const int &)*(intDataPtr + stride * i);
        
        mappedData.get<float>(0, 0, 0, 0, instanceOffset+i) = intValue;
    }
    //std::cout << mappedData.get<float>(0, 0, 0, 0, 0) << std::endl;
    data.getValues().unlock(mappedData);
}
        
    
void HistogramDataSource::uploadConst(convnet::InputStagingBuffer &miniBatch, float value, unsigned output, unsigned numInstances, unsigned instanceOffset) 
{
    if (output == ~0u) return;
            
    convnet::TensorData &data = dynamic_cast<convnet::TensorData&>(*miniBatch.outputs[output]);

    if (data.getValues().getSize().width != 1)
        throw std::runtime_error("Invalid width!");
    if (data.getValues().getSize().height != 1)
        throw std::runtime_error("Invalid height!");
    if (data.getValues().getSize().depth != 1)
        throw std::runtime_error("Invalid depth!");
    if (data.getValues().getSize().channels != 1)
        throw std::runtime_error("Invalid number of channels!");

    convnet::MappedTensor mappedData = data.getValues().lock(numInstances == data.getValues().getNumInstances());
    for (unsigned i = 0; i < numInstances; i++) {
        mappedData.get<float>(0, 0, 0, 0, instanceOffset+i) = value;
    }
    data.getValues().unlock(mappedData);
}   


bool HistogramDataSource::produceMinibatch(convnet::InputStagingBuffer &miniBatch)
{
    unsigned numInstances = dynamic_cast<convnet::TensorData&>(*miniBatch.outputs[0]).getValues().getNumInstances();
    if (m_nextScanPath+numInstances > m_histograms.size()) return false;
    
    uploadHistogram(miniBatch, (const char*)&m_histograms[m_nextScanPath].view, sizeof(HistogramData::Histogram), viewOutput, numInstances, 0);
    uploadHistogram(miniBatch, (const char*)&m_histograms[m_nextScanPath].recall, sizeof(HistogramData::Histogram), recallOutput, numInstances, 0);
    uploadInt(miniBatch, (const char*)&m_histograms[m_nextScanPath].image, sizeof(HistogramData::Histogram), imageOutput, numInstances, 0);
    uploadInt(miniBatch, (const char*)&m_histograms[m_nextScanPath].subject, sizeof(HistogramData::Histogram), subjectOutput, numInstances, 0);
    uploadInt(miniBatch, (const char*)&m_histograms[m_nextScanPath].category, sizeof(HistogramData::Histogram), categoryOutput, numInstances, 0);

    if ((opposingViewOutput != ~0u) || (opposingRecallOutput != ~0u)) {   
        for (unsigned i = 0; i < numInstances; i++) {
            unsigned otherSequenceIndex;
            std::uniform_int_distribution<unsigned> rndSeq(0, m_histograms.size()-1);
            do {
                otherSequenceIndex = rndSeq(m_rng);
            } while ((otherSequenceIndex == m_nextScanPath+i) || 
                    (opposingSequenceFromDifferentImage && (m_histograms[otherSequenceIndex].image == m_histograms[m_nextScanPath+i].image)) ||
                    (opposingSequenceFromSameSubject && (m_histograms[otherSequenceIndex].subject != m_histograms[m_nextScanPath+i].subject)));

            uploadHistogram(miniBatch, (const char*)&m_histograms[otherSequenceIndex].view, sizeof(HistogramData::Histogram), opposingViewOutput, 1, i);
            uploadHistogram(miniBatch, (const char*)&m_histograms[otherSequenceIndex].recall, sizeof(HistogramData::Histogram), opposingRecallOutput, 1, i);
        }
    }    
        
    uploadConst(miniBatch, 0, outputZero, numInstances, 0);
    uploadConst(miniBatch, 1, outputOne, numInstances, 0);
    
    m_nextScanPath += numInstances;

    return true;
    
}

    
    
    
