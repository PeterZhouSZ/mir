/*
 * C++ Convnet Implementation - Distributed for "Mental Image Retrieval" implementation
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

#include "StreamingTrainingDataSource.h"

#include "CocoHelpers.h"

namespace convnet {

StreamingTrainingDataSource::StreamingTrainingDataSource()
{
}

void StreamingTrainingDataSource::startThread()
{
    m_thread = std::thread(&StreamingTrainingDataSource::streamingThread, this);
}

void StreamingTrainingDataSource::stopThread()
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_shutdown = true;
        m_cacheEmpty.notify_one();
    }
    m_thread.join();
}

StreamingTrainingDataSource::~StreamingTrainingDataSource()
{
}

void StreamingTrainingDataSource::restart()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_restart = true;
    m_endOfData = false;
    m_cacheEmpty.notify_one();
}

bool StreamingTrainingDataSource::produceMinibatch(std::vector<TrainingDataProcessingBatch> &miniBatch)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    while (m_cache.empty()) {
        if (m_endOfData) {
            miniBatch.clear();
            return false;
        }
        m_cacheEmpty.notify_one();
        m_cacheEmpty.wait(lock);
    }
    miniBatch = std::move(m_cache.back());
    m_cache.pop_back();
    if (m_cache.empty())
        m_cacheEmpty.notify_one();
    return true;
}

void StreamingTrainingDataSource::streamingThread()
{
    std::vector<TrainingDataProcessingBatch> samples;
    while (true) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_restart) {
                m_restart = false;
                inThreadReset();
            }
        }
        
        fetch(samples);

        {
            std::unique_lock<std::mutex> lock(m_mutex);
            while (!m_cache.empty() && !m_shutdown) {
                m_cacheEmpty.wait(lock);
            }
            if (m_shutdown) {
                return;
            }

            if (samples.empty()) {
                m_endOfData = true; 
            } else {
                m_endOfData = false;
                m_cache.resize(samples.size() / m_processingBatchesPerMinibatch);
                std::cout << "Refilling cache with " << m_cache.size() << " minibatches!" << std::endl;
                for (unsigned i = 0; i < m_cache.size(); i++) {
                    m_cache[i].resize(m_processingBatchesPerMinibatch);
                    for (unsigned j = 0; j < m_processingBatchesPerMinibatch; j++)
                        m_cache[i][j] = std::move(samples[i*m_processingBatchesPerMinibatch+j]);
                }
            }
            
            m_cacheEmpty.notify_one();
        }
    }
}

}
