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

#ifndef STREAMINGTRAININGDATASOURCE_H
#define STREAMINGTRAININGDATASOURCE_H

#include "TrainingDataSource.h"


#include <vector>

#include <thread>
#include <mutex>
#include <condition_variable>

namespace convnet {

class StreamingTrainingDataSource : public TrainingDataSource
{
    public:
        StreamingTrainingDataSource();
        virtual ~StreamingTrainingDataSource();
        
        virtual void restart() override;
        virtual bool produceMinibatch(InputStagingBuffer &miniBatch) override;
    protected:
        std::mutex m_mutex;
        std::thread m_thread;
        std::condition_variable m_cacheEmpty;
        bool m_endOfData = false;
        bool m_restart = false;
        bool m_shutdown = false;
        
        
        
        std::vector<std::vector<TrainingDataProcessingBatch>> m_cache;
        
        void startThread();
        void stopThread();

        virtual void inThreadReset() = 0;
        virtual void fetch(std::vector<TrainingDataProcessingBatch> &samples) = 0;
        void streamingThread();
}
;
}

#endif // STREAMINGTRAININGDATASOURCE_H
