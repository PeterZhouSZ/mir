/*
 * Common Utilities - Distributed for "Mental Image Retrieval" implementation
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

#ifndef TASKSCHEDULER_H
#define TASKSCHEDULER_H

#include <boost/thread.hpp>
#include <boost/thread/condition.hpp>
#include <boost/function.hpp>
#include <list>
#include <stdexcept>

/** @addtogroup Codebase_Group
 *  @{
 */


class TaskScheduler;

class TaskGroup;

class Task
{
    public:
        enum State {
            STATE_IDLE,
            STATE_SCHEDULED,
            STATE_INPROGRESS,
            STATE_FINISHED
        };

        typedef boost::function<void(void)> TaskProc;
        Task();
        ~Task();
        void schedule(const TaskProc &proc, TaskScheduler &scheduler, TaskGroup *taskGroup = NULL);
        inline State getState() const { return m_state; }
    private:
        TaskGroup *m_taskGroup;
        volatile State m_state;
        TaskProc m_proc;
        friend class TaskScheduler;
};

class TaskGroup
{
    public:
        TaskGroup();
        ~TaskGroup();
        void clear();
        void add(const Task::TaskProc &proc, TaskScheduler &scheduler);
        uint32_t getRemainingUnfinishedTasks() const { return m_remainingUnfinishedTasks; }
    protected:
        volatile uint32_t m_remainingUnfinishedTasks;
        std::vector<Task*> m_tasks;
        friend class TaskScheduler;
};

class LockedTaskGroup : public TaskGroup
{
    public:
        void clear();
        void add(const Task::TaskProc &proc, TaskScheduler &scheduler);
    private:
        boost::mutex m_mutex;
        friend class TaskScheduler;
};

class TaskScheduler
{
    public:
        static void Init(unsigned helperThreads);
        static void Shutdown();
        TaskScheduler(unsigned helperThreads);
        ~TaskScheduler();

        void waitFor(Task *task);
        void waitFor(TaskGroup *taskGroup);

        static TaskScheduler &get() { if (m_mainInstance != NULL) return *m_mainInstance; throw std::runtime_error("TaskScheduler not initialized yet!"); }

        inline std::size_t getNumThreads() const { return m_helperThreads.size(); }
    private:
        void scheduleTask(Task *task);
        friend class Task;

        static TaskScheduler *m_mainInstance;

        volatile bool m_shutdown;
        typedef std::list<Task*> TaskList;
        TaskList m_scheduledTasks;

        #define TASK_SCHEDULER_USE_KERNEL

        #ifdef TASK_SCHEDULER_USE_KERNEL
        boost::condition m_wakeupCondition;
        boost::mutex m_wakeupMutex;

        boost::condition m_waitForCondition;
        boost::condition m_groupFinishedCondition;
        boost::mutex m_waitForMutex;
        #else
        volatile uint32_t m_taskListLock;
        volatile uint32_t m_numTasksEnqueued;
        #endif


        std::vector<boost::thread*> m_helperThreads;

        void helperThreadOperate();
};

/// @}

#endif // TASKSCHEDULER_H
