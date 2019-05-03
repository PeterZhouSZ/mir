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

#include "TaskScheduler.h"
#include <immintrin.h>

Task::Task()
{
    m_state = STATE_IDLE;
}

Task::~Task()
{
    assert((m_state == STATE_IDLE) || (m_state == STATE_FINISHED));
}

void Task::schedule(const TaskProc &proc, TaskScheduler &scheduler, TaskGroup *taskGroup)
{
    m_proc = proc;
    m_taskGroup = taskGroup;
    __sync_synchronize();
    scheduler.scheduleTask(this);
}


TaskGroup::TaskGroup()
{
    m_remainingUnfinishedTasks = 0;
}

TaskGroup::~TaskGroup()
{
    clear();
}

void TaskGroup::clear()
{
    for (unsigned i = 0; i < m_tasks.size(); i++) {
        delete m_tasks[i];
    }
    m_tasks.clear();
}

void TaskGroup::add(const Task::TaskProc &proc, TaskScheduler &scheduler)
{
    Task *task = new Task();
    m_tasks.push_back(task);
    __sync_fetch_and_add(&m_remainingUnfinishedTasks, 1);
    __sync_synchronize();
    task->schedule(proc, scheduler, this);
}


void LockedTaskGroup::clear()
{
    boost::mutex::scoped_lock lock(m_mutex);
    TaskGroup::clear();
}

void LockedTaskGroup::add(const Task::TaskProc &proc, TaskScheduler &scheduler)
{
    boost::mutex::scoped_lock lock(m_mutex);
    TaskGroup::add(proc, scheduler);
}


void TaskScheduler::Init(unsigned helperThreads)
{
    if (m_mainInstance != nullptr)
        throw std::runtime_error("Already initialized!");

    m_mainInstance = new TaskScheduler(helperThreads);
}

void TaskScheduler::Shutdown()
{
    delete m_mainInstance;
    m_mainInstance = nullptr;
}

TaskScheduler *TaskScheduler::m_mainInstance = NULL;


TaskScheduler::TaskScheduler(unsigned helperThreads)
{
    m_shutdown = false;

    #ifndef TASK_SCHEDULER_USE_KERNEL
    m_taskListLock = 0;
    m_numTasksEnqueued = 0;
    #endif

    m_helperThreads.resize(helperThreads);
    for (unsigned i = 0; i < helperThreads; i++) {
        m_helperThreads[i] = new boost::thread(boost::bind(&TaskScheduler::helperThreadOperate, this));
    }
}

TaskScheduler::~TaskScheduler()
{
#ifdef TASK_SCHEDULER_USE_KERNEL
    {
        boost::mutex::scoped_lock lock(m_wakeupMutex);
        m_shutdown = true;
    }
    m_wakeupCondition.notify_all();
#else
    m_shutdown = true;
#endif
    for (unsigned i = 0; i < m_helperThreads.size(); i++) {
        m_helperThreads[i]->join();
        delete m_helperThreads[i];
    }
}


void TaskScheduler::scheduleTask(Task *task)
{
    task->m_state = Task::STATE_SCHEDULED;
#ifdef TASK_SCHEDULER_USE_KERNEL
    {
        boost::mutex::scoped_lock lock(m_wakeupMutex);
        m_scheduledTasks.push_back(task);
        m_wakeupCondition.notify_one();
    }
#else
    while (__sync_lock_test_and_set(&m_taskListLock, 1) == 1)
        _mm_pause();

    m_scheduledTasks.push_back(task);
    m_numTasksEnqueued++;

    __sync_lock_release(&m_taskListLock);
#endif
}


void TaskScheduler::helperThreadOperate()
{
#ifdef TASK_SCHEDULER_USE_KERNEL
    while (true) {
        Task *task = NULL;
        {
            boost::mutex::scoped_lock lock(m_wakeupMutex);
            if (m_shutdown)
                break;
            if (m_scheduledTasks.size() > 0) {
                task = m_scheduledTasks.front();
                m_scheduledTasks.pop_front();
            } else
                m_wakeupCondition.wait(lock);
        }
        if (task != NULL) {
            task->m_state = Task::STATE_INPROGRESS;
            __sync_synchronize();
            task->m_proc();
            boost::mutex::scoped_lock lock(m_waitForMutex);
            task->m_state = Task::STATE_FINISHED;
            bool groupFinished = false;
            if (task->m_taskGroup != NULL)
                groupFinished = __sync_sub_and_fetch(&task->m_taskGroup->m_remainingUnfinishedTasks, 1) == 0;
            __sync_synchronize();
            m_waitForCondition.notify_all();
            if (groupFinished)
                m_groupFinishedCondition.notify_all();
        }
    }
#else
    while (true) {
        while (m_numTasksEnqueued == 0) {
            _mm_pause();
            if (m_shutdown)
                return;
        }

        Task *task = NULL;

        while (__sync_lock_test_and_set(&m_taskListLock, 1) == 1)
            _mm_pause();

        if (m_scheduledTasks.size() > 0) {
            task = m_scheduledTasks.front();
            m_scheduledTasks.pop_front();
            m_numTasksEnqueued--;
        }

        __sync_lock_release(&m_taskListLock);

        if (task != NULL) {
            task->m_state = Task::STATE_INPROGRESS;
            __sync_synchronize();
            task->m_proc();
            __sync_synchronize();
            task->m_state = Task::STATE_FINISHED;
            if (task->m_taskGroup != NULL)
                __sync_sub_and_fetch(&task->m_taskGroup->m_remainingUnfinishedTasks, 1) == 0;
        }
    }
#endif
}


void TaskScheduler::waitFor(Task *task)
{
#ifdef TASK_SCHEDULER_USE_KERNEL
    boost::mutex::scoped_lock lock(m_waitForMutex);
    while (task->m_state != Task::STATE_FINISHED) {
        m_waitForCondition.wait(lock);
    }
#else
    while (task->m_state != Task::STATE_FINISHED)
        _mm_pause();
#endif
}

void TaskScheduler::waitFor(TaskGroup *taskGroup)
{
#ifdef TASK_SCHEDULER_USE_KERNEL
    boost::mutex::scoped_lock lock(m_waitForMutex);

    while (taskGroup->getRemainingUnfinishedTasks() != 0) {
        m_groupFinishedCondition.wait(lock);
    }
#else
    while (taskGroup->getRemainingUnfinishedTasks() != 0)
        _mm_pause();
#endif
}
